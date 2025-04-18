import logging
from typing import List, Dict, Optional, Tuple

import torch
from torch.cuda import Event, Stream


class OptimizationOrchestrator:
    """This orchestrator assumes that the model calls all layers sequentially, without skipping any layers.
    The start_forward is called before the start of the main layer payload. Methods start_optimization and
    end_optimization are called at every layer in a separate CUDA stream.

    The core idea is to interleave the optimization with the forward calls. Each optimization for each layer should have
    a separate dedicated CUDA stream.

    The core assumption is that the forward pass on the current layer is blocked by the optimization on the current
    layer from the previous call of the forward pass. Additionally, optimization is assumed to be blocked by the
    forward pass on the current layer within current call.
    """
    def __init__(self):
        assert torch.cuda.is_available()

        # timestep -> layer id -> CUDA Event
        # keeps track of handed out events that need to be completed
        self.incomplete_optimization: Dict[int, Dict[int, Event]] = {}
        self.incomplete_forward: Dict[int, Dict[int, Event]] = {}

        # layer id -> timestep
        # indicates the last timestep for each optimization process set up (but optimization may have not finished!)
        self.optimization_progress: List[int] = []

        # (timestep, layer_id)
        # indicates last layer that was set up for forward call (but may have not finished!)
        self.forward_progress: Tuple[int, int] = (-1, -1)

    def increment_optimization_progress(self, layer_id: int):
        """This is called at the start of each layer optimization payload."""
        while len(self.optimization_progress) < layer_id:
            self.optimization_progress.append(-1)
        self.optimization_progress[layer_id] += 1

    @staticmethod
    def create_event(storage: Dict[int, Dict[int, Event]], timestep: int, layer_id: int):
        if timestep < 0 or layer_id < 0:
            raise ValueError(f'Incorrect event creation arguments: (t={timestep}, l={layer_id})')

        if timestep not in storage:
            storage[timestep] = {}

        if layer_id not in storage[timestep]:
            storage[timestep][layer_id] = Event()

        return storage[timestep][layer_id]

    @classmethod
    def claim_event(
            cls,
            storage: Dict[int, Dict[int, Event]],
            timestep: int,
            layer_id: int,
            stream: Optional[Stream] = None
    ):
        if timestep < 0 or layer_id < 0:
            raise ValueError(f'Incorrect event claim arguments: (t={timestep}, l={layer_id})')

        cls.create_event(storage, timestep, layer_id).wait(stream)

    @staticmethod
    def cleanup(storage: Dict[int, Dict[int, Event]]):
        to_delete = []
        for timestep in storage:
            for layer_id in storage[timestep]:
                if storage[timestep][layer_id].query():
                    to_delete.append((timestep, layer_id))
                    
        for timestep, layer_id in to_delete:
            del storage[timestep][layer_id]
            if not len(storage[timestep]):
                del storage[timestep]

    def start_forward(self, layer_id: int, stream: Optional[Stream] = None, *, strict: bool = False) -> int:
        """
        This is called before the start of the forward payload.

        IMPORTANT: pass the same stream that processes the forward pass of the model.

        Records all completed forward events on the provided stream.

        Waits for the optimization end event. Returns the current tracked timestep.
        """
        prev_timestep, prev_layer_id = self.forward_progress

        if layer_id <= prev_layer_id or prev_timestep == -1:
            # the previous timestep is finished
            self.complete_events_at_timestep(self.incomplete_forward, prev_timestep, stream=stream, strict=strict)

            timestep = prev_timestep + 1

            # clean up once per timestep
            self.cleanup(self.incomplete_optimization)
            self.cleanup(self.incomplete_forward)
        elif prev_timestep < 0:
            # the first call
            timestep = 0
        else:
            timestep = prev_timestep

            # previous layers are finished
            for lid in range(prev_layer_id, layer_id):
                self.complete_events_at_timestep(
                    self.incomplete_forward, prev_timestep,
                    layer_id=lid,
                    stream=stream,
                    strict=strict
                )

        self.forward_progress = (timestep, layer_id)

        self.create_event(self.incomplete_forward, timestep, layer_id)

        if timestep <= 0:
            return timestep  # no optimization to wait for

        # we can start the forward once the optimization from the previous step is complete
        if timestep > 0:
            self.claim_event(self.incomplete_optimization, timestep - 1, layer_id, stream=stream)
        return timestep

    def start_optimization(self, layer_id: int, stream: Optional[Stream] = None) -> int:
        """
        This is called before the start of the optimization payload.

        IMPORTANT: pass the same stream that processes the optimization payload.

        Use end_optimization method to record optimization completion.
        """
        self.increment_optimization_progress(layer_id)
        timestep = self.optimization_progress[layer_id]

        self.create_event(self.incomplete_optimization, timestep, layer_id)

        # we can start optimization once the forward of current timestep is complete
        self.claim_event(self.incomplete_forward, timestep, layer_id, stream=stream)
        return timestep

    def end_optimization(
            self, timestep: int, layer_id: int,
            stream: Optional[Stream] = None,
            *,
            strict: bool = False
    ):
        """
        This is called after the end of the optimization payload.

        IMPORTANT: pass the same stream that processes the optimization payload.
        """
        self.complete_events_at_timestep(
            self.incomplete_optimization, timestep,
            layer_id=layer_id,
            stream=stream,
            strict=strict
        )

    @staticmethod
    def complete_events_at_timestep(
            storage: Dict[int, Dict[int, Event]], timestep: int,
            layer_id: Optional[int] = None,
            stream: Optional[Stream] = None,
            strict: bool = False
    ):
        def handle_not_found():
            unexpected_msg = f'Unexpected call to complete (t={timestep}, l={layer_id})'
            if strict:
                raise RuntimeError(unexpected_msg)
            else:
                logging.warning(unexpected_msg)

        if timestep < 0:
            return

        if timestep not in storage:
            handle_not_found()
            return

        if layer_id is None:
            for layer_id, event in storage[timestep].items():
                event.record(stream)
        else:
            if layer_id not in storage[timestep]:
                handle_not_found()
                return

            storage[timestep][layer_id].record(stream)
