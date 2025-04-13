from typing import Iterable, Optional, Dict, Any, List, Tuple, ContextManager

import torch
from transformers import DynamicCache


class DudContextManager(ContextManager):
    def __exit__(self, __exc_type, __exc_value, __traceback):
        # If an exception is raised, re-raise it
        if __exc_type is not None:
            raise __exc_value
        return False


class IterativeReduceKVBiasCache(DynamicCache):
    def __init__(
            self,
            _distributed_cache_data: Iterable = None,
            *,
            max_variance_increase: float = 0.0,
            protect_first: int = 0
    ) -> None:
        super().__init__(_distributed_cache_data)

        self.curr_length: List[int] = []

        # (B, H, S)
        self.cluster_sizes: List[torch.Tensor] = []
        self.cached_sq_key_norms: List[torch.Tensor] = []
        self.last_available_idx: List[torch.Tensor] = []

        self.per_head_added_variance: List[torch.Tensor] = []
        self.max_variance_increase = max_variance_increase
        self.protect_first = protect_first

        # Overview of the synchronization algorithm:
        # 1. We need to start optimizing the layer before exiting update method
        # 2. We need to make sure the model is not using the cache while it is optimizing
        # 3. We need to make sure the optimization is complete before starting update
        #
        #
        # At time step t, update[i] call will:
        # 1. Wait for optimize_stream[i] stream to synchronize (equivalent to finishing
        #   the optimization from the t - 1 step)
        # 2. Add new KV, launch the optimization
        #   2.1 Record the self.optimization_start_event[i]
        #   2.2 During optimization, do not modify cache while optimization for the layer (i + 1) mod L has not started.
        #       This is done by waiting on self.optimization_start_event[(i + 1) % L] event.
        #   2.3 Reset the self.optimization_start_event[(i + 1) % L] event. Double record will not happen since
        #       layer i at t + 1 will wait for the optimization from t to finish.
        # 3. Return from update <- this is done way before 2 is complete
        self.optimize_streams: List[torch.cuda.Stream] = []
        self.optimization_start_event: List[torch.cuda.Event] = []

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states. A layer index can be optionally passed."""
        # TODO: deprecate this function in favor of `cache_position`
        is_empty_layer = (
            len(self.key_cache) == 0  # no cache in any layer
            or len(self.key_cache) <= layer_idx  # skipped `layer_idx` and hasn't run a layer with cache after it
            or len(self.key_cache[layer_idx]) == 0  # the layer has no cache
        )

        if is_empty_layer:
            return 0

        return self.key_cache[layer_idx].shape[-2]

    def _init_empty(
            self,
            size: tuple[int, int, int],
            device: torch.device | str,
            layer_idx: int
    ) -> None:
        """WARNING: call this AFTER super call!"""
        batch_size, n_heads, seq_length = size

        self.curr_length[layer_idx] = seq_length

        self.cluster_sizes[layer_idx] = torch.ones(
            size=(batch_size, n_heads, seq_length),
            device=device,
            dtype=torch.long
        )

        self.per_head_added_variance[layer_idx] = torch.zeros(
            size=(batch_size, n_heads),
            device=device,
            dtype=torch.long
        )

        self.cached_sq_key_norms[layer_idx] = torch.sum(self.key_cache[layer_idx] ** 2, dim=-1)
        self.last_available_idx[layer_idx] = torch.full(
            size=(batch_size, n_heads),
            fill_value=seq_length,
            device=device,
            dtype=torch.long
        )

        if torch.cuda.is_available():
            self.optimize_streams[layer_idx] = torch.cuda.Stream(priority=-1)
            self.optimization_start_event[layer_idx] = torch.cuda.Event()

    def optimize_last_entry(self, layer_idx: int) -> None:
        # First order of business: indicate to the optimization process from the previous layer that the model
        #   forward pass is done with KV cache from that layer. If CUDA is not available, everything is sequential,
        #   so there is no need in this.
        if self.optimization_start_event[layer_idx] is not None and self.optimize_streams[layer_idx] is not None:
            self.optimization_start_event[layer_idx].record(self.optimize_streams[layer_idx])

        if len(self.key_cache) < layer_idx or not len(self.key_cache[layer_idx]):
            return  # empty cache

        if self.get_seq_length(layer_idx) < 2:
            return  # nothing to optimize

        max_available_idx = torch.max(self.last_available_idx[layer_idx])
        if max_available_idx <= self.protect_first:
            return  # do not merge the first self.protect_first from merging into anything

        manager = DudContextManager()
        if self.optimize_streams[layer_idx] is not None:
            manager = torch.cuda.stream(self.optimize_streams[layer_idx])

        with manager:
            # Two possibilities:
            # 1) CUDA is not available and this call will block the forward execution of the model
            # 2) CUDA is available and this call will be executed right after the optimization from the previous
            #   time step on this layer is complete. This is expected to happen because we are using layer-specific
            #   stream and CUDA executes operations FIFO.
            batch_size, n_heads, _, head_dims = self.key_cache[layer_idx].shape
            dtype = self.key_cache[layer_idx].dtype

            batch_idx = torch.arange(batch_size).unsqueeze(1)
            head_idx = torch.arange(n_heads).unsqueeze(0)

            last_entry_idx = self.last_available_idx[layer_idx] - 1

            # select last entry to optimize
            key_states = torch.take_along_dim(self.key_cache[layer_idx], last_entry_idx[:, :, None, None], dim=-2)
            value_states = torch.take_along_dim(self.value_cache[layer_idx], last_entry_idx[:, :, None, None], dim=-2)
            new_norm = torch.take_along_dim(
                self.cached_sq_key_norms[layer_idx],
                last_entry_idx[:, :, None],
                dim=-1
            )[:, :, :, None].transpose(-1, -2)  # move seq dimension to the back

            # remove the last entry from current cache
            new_cluster_sizes = torch.clone(self.cluster_sizes[layer_idx])  # clone to not affect the cache state
            new_cluster_sizes[batch_idx, head_idx, last_entry_idx] = 0

            # here S_new will be 1

            # examine for matches in the cache by computing pairwise square distance to each element in th cache
            old_norm = self.cached_sq_key_norms[layer_idx][:, :, :, None]  # (B, H, S_old, 1)
            cross_term = torch.matmul(self.key_cache[layer_idx], key_states.transpose(-1, -2))  # (B, H, S_old, S_new)
            sq_dist = new_norm + old_norm - 2 * cross_term  # (B, H, S_old, S_new)

            # CAREFUL: this value will be 0 for padding
            variance_weight = new_cluster_sizes / (new_cluster_sizes + 1)
            vw_broadcast = variance_weight[:, :, :, None]  # broadcast over new seq

            # depends on cluster sizes and distance to new element
            variance_increase = vw_broadcast * sq_dist

            # protect from selecting padding as merging candidate (still can happen if there is no other candidate)
            # padding is determined by 0 in cluster size
            protect_range = 1 / variance_weight  # this will make 0 into inf
            protect_range = torch.clamp(protect_range - 100, 0)  # inf - 100 = inf; vw \in [1/2, 1)
            protect_range[:, :, :self.protect_first] = torch.inf
            pr_broadcast = protect_range[:, :, :, None]  # broadcast over new seq
            variance_increase = variance_increase + pr_broadcast

            # (B, H, S_new) and (B, H, S_new)
            min_variance_increase, min_idx = torch.min(variance_increase, dim=-2)
            merge_condition = min_variance_increase < self.max_variance_increase

            invalid_variance = torch.isinf(min_variance_increase) | torch.isnan(min_variance_increase)
            merge_condition = merge_condition & ~invalid_variance

            # protect against merging with padding keys
            merge_condition = merge_condition & (min_idx < last_entry_idx[:, :, None])

            # UPDATE KV CACHE STATE [RACE CONDITION WARNING]
            # We need to wait for the forward layer call to finish working with the current state of KV cache of this
            #   layer. The certain indication of this is that the optimization for the next layer has started.
            #   No need to wait if no CUDA is available as everything is sequential this way.
            if self.optimization_start_event[layer_idx] is not None and self.optimize_streams[layer_idx] is not None:
                next_layer_idx = (layer_idx + 1) % len(self.key_cache)  # TODO
                self.optimization_start_event[next_layer_idx].wait(self.optimize_streams[layer_idx])

                # reset for the next call of this function
                self.optimization_start_event[next_layer_idx] = torch.cuda.Event()

            # remove the last entry from the current cache
            self.cluster_sizes[layer_idx][batch_idx, head_idx, last_entry_idx] = 0
            self.last_available_idx[layer_idx] -= 1

            # update current added variance
            added_variance = torch.sum(min_variance_increase * merge_condition, dim=-1)
            self.per_head_added_variance[layer_idx] = self.per_head_added_variance[layer_idx] + added_variance

            # we will update KV cache at [batch_idx, head_idx, seq_idx] locations
            # S_new = 1
            seq_idx = min_idx.squeeze(-1)
            merge_condition = merge_condition.squeeze(-1)
            key_states = key_states.squeeze(-2)
            value_states = value_states.squeeze(-2)

            # update existing cluster sizes
            selected_cluster_sizes = self.cluster_sizes[layer_idx][batch_idx, head_idx, seq_idx]
            self.cluster_sizes[layer_idx][batch_idx, head_idx, seq_idx] = selected_cluster_sizes + merge_condition

            # update cluster centroids, inertia is computed based on the previous values
            inertia = 1 / (selected_cluster_sizes + 1)
            inertia = inertia.to(dtype=dtype)

            # update keys
            selected_cluster_centroids = self.key_cache[layer_idx][batch_idx, head_idx, seq_idx]
            change = inertia[:, :, None] * (key_states - selected_cluster_centroids)
            new_selected_clusters = selected_cluster_centroids + change * merge_condition[:, :, None]
            self.key_cache[layer_idx][batch_idx, head_idx, seq_idx] = new_selected_clusters

            # update cached norms for updated keys
            new_sq_key_norms = torch.sum(new_selected_clusters ** 2, dim=-1)
            self.cached_sq_key_norms[layer_idx][batch_idx, head_idx, seq_idx] = new_sq_key_norms

            # update values
            selected_cluster_centroids = self.value_cache[layer_idx][batch_idx, head_idx, seq_idx]
            change = inertia[:, :, None] * (value_states - selected_cluster_centroids)
            new_selected_clusters = selected_cluster_centroids + change * merge_condition[:, :, None]
            self.value_cache[layer_idx][batch_idx, head_idx, seq_idx] = new_selected_clusters

            last_idx = self.last_available_idx[layer_idx]
            not_merged = ~merge_condition

            self.cluster_sizes[layer_idx][batch_idx, head_idx, last_idx] = not_merged.to(dtype=torch.long)
            self.key_cache[layer_idx][batch_idx, head_idx, last_idx] = key_states * not_merged[:, :, None]
            self.cached_sq_key_norms[layer_idx][batch_idx, head_idx, last_idx] = new_sq_key_norms * not_merged
            self.value_cache[layer_idx][batch_idx, head_idx, last_idx] = value_states * not_merged[:, :, None]
            self.last_available_idx[layer_idx] += not_merged

    def _resize_if_needed(self, layer_idx: int, additional_length: int):
        current_capacity = self.cluster_sizes[layer_idx].shape[-1]
        max_occupied = self.last_available_idx[layer_idx].max()
        min_available = current_capacity - max_occupied.item()

        if min_available < additional_length:
            # extra_pad = max(additional_length - min_available, 64, int(current_capacity / 3))
            # FIXME: for some reason padding masking does not work..
            extra_pad = max(additional_length - min_available, 1)

            batch_size, n_heads, current_capacity, head_dims = self.key_cache[layer_idx].shape
            device = self.key_cache[layer_idx].device
            dtype = self.key_cache[layer_idx].dtype

            pad_vector = torch.zeros(size=(batch_size, n_heads, extra_pad, head_dims), device=device, dtype=dtype)
            pad_cluster_size = torch.zeros(size=(batch_size, n_heads, extra_pad), device=device, dtype=torch.long)
            pad_norm = torch.zeros(size=(batch_size, n_heads, extra_pad), device=device, dtype=dtype)

            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], pad_vector], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], pad_vector], dim=-2)
            self.cluster_sizes[layer_idx] = torch.cat([self.cluster_sizes[layer_idx], pad_cluster_size], dim=-1)
            self.cached_sq_key_norms[layer_idx] = torch.cat([self.cached_sq_key_norms[layer_idx], pad_norm], dim=-1)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, n_heads, seq_length, head_dims = key_states.shape
        device = key_states.device

        # Update the cache
        if key_states is not None:
            if len(self.cluster_sizes) <= layer_idx:
                # There may be skipped layers, fill them with empty lists, as well as the layer_idx layer
                for _ in range(len(self.cluster_sizes), layer_idx + 1):
                    self.per_head_added_variance.append([])
                    self.last_available_idx.append([])
                    self.cluster_sizes.append([])
                    self.cached_sq_key_norms.append([])
                    self.curr_length.append(0)
                    self.optimize_streams.append(None)
                    self.optimization_start_event.append(None)

                # init kv state
                super().update(key_states, value_states, layer_idx, cache_kwargs)

                # init cluster information
                self._init_empty(size=(batch_size, n_heads, seq_length), device=device, layer_idx=layer_idx)
            elif (
                    len(self.cluster_sizes[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                # init kv state
                super().update(key_states, value_states, layer_idx, cache_kwargs)

                # init cluster information
                self._init_empty(size=(batch_size, n_heads, seq_length), device=device, layer_idx=layer_idx)
            else:
                self._resize_if_needed(layer_idx, seq_length)

                self.curr_length[layer_idx] += 1

                batch_index = torch.arange(batch_size).unsqueeze(1)
                head_index = torch.arange(n_heads).unsqueeze(0)
                seq_index = self.last_available_idx[layer_idx]

                # FIXME: only works auto-regressively for now
                assert seq_length == 1

                # important: we do not change last_available_idx (see optimize)
                self.cluster_sizes[layer_idx][batch_index, head_index, seq_index] = 1
                self.key_cache[layer_idx][batch_index, head_index, seq_index] = key_states.squeeze(dim=-2)
                self.value_cache[layer_idx][batch_index, head_index, seq_index] = value_states.squeeze(dim=-2)

                key_norm = torch.sum(key_states ** 2, dim=-1).squeeze(dim=-1)
                self.cached_sq_key_norms[layer_idx][batch_index, head_index, seq_index] = key_norm
                self.last_available_idx[layer_idx] += 1

                # optimize the last entry added
                # on CUDA, this will be called in a separate Stream and CPU process will continue ASAP
                self.optimize_last_entry(layer_idx=layer_idx)

        # log(0) = -inf, so the padding will have 0 attention score
        return self.key_cache[layer_idx], self.value_cache[layer_idx], torch.log(self.cluster_sizes[layer_idx])
