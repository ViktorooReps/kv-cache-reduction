from typing import Iterable, Optional, Dict, Any, List, Tuple

import torch
from transformers import DynamicCache

from kvcache.optimization_orchestrator import OptimizationOrchestrator
from kvcache.wrapped_manager import WrappedManager


# TODO: test on CUDA


class IterativeReduceKVBiasCache(DynamicCache):
    def __init__(
            self,
            _distributed_cache_data: Iterable = None,
            *,
            # MVI = Maximum Variance Increase
            protect_first: int = 20,
            mvi: float = 0.0,
    ) -> None:
        super().__init__(_distributed_cache_data)

        self.curr_length: List[int] = []

        # (B, H, S)
        self.cluster_sizes: List[torch.Tensor] = []
        self.cached_sq_norms: List[torch.Tensor] = []
        self.last_available_idx: List[torch.Tensor] = []
        self.first_non_optimized_idx: List[torch.Tensor] = []

        self.per_head_added_variance: List[torch.Tensor] = []
        self.mvi = mvi
        self.protect_first = protect_first

        self.optimize_streams: List[torch.cuda.Stream] = []
        self.orchestrator = OptimizationOrchestrator() if torch.cuda.is_available() else None

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

    def _init_cuda(self, layer_idx: int) -> None:
        while len(self.optimize_streams) <= layer_idx:
            self.optimize_streams.append(torch.cuda.Stream(priority=-1))

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

        self.cached_sq_norms[layer_idx] = torch.sum(self.key_cache[layer_idx] ** 2, dim=-1)
        self.last_available_idx[layer_idx] = torch.full(
            size=(batch_size, n_heads),
            fill_value=seq_length,
            device=device,
            dtype=torch.long
        )
        self.first_non_optimized_idx[layer_idx] = torch.zeros_like(self.last_available_idx[layer_idx])

        if torch.cuda.is_available():
            self._init_cuda(layer_idx)

    def optimize_last_entries(self, layer_idx: int, n_optimize: int) -> None:
        to_wrap = None
        timestep = None

        if self.optimize_streams[layer_idx] is not None:
            to_wrap = torch.cuda.stream(self.optimize_streams[layer_idx])

        def end_optimize():
            if self.orchestrator is not None and self.optimize_streams[layer_idx] is not None:
                if timestep is None:
                    raise RuntimeError(f'Premature end of optimization! start_optimization has not been called!')
                self.orchestrator.end_optimization(timestep, layer_idx, stream=self.optimize_streams[layer_idx])

        manager = WrappedManager(to_wrap, on_exit=end_optimize)

        with manager:
            # Two possibilities:
            # 1) CUDA is not available and this call will block the forward execution of the model
            # 2) CUDA is available and this call will be executed right after the optimization from the previous
            #   time step on this layer is complete. This is expected to happen because we are using layer-specific
            #   stream and CUDA executes operations FIFO.
            batch_size, n_heads, _, head_dims = self.key_cache[layer_idx].shape
            dtype = self.key_cache[layer_idx].dtype

            first_entry = self.first_non_optimized_idx[layer_idx]   # (B, H)
            first_entry_3d = first_entry[:, :, None].expand(-1, -1, n_optimize)
            relative_index = torch.arange(n_optimize, device=first_entry.device)
            last_entries_idx = first_entry_3d + relative_index[None, None, :]  # (B, H, S_new)

            last_entries_idx_4d = last_entries_idx[:, :, :, None].expand(-1, -1, -1, head_dims)

            # select last entry to optimize
            key_states = torch.take_along_dim(self.key_cache[layer_idx], last_entries_idx_4d, dim=-2)
            value_states = torch.take_along_dim(self.value_cache[layer_idx], last_entries_idx_4d, dim=-2)
            new_norm = torch.take_along_dim(
                self.cached_sq_norms[layer_idx],
                last_entries_idx,
                dim=-1
            )[:, :, :, None].transpose(-1, -2)  # move seq dimension to the back

            # remove the last entry from current cache
            new_cluster_sizes = torch.clone(self.cluster_sizes[layer_idx])  # clone to not affect the cache state
            cluster_sizes_zero_fill = torch.zeros_like(last_entries_idx, device=new_cluster_sizes.device)
            new_cluster_sizes.scatter_(dim=-1, index=last_entries_idx, src=cluster_sizes_zero_fill)

            # examine for matches in the cache by computing pairwise square distance to each element in the cache
            old_norm = self.cached_sq_norms[layer_idx][:, :, :, None]  # (B, H, S_old, 1)
            cross_term = torch.matmul(self.key_cache[layer_idx], key_states.transpose(-1, -2))  # (B, H, S_old, S_new)
            sq_dist = new_norm + old_norm - 2 * cross_term  # (B, H, S_old, S_new)
            sq_dist.clamp_(0)  # sometimes numerical error can create negative values here

            # CAREFUL: this value will be 0 for padding
            variance_weight = new_cluster_sizes / (new_cluster_sizes + 1)
            variance_weight_bcast_4d = variance_weight[:, :, :, None]  # broadcast over new seq

            # depends on cluster sizes and distance to new element
            variance_increase = variance_weight_bcast_4d * sq_dist

            # protect from selecting padding as merging candidate (still can happen if there is no other candidate)
            # padding is determined by 0 in cluster size
            protect_range = 1 / variance_weight  # this will make 0 into inf
            protect_range = torch.clamp(protect_range - 100, 0)  # inf - 100 = inf; vw \in [1/2, 1)
            protect_range[:, :, :self.protect_first] = torch.inf
            protect_range_bcast_4d = protect_range[:, :, :, None]  # broadcast over new seq
            variance_increase = variance_increase + protect_range_bcast_4d

            # (B, H, S_new) and (B, H, S_new)
            min_variance_increase, min_idx = torch.min(variance_increase, dim=-2)
            merge_condition = min_variance_increase < self.mvi

            invalid_variance = torch.isinf(min_variance_increase) | torch.isnan(min_variance_increase)
            merge_condition = merge_condition & ~invalid_variance

            # protect against merging with padding keys and last entries
            merge_condition = merge_condition & (min_idx < first_entry_3d)
            merge_condition_4d = merge_condition[:, :, :, None].expand(-1, -1, -1, head_dims)

            # UPDATE KV CACHE STATE
            # We need to wait for the forward layer call to finish working with the current state of KV cache of this
            if self.orchestrator is not None and self.optimize_streams[layer_idx] is not None:
                timestep = self.orchestrator.start_optimization(layer_idx, stream=self.optimize_streams[layer_idx])

            # remove the last entry from the current cache
            self.cluster_sizes[layer_idx].scatter_(dim=-1, index=last_entries_idx, src=cluster_sizes_zero_fill)

            # update current added variance
            added_variance = torch.sum(min_variance_increase * merge_condition, dim=-1)
            self.per_head_added_variance[layer_idx] = self.per_head_added_variance[layer_idx] + added_variance

            # we will update KV cache at seq_idx locations
            seq_idx = min_idx  # seq_idx is the positions of the elements to merge into
            seq_idx_4d = seq_idx[:, :, :, None].expand(-1, -1, -1, head_dims)

            # update existing cluster sizes
            selected_cluster_sizes = torch.take_along_dim(self.cluster_sizes[layer_idx], seq_idx, dim=-1)
            new_cluster_sizes = selected_cluster_sizes + merge_condition
            self.cluster_sizes[layer_idx].scatter_(dim=-1, index=seq_idx, src=new_cluster_sizes)

            # update cluster centroids, inertia is computed based on the previous values
            inertia = 1 / (selected_cluster_sizes + 1)
            inertia = inertia.to(dtype=dtype)
            inertia_4d = inertia[:, :, :, None].expand(-1, -1, -1, head_dims)

            # update keys
            selected_cluster_centroids = torch.take_along_dim(self.key_cache[layer_idx], seq_idx_4d, dim=-2)
            change = inertia_4d * (key_states - selected_cluster_centroids)
            new_selected_clusters = selected_cluster_centroids + change * merge_condition_4d
            self.key_cache[layer_idx].scatter_(dim=-2, index=seq_idx_4d, src=new_selected_clusters)

            # update cached norms for updated keys
            new_sq_key_norms = torch.sum(new_selected_clusters ** 2, dim=-1)
            self.cached_sq_norms[layer_idx].scatter_(dim=-1, index=seq_idx, src=new_sq_key_norms)

            # update values
            selected_cluster_centroids = torch.take_along_dim(self.value_cache[layer_idx], seq_idx_4d, dim=-2)
            change = inertia_4d * (value_states - selected_cluster_centroids)
            new_selected_clusters = selected_cluster_centroids + change * merge_condition_4d
            self.value_cache[layer_idx].scatter_(dim=-2, index=seq_idx_4d, src=new_selected_clusters)

            # we will update KV cache at [batch_idx, head_idx, seq_idx] locations
            seq_idx = last_entries_idx  # seq_idx are the positions of the last elements
            seq_idx_4d = seq_idx[:, :, :, None].expand(-1, -1, -1, head_dims)

            not_merged = ~merge_condition

            # sort new entries by not_merged condition: not_merged - [1, 1, 1, 1, 0, 0, 0] <- over seq dimension
            not_merged_sorted, idx_sorted = torch.sort(not_merged, dim=-1, descending=True)
            not_merged_srt_4d = not_merged_sorted[:, :, :, None].expand(-1, -1, -1, head_dims)
            idx_srt_4d = idx_sorted[:, :, :, None].expand(-1, -1, -1, head_dims)

            new_keys_sorted = torch.take_along_dim(key_states, idx_srt_4d, dim=-2) * not_merged_srt_4d
            new_values_sorted = torch.take_along_dim(value_states, idx_srt_4d, dim=-2) * not_merged_srt_4d
            new_norms_sorted = (torch.sum(new_keys_sorted ** 2, dim=-1) ** 2) * not_merged_sorted

            self.cluster_sizes[layer_idx].scatter_(dim=-1, index=seq_idx, src=not_merged_sorted.to(dtype=torch.long))
            self.key_cache[layer_idx].scatter_(dim=-2, index=seq_idx_4d, src=new_keys_sorted)
            self.cached_sq_norms[layer_idx].scatter_(dim=-1, index=seq_idx, src=new_norms_sorted)
            self.value_cache[layer_idx].scatter_(dim=-2, index=seq_idx_4d, src=new_values_sorted)

            n_not_merged = not_merged.sum(dim=-1)
            not_merged_end = first_entry + n_not_merged
            self.last_available_idx[layer_idx] = not_merged_end
            self.first_non_optimized_idx[layer_idx] = not_merged_end

    def _resize_if_needed(self, layer_idx: int, additional_length: int):
        current_capacity = self.cluster_sizes[layer_idx].shape[-1]
        max_occupied = self.last_available_idx[layer_idx].max()
        min_available = current_capacity - max_occupied.item()

        if min_available < additional_length:
            extra_pad = max(additional_length - min_available, 64, int(current_capacity / 3))

            batch_size, n_heads, current_capacity, head_dims = self.key_cache[layer_idx].shape
            device = self.key_cache[layer_idx].device
            dtype = self.key_cache[layer_idx].dtype

            pad_vector = torch.zeros(size=(batch_size, n_heads, extra_pad, head_dims), device=device, dtype=dtype)
            pad_cluster_size = torch.zeros(size=(batch_size, n_heads, extra_pad), device=device, dtype=torch.long)
            pad_norm = torch.zeros(size=(batch_size, n_heads, extra_pad), device=device, dtype=dtype)

            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], pad_vector], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], pad_vector], dim=-2)
            self.cluster_sizes[layer_idx] = torch.cat([self.cluster_sizes[layer_idx], pad_cluster_size], dim=-1)
            self.cached_sq_norms[layer_idx] = torch.cat([self.cached_sq_norms[layer_idx], pad_norm], dim=-1)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        batch_size, n_heads, seq_length, head_dims = key_states.shape
        device = key_states.device

        if self.orchestrator is not None:
            self.orchestrator.start_forward(layer_idx, stream=torch.cuda.current_stream())

        # Update the cache
        if key_states is not None:
            if len(self.cluster_sizes) <= layer_idx:
                # There may be skipped layers, fill them with empty lists, as well as the layer_idx layer
                for _ in range(len(self.cluster_sizes), layer_idx + 1):
                    self.per_head_added_variance.append([])
                    self.last_available_idx.append([])
                    self.first_non_optimized_idx.append([])
                    self.cluster_sizes.append([])
                    self.cached_sq_norms.append([])
                    self.curr_length.append(0)
                    self.optimize_streams.append(None)

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

                self.curr_length[layer_idx] += seq_length

                first_available = self.last_available_idx[layer_idx]  # (B, H)
                relative_index = torch.arange(seq_length, device=first_available.device)
                last_entries_idx = first_available[:, :, None] + relative_index[None, None, :]  # (B, H, S_new)
                last_entries_idx_4d = last_entries_idx[:, :, :, None].expand(-1, -1, -1, head_dims)

                self.cluster_sizes[layer_idx].scatter_(-1, last_entries_idx, torch.ones_like(last_entries_idx))
                self.key_cache[layer_idx].scatter_(-2, last_entries_idx_4d, key_states)
                self.value_cache[layer_idx].scatter_(-2, last_entries_idx_4d, value_states)

                key_norm = torch.sum(key_states ** 2, dim=-1)
                self.cached_sq_norms[layer_idx].scatter_(-1, last_entries_idx, key_norm)
                self.last_available_idx[layer_idx] = self.last_available_idx[layer_idx] + seq_length

        # optimize the last entry added
        # on CUDA, this will be called in a separate Stream and CPU process will continue ASAP
        self.optimize_last_entries(layer_idx=layer_idx, n_optimize=seq_length)

        # log(0) = -inf, so the padding will have 0 attention score
        return self.key_cache[layer_idx], self.value_cache[layer_idx], torch.log(self.cluster_sizes[layer_idx])
