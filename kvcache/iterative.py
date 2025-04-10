from typing import Iterable, Optional, Dict, Any, List, Tuple, ContextManager

import torch
from transformers import DynamicCache


class DudContextManager(ContextManager):
    def __exit__(self, __exc_type, __exc_value, __traceback):
        # If an exception is raised, re-raise it
        if __exc_type is not None:
            raise __exc_value
        # Return False to allow the exception to propagate (if any)
        return False


class IterativeReduceKVBiasCache(DynamicCache):
    def __init__(self, _distributed_cache_data: Iterable = None, *, max_variance_increase: float = 0.0) -> None:
        super().__init__(_distributed_cache_data)

        self.curr_length: List[int] = []

        # (B, H, S)
        self.cluster_sizes: List[torch.Tensor] = []
        self.cached_sq_key_norms: List[torch.Tensor] = []
        self.last_available_idx: List[torch.Tensor] = []

        self.per_head_added_variance: List[torch.Tensor] = []
        self.max_variance_increase = max_variance_increase

        self.optimize_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

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

        return self.curr_length[layer_idx]

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

    def optimize_layer(self, layer_idx: int) -> None:
        if len(self.key_cache) < layer_idx or not len(self.key_cache[layer_idx]):
            return  # empty cache

        if self.get_seq_length(layer_idx) < 2:
            return  # nothing to optimize

        with self.optimize_stream if torch.cuda.is_available() else DudContextManager():
            batch_size, n_heads, _, head_dims = self.key_cache[layer_idx].shape
            device = self.key_cache[layer_idx].device

            # select subset to optimize
            key_states = torch.take_along_dim(
                self.key_cache[layer_idx],
                self.last_available_idx[layer_idx][:, :, None, None],
                dim=-2
            )
            value_states = torch.take_along_dim(
                self.value_cache[layer_idx],
                self.last_available_idx[layer_idx][:, :, None, None],
                dim=-2
            )
            new_norm = torch.take_along_dim(
                self.cached_sq_key_norms[layer_idx],
                self.last_available_idx[layer_idx][:, :, None],
                dim=-1
            )[:, :, :, None].transpose(-1, -2)  # move seq dimension to the back

            # examine for matches in the cache by computing pairwise square distance to each element in th cache
            old_norm = self.cached_sq_key_norms[layer_idx][:, :, :, None]  # (B, H, S_old, 1)
            cross_term = torch.matmul(self.key_cache[layer_idx], key_states.transpose(-1, -2))  # (B, H, S_old, S_new)
            sq_dist = new_norm + old_norm - 2 * cross_term  # (B, H, S_old, S_new)

            # CAREFUL: this value will be 0 for padding
            variance_weight = self.cluster_sizes[layer_idx] / (self.cluster_sizes[layer_idx] + 1)
            vw_broadcast = variance_weight[:, :, :, None]  # broadcast over new seq

            # depends on cluster sizes and distance to new element
            variance_increase = vw_broadcast * sq_dist

            # protect from selecting padding as merging candidate (still can happen if there is no other candidate)
            # padding is determined by 0 in cluster size
            protect_range = 1 / variance_weight  # this will make 0 into inf
            protect_range = torch.clamp(protect_range - 100, 0)  # inf - 100 = inf; vw \in [1/2, 1)
            pr_broadcast = protect_range[:, :, :, None]  # broadcast over new seq
            variance_increase = variance_increase + pr_broadcast

            # (B, H, S_new)
            min_variance_increase, min_idx = torch.min(variance_increase, dim=-2)
            merge_condition = min_variance_increase < self.max_variance_increase

            invalid_variance = torch.isinf(min_variance_increase) | torch.isnan(min_variance_increase)
            assert not invalid_variance.any()

            # protect against merging with padding keys
            merge_condition = merge_condition & (min_idx < self.last_available_idx[layer_idx][:, :, None])

            # update current added variance
            added_variance = torch.sum(min_variance_increase * merge_condition, dim=-1)
            self.per_head_added_variance[layer_idx] = self.per_head_added_variance[layer_idx] + added_variance

            # this is inefficient, but easier to implement, will do for now
            for batch_idx in range(batch_size):
                for head_idx in range(n_heads):
                    curr_key_states = key_states[batch_idx, head_idx, :, :]
                    curr_value_states = value_states[batch_idx, head_idx, :, :]
                    curr_merge_condition = merge_condition[batch_idx, head_idx]
                    curr_min_idx = min_idx[batch_idx, head_idx]

                    # update existing cluster sizes
                    curr_selected_sizes = self.cluster_sizes[layer_idx][batch_idx, head_idx, curr_min_idx]
                    new_selected_cluster_sizes = curr_merge_condition + curr_selected_sizes
                    self.cluster_sizes[layer_idx][batch_idx, head_idx, curr_min_idx] = new_selected_cluster_sizes

                    # update cluster centroids
                    inertia = 1 / (curr_selected_sizes + 1)

                    # update keys
                    curr_selected_cluster_centroids = self.key_cache[layer_idx][batch_idx, head_idx, curr_min_idx]
                    change = inertia[:, None] * (curr_key_states - curr_selected_cluster_centroids)
                    new_selected_clusters = curr_selected_cluster_centroids + change * curr_merge_condition[:, None]
                    self.key_cache[layer_idx][batch_idx, head_idx, curr_min_idx] = new_selected_clusters

                    # update cached norms for updated keys
                    new_sq_key_norms = torch.sum(new_selected_clusters ** 2, dim=-1)
                    self.cached_sq_key_norms[layer_idx][batch_idx, head_idx, curr_min_idx] = new_sq_key_norms

                    # update values
                    curr_selected_cluster_centroids = self.value_cache[layer_idx][batch_idx, head_idx, curr_min_idx]
                    change = inertia[:, None] * (curr_value_states - curr_selected_cluster_centroids)
                    new_selected_clusters = curr_selected_cluster_centroids + change * curr_merge_condition[:, None]
                    self.value_cache[layer_idx][batch_idx, head_idx, curr_min_idx] = new_selected_clusters

                    # append unmerged
                    n_new = torch.sum(~curr_merge_condition).item()
                    last_idx = self.last_available_idx[layer_idx][batch_idx, head_idx]

                    self.cluster_sizes[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = torch.ones(
                        size=(n_new,),
                        device=device,
                        dtype=torch.long
                    )

                    new_keys = curr_key_states[~curr_merge_condition, :]
                    self.key_cache[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = new_keys

                    sq_key_norms = torch.sum(new_keys ** 2, dim=-1)
                    self.cached_sq_key_norms[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = sq_key_norms

                    new_values = curr_value_states[~curr_merge_condition, :]
                    self.value_cache[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = new_values

                    self.last_available_idx[layer_idx][batch_idx, head_idx] += 1

    def _resize_if_needed(self, layer_idx: int, additional_length: int):
        current_capacity = self.cluster_sizes[layer_idx].shape[-1]
        max_occupied = self.last_available_idx[layer_idx].max()
        min_available = current_capacity - max_occupied.item()

        if min_available < additional_length:
            # max(additional_length - min_available, 64, int(current_capacity / 3))
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
                # initiate async optimization for the next layer
                if torch.cuda.is_available():
                    self.optimize_stream.synchronize()
                self.optimize_layer((layer_idx + 1) % len(self))

                self._resize_if_needed(layer_idx, seq_length)

                self.curr_length[layer_idx] += 1

                batch_index = torch.arange(batch_size).unsqueeze(1)
                head_index = torch.arange(n_heads).unsqueeze(0)
                seq_index = self.last_available_idx[layer_idx]

                # FIXME: only works auto-regressively for now
                assert seq_length == 1

                # important: we do not change last_available_idx or cluster size (see optimize)
                self.key_cache[layer_idx][batch_index, head_index, seq_index] = key_states.squeeze(dim=-2)
                self.value_cache[layer_idx][batch_index, head_index, seq_index] = value_states.squeeze(dim=-2)

                key_norm = torch.sum(key_states ** 2, dim=-1).squeeze(dim=-1)
                self.cached_sq_key_norms[layer_idx][batch_index, head_index, seq_index] = key_norm

        return self.key_cache[layer_idx], self.value_cache[layer_idx], torch.log(self.cluster_sizes[layer_idx])
