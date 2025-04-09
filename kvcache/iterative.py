from typing import Iterable, Optional, Dict, Any, List, Tuple, ContextManager

import torch
from transformers import DynamicCache


VARY_LARGE_LONG = torch.tensor([2**62])


class DudContextManager(ContextManager):
    def __exit__(self, __exc_type, __exc_value, __traceback):
        return self


class IterativeReduceKVBiasCache(DynamicCache):
    def __init__(self, _distributed_cache_data: Iterable = None, *, max_variance_increase: float = 0.0) -> None:
        super().__init__(_distributed_cache_data)

        self.cluster_sizes: List[torch.Tensor] = []
        self.per_head_added_variance: List[torch.Tensor] = []
        self.max_variance_increase = max_variance_increase

        self.optimize_stream = torch.cuda.Stream() if torch.cuda.is_available() else None

    def _init_empty(
            self,
            size: tuple[int, int, int],
            device: torch.device | str,
            layer_idx: int
    ) -> None:
        batch_size, n_heads, seq_length = size

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

    def optimize_layer(self, layer_idx: int) -> None:
        if len(self.key_cache) < layer_idx or not len(self.key_cache[layer_idx]):
            return  # empty cache

        if self.cluster_sizes[layer_idx].shape[-1] < 2:
            return  # nothing to optimize

        with self.optimize_stream if torch.cuda.is_available() else DudContextManager():
            # select subset to optimize
            key_states = self.key_cache[layer_idx][:, :, [-1], :]
            value_states = self.value_cache[layer_idx][:, :, [-1], :]

            batch_size, n_heads, seq_length, head_dims = key_states.shape
            device = key_states.device
            dtype = key_states.dtype

            # examine for matches in the cache
            new_ks_broadcast = key_states[:, :, :, None, :]  # broadcast over old seq
            old_ks_broadcast = self.key_cache[layer_idx][:, :, None, :, :]  # broadcast over new seq

            # (B, H, S_new, S_old)
            sq_dist = torch.square(new_ks_broadcast - old_ks_broadcast).sum(dim=-1)
            variance_weight = self.cluster_sizes[layer_idx] / (self.cluster_sizes[layer_idx] + 1)
            vw_broadcast = variance_weight[:, :, None, :]  # broadcast over new seq

            variance_increase = vw_broadcast * sq_dist
            # (B, H, S_new)
            min_variance_increase, min_idx = torch.min(variance_increase, dim=-1)
            merge_condition = min_variance_increase < self.max_variance_increase

            # protect against merging with padding keys
            padding_mask = self.cluster_sizes[layer_idx] < VARY_LARGE_LONG.to(device)
            per_head_last_idx = padding_mask.sum(dim=-1, keepdim=True)
            merge_condition = merge_condition & (min_idx < per_head_last_idx)

            # update current added variance
            added_variance = torch.sum(min_variance_increase * merge_condition, dim=-1)
            self.per_head_added_variance[layer_idx] = self.per_head_added_variance[layer_idx] + added_variance

            # compute new pad size
            extra_capacity = torch.sum(~padding_mask, dim=-1)
            needed_capacity = torch.sum(~merge_condition, dim=-1)
            extra_pad = (needed_capacity - extra_capacity).max()

            # make extra space for new key-values
            if extra_pad > 0:
                key_pad = torch.full(
                    size=(batch_size, n_heads, extra_pad, head_dims),
                    fill_value=torch.inf,
                    device=device,
                    dtype=dtype
                )
                self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_pad], dim=-2)

                value_pad = torch.zeros_like(key_pad)
                self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_pad], dim=-2)

                cluster_size_pad = torch.full(
                    size=(batch_size, n_heads, extra_pad),
                    fill_value=VARY_LARGE_LONG.item(),
                    device=device,
                    dtype=torch.long
                )
                self.cluster_sizes[layer_idx] = torch.cat([self.cluster_sizes[layer_idx], cluster_size_pad], dim=-1)

            # this is inefficient, but easier to implement, will do for now
            for batch_idx in range(batch_size):
                for head_idx in range(n_heads):
                    curr_key_states = key_states[batch_idx, head_idx, :, :]
                    curr_value_states = value_states[batch_idx, head_idx, :, :]
                    curr_merge_condition = merge_condition[batch_idx, head_idx]
                    curr_min_idx = min_idx[batch_idx, head_idx]

                    # update existing cluster sizes
                    curr_selected_cluster_sizes = self.cluster_sizes[layer_idx][batch_idx, head_idx, curr_min_idx]
                    new_selected_cluster_sizes = curr_merge_condition + curr_selected_cluster_sizes
                    self.cluster_sizes[layer_idx][batch_idx, head_idx, curr_min_idx] = new_selected_cluster_sizes

                    # update cluster centroids
                    inertia = 1 / (curr_selected_cluster_sizes + 1)

                    # update keys
                    curr_selected_cluster_centroids = self.key_cache[layer_idx][batch_idx, head_idx, curr_min_idx]
                    change = inertia * (curr_key_states - curr_selected_cluster_centroids)
                    new_selected_clusters = curr_selected_cluster_centroids + change * curr_merge_condition
                    self.key_cache[layer_idx][batch_idx, head_idx, curr_min_idx] = new_selected_clusters

                    # update values
                    curr_selected_cluster_centroids = self.value_cache[layer_idx][batch_idx, head_idx, curr_min_idx]
                    change = inertia * (curr_value_states - curr_selected_cluster_centroids)
                    new_selected_clusters = curr_selected_cluster_centroids + change * curr_merge_condition
                    self.value_cache[layer_idx][batch_idx, head_idx, curr_min_idx] = new_selected_clusters

                    # append unmerged
                    n_new = torch.sum(~curr_merge_condition).item()
                    last_idx = per_head_last_idx[batch_idx, head_idx]

                    self.cluster_sizes[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = torch.ones(
                        size=(n_new,),
                        device=device,
                        dtype=torch.long
                    )

                    new_keys = curr_key_states[~curr_merge_condition]
                    self.key_cache[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = new_keys

                    new_values = curr_value_states[~curr_merge_condition]
                    self.value_cache[layer_idx][batch_idx, head_idx, last_idx:last_idx + n_new] = new_values

    def __getitem__(self, layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Gets the cache for this layer. Optimizes the next layer."""

        if layer_idx < len(self):
            if torch.cuda.is_available():
                self.optimize_stream.synchronize()

            key_tensor = self.key_cache[layer_idx]
            value_tensor = self.value_cache[layer_idx]
            size_tensor = self.cluster_sizes[layer_idx]

            self.optimize_layer((layer_idx + 1) % len(self))
            return key_tensor, value_tensor, size_tensor
        else:
            raise KeyError(f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}")

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
                    self.cluster_sizes.append([])
                    self.per_head_added_variance.append([])

                # init cluster information
                self._init_empty(size=(batch_size, n_heads, seq_length), device=device, layer_idx=layer_idx)

                # init kv state
                super().update(key_states, value_states, layer_idx, cache_kwargs)
            elif (
                    len(self.cluster_sizes[layer_idx]) == 0
            ):  # fills previously skipped layers; checking for tensor causes errors
                # init cluster information
                self._init_empty(size=(batch_size, n_heads, seq_length), device=device, layer_idx=layer_idx)

                # init kv state
                super().update(key_states, value_states, layer_idx, cache_kwargs)
            else:
                # initiate async optimization for the next layer
                key_tensor, value_tensor, cluster_sizes = self[layer_idx]

                # update cluster info
                self.cluster_sizes[layer_idx] = torch.cat([
                    cluster_sizes,
                    torch.ones(size=(batch_size, n_heads, seq_length), device=device, dtype=torch.long),
                ], dim=-1)

                # update kv state
                self.key_cache[layer_idx] = torch.cat([key_tensor, key_states], dim=-2)
                self.value_cache[layer_idx] = torch.cat([value_tensor, value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx], torch.log(self.cluster_sizes[layer_idx])
