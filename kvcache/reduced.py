"""
Implementation is based on OffloadedCache from transformers.
"""
from typing import List, Tuple, Optional, Dict, Any, Union

import torch
from transformers import StaticCache, PretrainedConfig
from transformers.utils import is_torchdynamo_compiling
from transformers.utils.deprecation import deprecate_kwarg


@torch.compile()
@torch.no_grad()
def kmeans_torch_reduction(X, num_clusters, num_iters=10):
    """
    Memory-intensive k-means clustering implementation.

    X: Tensor of shape (B, H, L, D)
    num_clusters: Number of clusters C
    Returns: centroids of shape (B, H, C, D)
    """
    B, H, L, D = X.shape
    C = num_clusters
    device = X.device

    X = X.contiguous()

    # Initialize centroids by selecting random points directly
    indices = torch.randint(0, L, (B, H, C), device=device)
    centroids = X[torch.arange(B)[:, None, None],
                   torch.arange(H)[None, :, None],
                   indices]

    assignments = torch.zeros((B, H, L), device=device, dtype=torch.long)
    cluster_sizes = torch.zeros((B, H, C), device=device, dtype=X.dtype)

    for _ in range(num_iters):
        # Compute distances & assign clusters
        distances = torch.cdist(X, centroids)  # (B, H, L, C)
        assignments = torch.argmin(distances, dim=-1)  # (B, H, L)

        # Compute new centroids using vectorized operations
        one_hot_assignments = torch.nn.functional.one_hot(assignments, num_clusters).to(X.dtype)  # (B, H, L, C)
        cluster_sizes = one_hot_assignments.sum(dim=2).clamp(min=1)  # (B, H, C)
        new_centroids = torch.einsum('bhld,bhlc->bhcd', X, one_hot_assignments) / cluster_sizes.unsqueeze(-1)

        # Reset empty clusters
        empty_clusters = one_hot_assignments.sum(dim=2) == 0
        if empty_clusters.any():
            new_samples = torch.randint(0, L, (B, H, C), device=device)
            new_centroids[empty_clusters] = X[torch.arange(B)[:, None, None],
                                              torch.arange(H)[None, :, None],
                                              new_samples][empty_clusters]

        centroids = new_centroids.clone()  # Update centroids

    return centroids, assignments, cluster_sizes



REDUCTION_IMPL = {
    'torch': kmeans_torch_reduction
}


class NaiveKVBiasCache(StaticCache):
    """
    Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

    Parameters:
        config (`PretrainedConfig`):
            The configuration file defining the shape-related attributes required to initialize the static cache.
        batch_size (`int`):
            The batch size with which the model will be used. Note that a new instance must be instantiated if a
            smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
        max_cache_len (`int`):
            The maximum sequence length with which the model will be used.
        device (`torch.device` or `str`):
            The device on which the cache should be initialized. Should be the same as the layer.
            The recommended way however is not not indicate any `device`, in that case cache will be initialized on `meta`
            device by default, and then moved to input device when updating.
        dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
            The default `dtype` to use when initializing the layer.
        layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
            Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
            You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.


    Example:

        ```python
        >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

        >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
        >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

        >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

        >>> # Prepare a cache class and pass it to model's forward
        >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
        >>> max_generated_length = inputs.input_ids.shape[1] + 10
        >>> past_key_values = StaticCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
        >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
        >>> outputs.past_key_values # access cache filled with key/values from generation
        StaticCache()
        ```
    """

    is_compileable = True

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    @deprecate_kwarg("layer_device_map", version="4.52.0")
    def __init__(
        self,
        config: PretrainedConfig,
        batch_size: int = None,
        max_cache_len: int = None,
        target_reduction: int = None,
        kmeans_iterations: int = 10,
        device: torch.device = None,
        dtype: torch.dtype = torch.float32,
        max_batch_size: Optional[int] = None,
        layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()

        self.max_batch_size = batch_size or max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len

        self.target_reduction = target_reduction or self.max_cache_len
        self.kmeans_iterations = kmeans_iterations

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("meta")
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []
        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        cache_shape = (self.max_batch_size, self.num_key_value_heads, self.max_cache_len, self.head_dim)
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = self.device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            if not is_torchdynamo_compiling():
                self.register_buffer(f"key_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
                self.register_buffer(f"value_cache_{idx}", torch.zeros(cache_shape, dtype=dtype, device=layer_device))
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)
            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        If the length of new `key_states` and `value_states` is 1, applies the reduction operation.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states, as well as the weight for each.
        """

        cache_position = cache_kwargs.get("cache_position")
        if self.key_cache[layer_idx].device.type == "meta":
            self.key_cache[layer_idx] = torch.zeros_like(self.key_cache[layer_idx], device=key_states.device)
            self.value_cache[layer_idx] = torch.zeros_like(self.value_cache[layer_idx], device=value_states.device)

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        # naive KV bias works only for auto-regressive decoding
        batch_size, num_heads, seq_length, head_dims = key_states.shape
        if seq_length == 1:
            num_clusters = self.target_reduction
            key_clusters, assignments, cluster_sizes = kmeans_torch_reduction(
                k_out,
                num_clusters=num_clusters,
                num_iters=self.kmeans_iterations
            )

            # equal to the sum of value_states in each cluster
            value_sums = torch.zeros(
                (batch_size, num_heads, num_clusters, head_dims),
                device=value_states.device,
                dtype=value_states.dtype
            )
            assignments_expanded = assignments.unsqueeze(-1).expand(batch_size, num_heads, seq_length, head_dims)
            value_sums.scatter_add_(2, assignments_expanded, value_states)
            return key_clusters, value_sums / cluster_sizes.unsqueeze(-1), cluster_sizes

        # no reduction, each entry has a weight of 1
        batch_size, num_heads, seq_length, _ = k_out.shape
        return k_out, v_out, torch.ones((batch_size, num_heads, seq_length), dtype=k_out.dtype, device=k_out.device)

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if self.key_cache[layer_idx].device.type == "meta":
            return 0
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].device.type != "meta":
                # In-place ops prevent breaking the static address
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()

    @property
    def batch_size(self):
        return self.max_batch_size


class KVBiasCache(StaticCache):
    """
        Static Cache class to be used with `torch.compile(model)` and `torch.export()`.

        Parameters:
            config (`PretrainedConfig`):
                The configuration file defining the shape-related attributes required to initialize the static cache.
            batch_size (`int`):
                The batch size with which the model will be used. Note that a new instance must be instantiated if a
                smaller batch size is used. If you are manually setting the batch size, make sure to take into account the number of beams if you are running beam search
            max_cache_len (`int`):
                The maximum sequence length with which the model will be used.
            device (`torch.device` or `str`):
                The device on which the cache should be initialized. Should be the same as the layer.
                The recommended way however is not not indicate any `device`, in that case cache will be initialized on `meta`
                device by default, and then moved to input device when updating.
            dtype (`torch.dtype`, *optional*, defaults to `torch.float32`):
                The default `dtype` to use when initializing the layer.
            layer_device_map(`Dict[int, Union[str, torch.device, int]]]`, `optional`):
                Mapping between the layers and its device. This is required when you are manually initializing the cache and the model is splitted between differents gpus.
                You can know which layers mapped to which device by checking the associated device_map: `model.hf_device_map`.


        Example:

            ```python
            >>> from transformers import AutoTokenizer, AutoModelForCausalLM, StaticCache

            >>> model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")

            >>> inputs = tokenizer(text="My name is Llama", return_tensors="pt")

            >>> # Prepare a cache class and pass it to model's forward
            >>> # Leave empty space for 10 new tokens, which can be used when calling forward iteratively 10 times to generate
            >>> max_generated_length = inputs.input_ids.shape[1] + 10
            >>> past_key_values = KVBiasCache(config=model.config, batch_size=1, max_cache_len=max_generated_length, device=model.device, dtype=model.dtype)
            >>> outputs = model(**inputs, past_key_values=past_key_values, use_cache=True)
            >>> outputs.past_key_values # access cache filled with key/values from generation
            StaticCache()
            ```
        """

    is_compileable = True

    # TODO (joao): remove `=None` in non-optional arguments in v4.46. Remove from `OBJECTS_TO_IGNORE` as well.
    @deprecate_kwarg("layer_device_map", version="4.52.0")
    def __init__(
            self,
            config: PretrainedConfig,
            batch_size: int = None,
            max_cache_len: int = None,
            cache_size_reduction: float = 1.0,
            device: torch.device = None,
            dtype: torch.dtype = torch.float32,
            max_batch_size: Optional[int] = None,
            layer_device_map: Optional[Dict[int, Union[str, torch.device, int]]] = None,
    ) -> None:
        super().__init__()

        self.max_batch_size = batch_size or max_batch_size
        self.max_cache_len = config.max_position_embeddings if max_cache_len is None else max_cache_len
        self.max_reduced_cache_size = int(cache_size_reduction * self.max_cache_len)

        # Some model define a custom `head_dim` != config.hidden_size // config.num_attention_heads
        self.head_dim = (
            config.head_dim if hasattr(config, "head_dim") else config.hidden_size // config.num_attention_heads
        )

        self.dtype = dtype
        self.device = torch.device(device) if device is not None else torch.device("meta")
        self.num_key_value_heads = (
            config.num_attention_heads
            if getattr(config, "num_key_value_heads", None) is None
            else config.num_key_value_heads
        )

        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

        self.reduced_key_cache: List[torch.Tensor] = []
        self.reduced_value_cache: List[torch.Tensor] = []
        self.n_kv_pairs: List[torch.Tensor] = []
        self._reduce_stream = torch.cuda.Stream() if self.device.type == "cuda" else None

        # Note: There will be significant perf decrease if switching to use 5D tensors instead.
        batch_head = (self.max_batch_size, self.num_key_value_heads)
        cache_shape = batch_head + (self.max_cache_len, self.head_dim)
        reduced_cache_shape = batch_head + (self.max_reduced_cache_size, self.head_dim)
        n_kv_pairs_shape = batch_head + (self.max_reduced_cache_size,)
        for idx in range(config.num_hidden_layers):
            if layer_device_map is not None:
                layer_device = layer_device_map[idx]
            else:
                layer_device = self.device
            new_layer_key_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_value_cache = torch.zeros(cache_shape, dtype=self.dtype, device=layer_device)

            new_layer_reduced_key_cache = torch.zeros(reduced_cache_shape, dtype=self.dtype, device=layer_device)
            new_layer_reduced_value_cache = torch.zeros(reduced_cache_shape, dtype=self.dtype, device=layer_device)
            new_n_kv_pairs = torch.zeros(n_kv_pairs_shape, dtype=self.dtype, device=layer_device)
            # Notes:
            # 1. `mark_static_address` is used to tag the cache as an fixed data pointer, preventing cuda graph
            #     breaks when updating the cache. It can't be used if the cache code is being compiled (but in that case
            #     it is not needed anyway)
            # 2. `torch.export()` requires mutations to be registered as buffers.
            if not is_torchdynamo_compiling():
                self.register_buffer(
                    f"key_cache_{idx}",
                    torch.zeros(cache_shape, dtype=dtype, device=layer_device)
                )
                self.register_buffer(
                    f"value_cache_{idx}",
                    torch.zeros(cache_shape, dtype=dtype, device=layer_device)
                )
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                torch._dynamo.mark_static_address(new_layer_key_cache)
                torch._dynamo.mark_static_address(new_layer_value_cache)

                self.register_buffer(
                    f"reduced_key_cache_{idx}",
                    torch.zeros(reduced_cache_shape, dtype=dtype, device=layer_device)
                )
                self.register_buffer(
                    f"reduced_value_cache_{idx}",
                    torch.zeros(reduced_cache_shape, dtype=dtype, device=layer_device)
                )
                self.register_buffer(
                    f"n_kv_pairs_{idx}",
                    torch.zeros(n_kv_pairs_shape, dtype=dtype, device=layer_device)
                )
                new_layer_key_cache = getattr(self, f"key_cache_{idx}")
                new_layer_value_cache = getattr(self, f"value_cache_{idx}")
                new_n_kv_pairs = getattr(self, f"n_kv_pairs_{idx}")
                torch._dynamo.mark_static_address(new_layer_reduced_key_cache)
                torch._dynamo.mark_static_address(new_layer_reduced_value_cache)
                torch._dynamo.mark_static_address(new_n_kv_pairs)

            self.key_cache.append(new_layer_key_cache)
            self.value_cache.append(new_layer_value_cache)

            self.reduced_key_cache.append(new_layer_key_cache)
            self.reduced_value_cache.append(new_layer_value_cache)
            self.n_kv_pairs.append(new_n_kv_pairs)

    def update(
            self,
            key_states: torch.Tensor,
            value_states: torch.Tensor,
            layer_idx: int,
            cache_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Updates the cache with the new `key_states` and `value_states` for the layer `layer_idx`.
        It is VERY important to index using a tensor, otherwise you introduce a copy to the device.

        Parameters:
            key_states (`torch.Tensor`):
                The new key states to cache.
            value_states (`torch.Tensor`):
                The new value states to cache.
            layer_idx (`int`):
                The index of the layer to cache the states for.
            cache_kwargs (`Dict[str, Any]`, `optional`):
                Additional arguments for the cache subclass. The `StaticCache` needs the `cache_position` input
                to know how where to write in the cache.

        Return:
            A tuple containing the updated key and value states.
        """

        cache_position = cache_kwargs.get("cache_position")
        if self.key_cache[layer_idx].device.type == "meta":
            self.key_cache[layer_idx] = torch.zeros_like(self.key_cache[layer_idx], device=key_states.device)
            self.value_cache[layer_idx] = torch.zeros_like(self.value_cache[layer_idx], device=value_states.device)

        if self.reduced_key_cache[layer_idx].device.type == "meta":
            self.reduced_key_cache[layer_idx] = torch.zeros_like(self.reduced_key_cache[layer_idx], device=key_states.device)
            self.reduced_value_cache[layer_idx] = torch.zeros_like(self.reduced_value_cache[layer_idx], device=value_states.device)
            self.n_kv_pairs[layer_idx] = torch.zeros_like(self.n_kv_pairs[layer_idx], device=key_states.device)

        k_out = self.key_cache[layer_idx]
        v_out = self.value_cache[layer_idx]
        key_states = key_states.to(k_out.dtype)
        value_states = value_states.to(v_out.dtype)

        # TODO: reduce here

        if cache_position is None:
            k_out.copy_(key_states)
            v_out.copy_(value_states)
        else:
            # Note: here we use `tensor.index_copy_(dim, index, tensor)` that is equivalent to
            # `tensor[:, :, index] = tensor`, but the first one is compile-friendly and it does explicitly an in-place
            # operation, that avoids copies and uses less memory.
            try:
                k_out.index_copy_(2, cache_position, key_states)
                v_out.index_copy_(2, cache_position, value_states)
            except NotImplementedError:
                # The operator 'aten::index_copy.out' is not currently implemented for the MPS device.
                k_out[:, :, cache_position] = key_states
                v_out[:, :, cache_position] = value_states

        return k_out, v_out

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Returns the sequence length of the cached states that were seen by the model."""
        # Occupied cache == any slot in the 3rd dim (sequence length) holds a non-zero value. To save on compute, let's
        # limit the check to the first batch member and head dimension.
        # TODO: deprecate this function in favor of `cache_position`
        if self.key_cache[layer_idx].device.type == "meta":
            return 0
        return (self.key_cache[layer_idx][0, 0].any(dim=-1)).sum()

    def get_max_cache_shape(self) -> Optional[int]:
        return self.max_cache_len

    def reset(self):
        """Resets the cache values while preserving the objects"""
        for layer_idx in range(len(self.key_cache)):
            if self.key_cache[layer_idx].device.type != "meta":
                # In-place ops prevent breaking the static address
                self.key_cache[layer_idx].zero_()
                self.value_cache[layer_idx].zero_()

    @property
    def batch_size(self):
        return self.max_batch_size
