"""
Implementation is based on OffloadedCache from transformers.
"""
from typing import List, Tuple, Optional, Dict, Any

import torch
from transformers import StaticCache


class KVBiasCache(StaticCache):
    pass
