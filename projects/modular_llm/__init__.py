from .src.utils.packed_attention_monkey_patch import (
    flash_attn_func_wrapper,
    flash_attn_varlen_func_wrapper,
    scaled_dot_product_attention,
)

try:
    import flash_attn
    from flash_attn import flash_attn_func, flash_attn_varlen_func

    flash_attn._default_flash_attn_func = flash_attn_func
    flash_attn._default_flash_attn_varlen_func = flash_attn_varlen_func
    flash_attn.flash_attn_varlen_func = flash_attn_varlen_func_wrapper
    flash_attn.flash_attn_func = flash_attn_func_wrapper
except ImportError:
    from mttl.logging import logger

    logger.info("Flash Attention not available")

import torch

torch.nn.functional._default_scaled_dot_product_attention = (
    torch.nn.functional.scaled_dot_product_attention
)
torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
