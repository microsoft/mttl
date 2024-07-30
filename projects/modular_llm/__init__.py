import transformers
import transformers.modeling_flash_attention_utils

from .src.utils.flash_attention_monkey_patch import (
    flash_attention_forward,
    get_unpad_data,
    scaled_dot_product_attention,
)

transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
transformers.modeling_flash_attention_utils._flash_attention_forward = (
    flash_attention_forward
)

import torch

torch.nn.functional._default_scaled_dot_product_attention = (
    torch.nn.functional.scaled_dot_product_attention
)
torch.nn.functional.scaled_dot_product_attention = scaled_dot_product_attention
