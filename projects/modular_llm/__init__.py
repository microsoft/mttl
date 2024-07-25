import transformers
import transformers.modeling_flash_attention_utils

from .src.utils.flash_attention_monkey_patch import (
    flash_attention_forward,
    get_unpad_data,
)

transformers.modeling_flash_attention_utils._get_unpad_data = get_unpad_data
transformers.modeling_flash_attention_utils._flash_attention_forward = (
    flash_attention_forward
)
