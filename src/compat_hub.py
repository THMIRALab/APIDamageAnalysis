import os
import sys
import importlib.util
from typing import Dict, List, Optional, Union, BinaryIO
import torch

# Check if huggingface_hub is installed
if importlib.util.find_spec("huggingface_hub") is not None:
    import huggingface_hub
    
    # Add missing functions if they don't exist
    if not hasattr(huggingface_hub, "split_torch_state_dict_into_shards"):
        def split_torch_state_dict_into_shards(
            state_dict: Dict[str, torch.Tensor],
            max_shard_size: str = "10GB",
            safe_serialization: bool = True,
        ) -> List[Dict[str, torch.Tensor]]:
            """
            Compatibility implementation of split_torch_state_dict_into_shards
            """
            print("Using compatibility function for split_torch_state_dict_into_shards")
            # This is a simplified implementation - in practice, it would just 
            # return the whole state dict as one shard since we don't need actual sharding
            return [state_dict]
        
        # Add the function to the huggingface_hub module
        huggingface_hub.split_torch_state_dict_into_shards = split_torch_state_dict_into_shards
        
    # Add other potentially missing functions
    '''if not hasattr(huggingface_hub, "cached_download"):
        def cached_download(*args, **kwargs):
            """Compatibility wrapper for cached_download"""
            from huggingface_hub import snapshot_download
            return snapshot_download(*args, **kwargs)
            
        huggingface_hub.cached_download = cached_download'''

# Print success message
print("HuggingFace Hub compatibility layer applied successfully")