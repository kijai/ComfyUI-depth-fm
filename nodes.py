import os
import torch
from .depthfm import DepthFM
import folder_paths
import utils
import model_management
from contextlib import nullcontext

def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError

class Depth_fm:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "depthfm_model": (folder_paths.get_filename_list("checkpoints"),),
            "images": ("IMAGE",),
            "steps": ("INT", {"default": 4}),
            "ensemble_size": ("INT", {"default": 1}),
            "dtype": (
                    [
                        'fp32',
                        'fp16',
                        'bf16',
                    ], {
                        "default": 'fp16'
                    }),
            "invert": ("BOOLEAN", {"default": True}),
            "per_batch": ("INT", {"default": 16, "min": 1, "max": 4096, "step": 1}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "process"
    CATEGORY = "depth_fm"

    def process(self, depthfm_model, images, ensemble_size, steps, dtype, invert, per_batch):
        device = model_management.get_torch_device()
        dtype = convert_dtype(dtype)

        custom_config = {
            "model_path": depthfm_model,
            "dtype": dtype,
        }
        if not hasattr(self, "model") or custom_config != self.current_config:
            self.current_config = custom_config
            DEPTHFM_MODEL_PATH = folder_paths.get_full_path("checkpoints", depthfm_model)
            self.model = DepthFM(DEPTHFM_MODEL_PATH)
            self.model.eval().to(dtype).to(device)

        images = images.permute(0, 3, 1, 2)
        images = images * 2.0 - 1.0
        images = images.to(device)

        pbar = utils.ProgressBar(images.shape[0])

        autocast_condition = not model_management.is_device_mps(device)
        with torch.autocast(model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            depth_list = []            
            for start_idx in range(0, images.shape[0], per_batch):
                sub_images = self.model.predict_depth(images[start_idx:start_idx+per_batch], num_steps=steps, ensemble_size=ensemble_size)
                depth_list.append(sub_images.cpu())
                batch_count = sub_images.shape[0]        
                pbar.update(batch_count)
        
        depth = torch.cat(depth_list, dim=0)
        print(depth.min(), depth.max())
        depth = depth.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).cpu()
        if invert:
            depth = 1.0 - depth
        return (depth,)
    
NODE_CLASS_MAPPINGS = {
    "Depth_fm": Depth_fm,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Depth_fm": "Depth_fm",
}
