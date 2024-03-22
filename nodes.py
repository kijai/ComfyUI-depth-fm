import os
import torch
import torch.nn.functional as F
from .depthfm import DepthFM
import folder_paths
import comfy.utils
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
            "vae": ("VAE",),
            "depthfm_model": (folder_paths.get_filename_list("checkpoints"),),
            "images": ("IMAGE",),
            "steps": ("INT", {"default": 4, "min": 1, "max": 200, "step": 1}),
            "ensemble_size": ("INT", {"default": 1, "min": 1, "max": 200, "step": 1}),
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

    def process(self, depthfm_model, vae, images, ensemble_size, steps, dtype, invert, per_batch):
        device = model_management.get_torch_device()
        dtype = convert_dtype(dtype)
        
        custom_config = {
            "model_path": depthfm_model,
            "dtype": dtype,
        }
        if not hasattr(self, "model") or custom_config != self.current_config:
            self.current_config = custom_config
            DEPTHFM_MODEL_PATH = folder_paths.get_full_path("checkpoints", depthfm_model)
            self.model = DepthFM(vae, DEPTHFM_MODEL_PATH)
            self.model.eval().to(dtype).to(device)

        images = images.permute(0, 3, 1, 2)
        images = images * 2.0 - 1.0

        B, C, H, W = images.shape
        orig_H, orig_W = H, W
        if W % 64 != 0:
            W = W - (W % 64)
        if H % 64 != 0:
            H = H - (H % 64)
        if orig_H % 64 != 0 or orig_W % 64 != 0:
            images = F.interpolate(images, size=(H, W), mode="bicubic")
            
        images = images.to(device)

        pbar = comfy.utils.ProgressBar(images.shape[0])

        autocast_condition = not model_management.is_device_mps(device)
        with torch.autocast(model_management.get_autocast_device(device), dtype=dtype) if autocast_condition else nullcontext():
            depth_list = []            
            for start_idx in range(0, images.shape[0], per_batch):
                sub_images = self.model.predict_depth(images[start_idx:start_idx+per_batch], num_steps=steps, ensemble_size=ensemble_size)
                depth_list.append(sub_images.cpu())
                batch_count = sub_images.shape[0]        
                pbar.update(batch_count)
        
        depth = torch.cat(depth_list, dim=0)
        depth = depth.repeat(1, 3, 1, 1).permute(0, 2, 3, 1).cpu()

        final_H = (orig_H // 2) * 2
        final_W = (orig_W // 2) * 2

        if depth.shape[1] != final_H or depth.shape[2] != final_W:
            depth = F.interpolate(depth.permute(0, 3, 1, 2), size=(final_H, final_W), mode="bicubic").permute(0, 2, 3, 1)
          
        if invert:
            depth = 1.0 - depth
       
        depth = torch.clamp(depth, 0.0, 1.0)
        return (depth,)
    
NODE_CLASS_MAPPINGS = {
    "Depth_fm": Depth_fm,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Depth_fm": "Depth_fm",
}
