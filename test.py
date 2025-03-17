from fast3r.models.fast3r import Fast3R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule
from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images
import torch

# Load the model from Hugging Face
model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512")
model = model.to("cuda")

# [Optional] Create a lightweight lightning module wrapper for the model.
# This provides functions to estimate camera poses, evaluate 3D reconstruction, etc.
# See fast3r/viz/demo.py for an example.
lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)

# Set model to evaluation mode
model.eval()
lit_module.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 示例：读取多张图像（8个相机）
filepath = "/media/xns/xns01/CARLA/Carla-0.10.0-Linux-Shipping/Videos/town10/002506"
filelist = [f"{filepath}/camera_{i}.png" for i in range(8)]

imgs = load_images(
    filelist,
    size=512,
    verbose=True,
    rotate_clockwise_90=False, 
    crop_to_landscape=False
)

# 推理
output_dict, profiling_info = inference(
    imgs,
    model,
    device,
    dtype=torch.float32,
    verbose=True,
    profiling=True,
)

# 将输出迁移到 CPU
try:
    for pred in output_dict['preds']:
        for k, v in pred.items():
            if isinstance(v, torch.Tensor):
                pred[k] = v.cpu()
    for view in output_dict['views']:
        for k, v in view.items():
            if isinstance(v, torch.Tensor):
                view[k] = v.cpu()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
except Exception as e:
    print(f"Warning: {e}")
    
lit_module.align_local_pts3d_to_global(
        preds=output_dict['preds'],
        views=output_dict['views'],
        min_conf_thr_percentile=85
    )

raise