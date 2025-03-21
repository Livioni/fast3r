from fast3r.models.fast3r import Fast3R
from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images
import torch
import json, os
import numpy as np
from fast3r.dust3r.utils.device import to_numpy
from copy import copy
from fast3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from tools.misc import scale_intrinsics, pose_unreal2opencv
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

############################################################
# 1) 工具函数
############################################################

def read_params_from_json(root_path, files, old_size=(1920, 1080), new_size=(512, 288)):
    intrinsics = []
    extrinsics = []
    for parmas_file in files:
        file_path = os.path.join(root_path, parmas_file)
        # 读取 JSON
        with open(file_path, "r") as f:
            data = json.load(f)
        K = np.around(np.array(data["intrinsic"]["K"]),decimals=4)
        T = np.around(np.array(data["extrinsic"]["T"]),decimals=4)
        
        K = scale_intrinsics(K, old_size, new_size)
        T = pose_unreal2opencv(T)
        intrinsics.append(K)
        extrinsics.append(T)
    return intrinsics, extrinsics


# -----------------------------
# 以下仅作示例调用
if __name__ == "__main__":
    schedule = 'linear'
    lr = 0.01
    niter = 500
    # 加载模型
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Fast3R.from_pretrained("jedyang97/Fast3R_ViT_Large_512").to(device)
    lit_module = MultiViewDUSt3RLitModule.load_for_inference(model)
    model.eval()
    lit_module.eval()
    # 示例：读取多张图像（8个相机）
    filepath = "/media/xns/xns01/CARLA/Carla-0.10.0-Linux-Shipping/Videos/town10/002506"
    filelist = [f"{filepath}/camera_{i}.png" for i in range(8)]
    parmas_file_path = "source/carla/town10/params"
    params_files = sorted(os.listdir(parmas_file_path))
    
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


