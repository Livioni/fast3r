from fast3r.models.fast3r import Fast3R
from fast3r.dust3r.inference_multiview import inference
from fast3r.dust3r.utils.image import load_images
import torch
import json, os
import numpy as np
from fast3r.dust3r.utils.device import to_numpy
from copy import copy
from fast3r.dust3r.cloud_opt import global_aligner, GlobalAlignerMode
from scipy.spatial.transform import Rotation as R
from fast3r.models.multiview_dust3r_module import MultiViewDUSt3RLitModule

############################################################
# 1) 工具函数
############################################################
def scale_intrinsics(K, old_size, new_size):
    new_width, new_height = new_size
    old_width, old_height = old_size
    scale_x = new_width / old_width
    scale_y = new_height / old_height

    K_scaled = copy(K) 
    K_scaled[0,0] *= scale_x
    K_scaled[0,2] *= scale_x
    K_scaled[1,1] *= scale_y
    K_scaled[1,2] *= scale_y
    
    return K_scaled


def pose_unreal2opencv(c2w_mat):
    translation = c2w_mat[:3, 3]
    rot = R.from_matrix(c2w_mat[:3, :3])
    rot_vec = rot.as_rotvec()

    rot_vec_new = rot_vec[[1, 2, 0]]
    rot_vec_new[0] *= -1
    rot_vec_new[2] *= -1

    rot = R.from_rotvec(rot_vec_new)
    
    translation_new = translation[[1, 2, 0]]
    translation_new[1] *= -1

    c2w_mat = np.eye(4)
    c2w_mat[:3, :3] = rot.as_matrix()
    c2w_mat[:3, 3] = translation_new

    rot = np.eye(4)
    rot[1,1]=-1
    rot[2, 2] = -1
    c2w_mat =  rot @ c2w_mat
    return c2w_mat

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

    intrinsics, extrinsics = read_params_from_json(parmas_file_path, params_files)
    # intrinsics, extrinsics = read_params_from_pt(parmas_file_path, parmas_files, old_size=(4608, 3456), new_size=(512, 384))
    # extrinsic_matrixs, center, scale = normalize_cameras(extrinsics)
    scene = global_aligner(output, device=device, mode=GlobalAlignerMode.ModularPointCloudOptimizer, fx_and_fy=True)
    scene.preset_pose(extrinsics,[True] * len(extrinsics))
    scene.preset_intrinsics(intrinsics,[True] * len(intrinsics))
    # scene.preset_focal(focals)

    loss = scene.compute_global_alignment(init="known_poses", niter=niter, schedule=schedule, lr=lr)
    # outfile = get_3D_model_from_scene(outdir="output", silent=False, scene=scene, as_pointcloud=False)



    # retrieve useful values from scene:
    imgs = scene.imgs
    focals = scene.get_focals()
    poses = scene.get_im_poses()
    pts3d = scene.get_pts3d()
    confidence_masks = scene.get_masks()

    # visualize reconstruction
    scene.show()

    # find 2D-2D matches between the two images
    from dust3r.utils.geometry import find_reciprocal_matches, xy_grid
    pts2d_list, pts3d_list = [], []
    for i in range(2):
        conf_i = confidence_masks[i].cpu().numpy()
        pts2d_list.append(xy_grid(*imgs[i].shape[:2][::-1])[conf_i])  # imgs[i].shape[:2] = (H, W)
        pts3d_list.append(pts3d[i].detach().cpu().numpy()[conf_i])
    reciprocal_in_P2, nn2_in_P1, num_matches = find_reciprocal_matches(*pts3d_list)
    print(f'found {num_matches} matches')
    matches_im1 = pts2d_list[1][reciprocal_in_P2]
    matches_im0 = pts2d_list[0][nn2_in_P1][reciprocal_in_P2]

    # visualize a few matches
    import numpy as np
    from matplotlib import pyplot as pl
    n_viz = 10
    match_idx_to_viz = np.round(np.linspace(0, num_matches-1, n_viz)).astype(int)
    viz_matches_im0, viz_matches_im1 = matches_im0[match_idx_to_viz], matches_im1[match_idx_to_viz]

    H0, W0, H1, W1 = *imgs[0].shape[:2], *imgs[1].shape[:2]
    img0 = np.pad(imgs[0], ((0, max(H1 - H0, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img1 = np.pad(imgs[1], ((0, max(H0 - H1, 0)), (0, 0), (0, 0)), 'constant', constant_values=0)
    img = np.concatenate((img0, img1), axis=1)
    pl.figure()
    pl.imshow(img)
    cmap = pl.get_cmap('jet')
    for i in range(n_viz):
        (x0, y0), (x1, y1) = viz_matches_im0[i].T, viz_matches_im1[i].T
        pl.plot([x0, x1 + W0], [y0, y1], '-+', color=cmap(i / (n_viz - 1)), scalex=False, scaley=False)
    pl.show(block=True)
