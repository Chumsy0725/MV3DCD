#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
import cv2
import os
import torch 
from gaussian_renderer import gsplat_render_mask as render
import sys
from scene import Scene, GaussianModel
from scene.cameras import Camera
from tqdm import tqdm
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import numpy as np
from PIL import Image
import random


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # If you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument('--test_iteration', type=int, default=None)
    parser.add_argument('--skip_training', action='store_true', default=False)
    parser.add_argument('--splatpose', action='store_true', default=False)
    parser.add_argument('--change', type=str , default='V1')
    parser.add_argument('--aug', type=bool, default=False)
    parser.add_argument('--mask', type=bool, default=False)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)

    set_seed(42)
  

    iteration = args.test_iteration if args.test_iteration else args.iterations
    dataset = lp.extract(args)
    first_iter = 0
    print(args.model_path)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False, change=args.change)
    train_viewpoint_stack = scene.getTrainCameras().copy()
    test_viewpoint_stack = scene.getTestCameras().copy()

    folder_n = "renders"
    os.makedirs(os.path.join(scene.model_path, folder_n), exist_ok=True)
    os.makedirs(os.path.join(scene.model_path, folder_n, "reference"), exist_ok=True)
    os.makedirs(os.path.join(scene.model_path, folder_n, "query"), exist_ok=True)
    os.makedirs(os.path.join(scene.model_path, folder_n, "alpha_masks"), exist_ok=True)
    os.makedirs(os.path.join(scene.model_path, folder_n, "binary_masks"), exist_ok=True) if args.mask else None

    viewpoint_stack = test_viewpoint_stack if args.mask else train_viewpoint_stack + test_viewpoint_stack

    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.load_ply(os.path.join(scene.model_path, "point_cloud", "iteration_{}".format(iteration), "point_cloud.ply"))

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    for i in tqdm(range(len(viewpoint_stack))):

        test_viewpoint = viewpoint_stack[i]

        q_image = test_viewpoint.original_image.to("cuda")

        test_view = Camera(colmap_id=0000000,
                        R=test_viewpoint.R,
                        T=test_viewpoint.T,
                        FoVx=test_viewpoint.FoVx, FoVy=test_viewpoint.FoVy,
                        image=test_viewpoint.original_image, gt_alpha_mask=None,
                        image_name="query", uid=0000000, change_mask=None, alpha_mask=None,)
        with torch.no_grad():
            render_pkg = render(test_view, gaussians, pp.extract(args), background)

        
        r_image = render_pkg["render"]
        alpha = render_pkg["alpha"]

        if args.mask:
            final_image_mask = render_pkg['change_mask']

            alpha_mask = test_viewpoint.alpha_mask
            alpha_mask = (alpha_mask - alpha_mask.min()) / (alpha_mask.max() - alpha_mask.min())
            alpha_mask = (alpha_mask > 0.5).astype(np.float32)

            final_image_mask = final_image_mask.permute(1, 2, 0).cpu().numpy() * np.expand_dims(alpha_mask, axis=-1)
            final_image_mask = np.clip(final_image_mask, 0, 1)
            final_image_mask = (final_image_mask * 255).astype(np.uint8)

            binary_mask = (final_image_mask > 127).astype(np.float32)
            binary_mask = Image.fromarray((binary_mask * 255).astype(np.uint8))
            gray_binary_mask = binary_mask.convert('L')
            gray_binary_mask.save(os.path.join(scene.model_path, folder_n, "binary_masks", "{}.png".format(test_viewpoint.image_name)))

            continue

        final_image = r_image.permute(1, 2, 0).cpu().numpy()
        final_image = np.clip(final_image, 0, 1)
        final_image = (final_image * 255).astype(np.uint8)
        final_image = cv2.cvtColor(final_image, cv2.COLOR_BGR2RGB)
        if not args.aug:
            cv2.imwrite(os.path.join(scene.model_path, folder_n, "reference", "{}.png".format(test_viewpoint.image_name)), final_image)
        elif args.aug:
            cv2.imwrite(os.path.join(scene.model_path, folder_n, "query", "{}.png".format(test_viewpoint.image_name)), final_image)


        q_image = q_image.permute(1, 2, 0).cpu().numpy()
        q_image = np.clip(q_image, 0, 1)
        q_image = (q_image * 255).astype(np.uint8)
        q_image = cv2.cvtColor(q_image, cv2.COLOR_BGR2RGB)
        if not args.aug:   
            cv2.imwrite(os.path.join(scene.model_path, folder_n, "query", "{}.png".format(test_viewpoint.image_name)), q_image)

        # save the alpha mask
        alpha = alpha.squeeze().cpu().numpy()
        alpha = (alpha * 255).astype(np.uint8)
        if "test" in test_viewpoint.image_name and not args.aug:
            cv2.imwrite(os.path.join(scene.model_path, folder_n, "alpha_masks", "{}.jpg".format(test_viewpoint.image_name)), alpha)

