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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import time

def render_set(model_path, name, iteration, views, gaussians, pipeline, background):

    
    render_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_color")
    gts_color_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_color")
    render_thermal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders_thermal")
    gts_thermal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt_thermal")
    makedirs(render_color_path, exist_ok=True)
    makedirs(gts_color_path, exist_ok=True)
    makedirs(render_thermal_path, exist_ok=True)
    makedirs(gts_thermal_path, exist_ok=True)

    for idx, view in enumerate(tqdm(views, desc="Rendering progress")):


        image = render(view, gaussians, pipeline, background)["render_color"]
        thermal = render(view, gaussians, pipeline, background)["render_thermal"]

        gt_image = view.original_image.cuda()
        gt_thermal = view.original_thermal.cuda()
        torchvision.utils.save_image(image, os.path.join(render_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_image, os.path.join(gts_color_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(thermal, os.path.join(render_thermal_path, '{0:05d}'.format(idx) + ".png"))
        torchvision.utils.save_image(gt_thermal, os.path.join(gts_thermal_path, '{0:05d}'.format(idx) + ".png"))
        

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)