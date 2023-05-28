import copy
import os

import cv2
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from mmdet3d.core.bbox import points_cam2img

from mmdet.datasets.builder import PIPELINES
from mmdet.models.utils import gaussian_radius, gen_gaussian_target
from .mono3d_utils import (
    draw_bbox,
    get_corners_2d,
    get_truncation_coefficient,
    plot_rect3d_on_img,
    save_image,
    show_image,
)


@PIPELINES.register_module()
class Mono3dVisTools(object):
    """
    For the Mono3d visualization script, it is recommended to use it in the pipeline.
    If you want to visualize in the network forward, you need to modify some code yourself.
    Args:
        debug_mode: If enabled, it will visualize training data and labels
        after_bundle: If used after DefaultFormatBundle3D, set to True
        save_prefix: If a save path is set, the visualization results will be saved

    some other examples:
    from qcraft.datasets.pipelines.mono3d.transforms_mono3d import Mono3dVisTools
    mono3d_vis = Mono3dVisTools(debug_mode=True, after_bundle=False)
    results = mono3d_vis(results)
    """

    def __init__(self, debug_mode=False, after_bundle=True, save_prefix=None):
        self.debug_mode = debug_mode
        MEAN = torch.tensor([123.675, 116.28, 103.53])
        STD = torch.tensor([58.395, 57.12, 57.375])
        self.mean = MEAN.view(3, 1, 1)
        self.std = STD.view(3, 1, 1)
        self.after_bundle = after_bundle
        self.save_prefix = save_prefix

    def __call__(self, result):
        if not self.debug_mode:
            return result

        if self.after_bundle:
            img_tensor = result["img"].data * self.std + self.mean
            single_image = (
                np.transpose(img_tensor.numpy(), (1, 2, 0)).astype(np.uint8).copy()
            )
            gt_bboxes = result["gt_bboxes"].data.cpu().numpy()
            gt_bboxes_3d = result["gt_bboxes_3d"].data
        else:
            single_image = result["img"][..., ::-1].copy()
            gt_bboxes = result["gt_bboxes"]
            gt_bboxes_3d = result["gt_bboxes_3d"]

        for bbox in gt_bboxes:
            single_image = draw_bbox(single_image, bbox)
        #import ipdb; ipdb.set_trace()
        
        cam_intrinsic = torch.tensor(result["img_info"]["cam_intrinsic"])
        try:
            single_corners_2d = get_corners_2d(gt_bboxes_3d, cam_intrinsic)
        except:
            return result
        for idx, curr_corners_2d in enumerate(single_corners_2d):
            curr_corners_2d = curr_corners_2d.reshape(1, 8, 2)
            single_image = plot_rect3d_on_img(
                single_image,
                1,
                curr_corners_2d,
                idx,
                show_corner=False,
            )
        print("result['img_info']['filename']", result["img_info"]["filename"])
        if self.save_prefix is None:
            show_image(single_image, figsize=(10.24 * 2, 5.12 * 2))
        else:
            file_name = os.path.basename(result["img_info"]["filename"])
            save_image(os.path.join(self.save_prefix, file_name), single_image)

        return result

