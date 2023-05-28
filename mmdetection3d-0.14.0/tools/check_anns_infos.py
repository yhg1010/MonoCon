import copy
import os

import cv2
import numpy as np
import torch

from mmdet3d.core.bbox import CameraInstance3DBoxes

MEAN = torch.tensor([123.675, 116.28, 103.53])
STD = torch.tensor([58.395, 57.12, 57.375])
mean = MEAN.view(3, 1, 1)
std = STD.view(3, 1, 1)


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

def plot_rect3d_on_img(
    img,
    num_rects,
    rect_corners,
    instance_index=None,
    show_corner=False,
    color=(0, 255, 0),
    thickness=1,
):
    line_indices = (
        (0, 1),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 5),
        (3, 2),
        (3, 7),
        (4, 5),
        (4, 7),
        (2, 6),
        (5, 6),
        (6, 7),
    )

    if np.min(rect_corners) < 0:
        color = BLUE
        thickness = 2

    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        for start, end in line_indices:
            if corners[start, 0] < 0 or corners[end, 0] < 0:
                continue
            cv2.line(
                img,
                (corners[start, 0], corners[start, 1]),
                (corners[end, 0], corners[end, 1]),
                color,
                thickness,
                cv2.LINE_AA,
            )
    if show_corner:
        for i in range(8):
            x = int(rect_corners[0][i][0])
            y = int(rect_corners[0][i][1])
            img = cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)
    if instance_index is not None:
        strs_info = f"{instance_index}"
        x = int(rect_corners[0][4][0])
        y = int(rect_corners[0][4][1])
        img = cv2.putText(img, strs_info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    return img.astype(np.uint8)


def draw_bbox(image, bbox, show_strs="", color=(255, 255, 0), thickness=1):
    x1 = round(bbox[0])
    y1 = round(bbox[1])
    x2 = round(bbox[2]) + round(bbox[0])
    y2 = round(bbox[3]) + round(bbox[1])
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
    image = cv2.putText(
        image,
        show_strs,
        (x1, y2),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 255),
        thickness,
    )
    return image


def get_corners_2d(gt_bboxes_3d, cam_intrinsic):
    #import ipdb; ipdb.set_trace()
    corners_3d = gt_bboxes_3d.corners.reshape(-1, 3)
    corners_3d_pad = torch.cat(
        [corners_3d, corners_3d.new_ones((corners_3d.shape[0], 1))], dim=-1
    )
    point_2d = corners_3d_pad @ cam_intrinsic.T
    uv_origin = point_2d[..., :2] / point_2d[..., 2:3]
    corners_2d = uv_origin[..., :2].reshape(-1, 8, 2).numpy()
    return corners_2d




def check(result):

    img_tensor = result["img"].data * self.std + self.mean
    single_image = (
        np.transpose(img_tensor.numpy(), (1, 2, 0)).astype(np.uint8).copy()
    )
    gt_bboxes = result["gt_bboxes"].data.cpu().numpy()
    gt_bboxes_3d = result["gt_bboxes_3d"].data


    for bbox in gt_bboxes:
        single_image = draw_bbox(single_image, bbox)
    import ipdb; ipdb.set_trace()
    
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
    file_name = os.path.basename(result["img_info"]["filename"])
    save_image(os.path.join(self.save_prefix, file_name), single_image)

    return result


import json
F = open('/mnt/vepfs/Perception/perception-public/QcraftDataset/Mono3d_20230119/monocon_v2_infos_val2_mono3d.coco.json', 'r')
content = json.load(F)

annos_len = len(content['annotations'])
annos = []

img_id = content['images'][0]['id']

for i in range(annos_len):
    if content['annotations'][i]['image_id'] == img_id:
        annos.append(content['annotations'][i])

root_path = '/mnt/vepfs/Perception/perception-public/QcraftDataset/Mono3d_20230119/'
img_path = root_path + 'image/' + img_id + '.jpg'

img = cv2.imread(img_path)

for i in range(len(annos)):
    img = draw_bbox(img, annos[i]['bbox'])

# gt_bboxes_3d = []

# for i in range(len(annos)):
#     gt_bboxes_3d.append(annos[i]['bbox_cam3d'])

# gt_bboxes_3d = torch.tensor(np.array(gt_bboxes_3d))
cam_intrinsic = torch.tensor(content["images"][0]["cam_intrinsic"])
# single_corners_2d = get_corners_2d(gt_bboxes_3d, cam_intrinsic)

# for idx, curr_corners_2d in enumerate(single_corners_2d):
#         curr_corners_2d = curr_corners_2d.reshape(1, 8, 2)
#         single_image = plot_rect3d_on_img(
#             single_image,
#             1,
#             curr_corners_2d,
#             idx,
#             show_corner=False,
#         )


gt_bboxes_cam3d = []
for i in range(len(annos)):
    bbox_cam3d = np.array(annos[i]['bbox_cam3d']).reshape(1, -1)
    # change orientation to local yaw
    bbox_cam3d[0, 6] = -np.arctan2(
        bbox_cam3d[0, 0], bbox_cam3d[0, 2]) + bbox_cam3d[0, 6]
    #velo_cam3d = np.array(annos[i]['velo_cam3d']).reshape(1, 2)
    #nan_mask = np.isnan(velo_cam3d[:, 0])
    #velo_cam3d[nan_mask] = [0.0, 0.0]
    #bbox_cam3d = np.concatenate([bbox_cam3d, velo_cam3d], axis=-1)
    gt_bboxes_cam3d.append(bbox_cam3d.squeeze())

gt_bboxes_cam3d = np.array(gt_bboxes_cam3d, dtype=np.float32)
gt_bboxes_cam3d = CameraInstance3DBoxes(
            gt_bboxes_cam3d,
            box_dim=gt_bboxes_cam3d.shape[-1],
            origin=(0.5, 0.5, 0.5))

single_corners_2d = get_corners_2d(gt_bboxes_cam3d, cam_intrinsic)   

for idx, curr_corners_2d in enumerate(single_corners_2d):
        curr_corners_2d = curr_corners_2d.reshape(1, 8, 2)
        img = plot_rect3d_on_img(
            img,
            1,
            curr_corners_2d,
            idx,
            show_corner=False,
        )

print(len(annos))

cv2.imwrite('/root/code/test.jpg', img)
