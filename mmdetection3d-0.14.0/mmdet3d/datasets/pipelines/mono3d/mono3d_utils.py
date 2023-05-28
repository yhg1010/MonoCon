import os

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

np.set_printoptions(suppress=True)


COLOR_LIST = [(255, 255, 255), (255, 255, 0), (0, 255, 255)]


RED = (0, 0, 255)
GREEN = (0, 255, 0)
BLUE = (255, 0, 0)
YELLOW = (0, 255, 255)
WHITE = (255, 255, 255)

bev_x_range_max = [-150, +150]
bev_z_range_max = [-10, +250]

bev_x_range = [-50, +50]
bev_z_range = [-10, +150]


bev_scale = 10


def show_image(image, figsize=(10, 10)):
    print("image shape:", image.shape)
    plt.figure(figsize=figsize)
    # plt.imshow(image[:,:,::-1])
    plt.imshow(image)
    plt.axis("off")
    plt.show()
    plt.close()


def draw_bbox(image, bbox, show_strs="", color=(255, 255, 0), thickness=1):
    x1 = round(bbox[0])
    y1 = round(bbox[1])
    x2 = round(bbox[2])
    y2 = round(bbox[3])
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


def save_image(file, image):
    dirname = os.path.dirname(file)
    os.makedirs(dirname, exist_ok=True)
    cv2.imwrite(file, image[..., ::-1])


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


def interpolate_3d_points(a, b, num_points):
    a = np.array(a)
    b = np.array(b)
    t = np.linspace(0, 1, num_points).reshape(-1, 1)
    points = a + t * (b - a)
    return points


def get_truncation_coefficient(gt_bboxes_3d, cam_intrinsic):
    num_instance = gt_bboxes_3d.tensor.shape[0]
    trunc_coeff_array = np.ones(num_instance)
    inter_2d_points_array = []
    for curr_idx in range(num_instance):
        curr_corners_3d = gt_bboxes_3d.corners[curr_idx]
        front_3d_center = np.mean(curr_corners_3d[4:8, :].numpy(), axis=0)
        back_3d_center = np.mean(curr_corners_3d[0:4, :].numpy(), axis=0)

        inter_3d_points = interpolate_3d_points(front_3d_center, back_3d_center, 20)
        corners_3d = torch.from_numpy(inter_3d_points).reshape(-1, 3)
        corners_3d_pad = torch.cat(
            [corners_3d, corners_3d.new_ones((corners_3d.shape[0], 1))], dim=-1
        )
        point_2d = corners_3d_pad @ cam_intrinsic.T
        uv_origin = point_2d[..., :2] / point_2d[..., 2:3]
        inter_2d_points = uv_origin[..., :2].reshape(-1, 2).numpy()
        inter_2d_points_array.append(inter_2d_points)
        x1, x2, y1, y2 = 0, 1024, 0, 512

        within_x_range = (inter_2d_points[:, 0] >= x1) & (inter_2d_points[:, 0] <= x2)
        within_y_range = (inter_2d_points[:, 1] >= y1) & (inter_2d_points[:, 1] <= y2)
        within_both_ranges = within_x_range & within_y_range

        count = np.sum(within_both_ranges)
        trunc_coeff = count / 20
        trunc_coeff_array[curr_idx] = trunc_coeff

    return trunc_coeff_array, np.array(inter_2d_points_array)
