import torch
import numpy as np
import json

import torch
import numpy as np
import json
import random

import torch
import json
import cv2
import os
import matplotlib
import matplotlib.pyplot as plt 

from collections import defaultdict
from collections import OrderedDict

np.set_printoptions(suppress=True)

from mmdet3d.core.bbox import CameraInstance3DBoxes


COLOR_LIST = [(255, 255, 255), (255, 255, 0), (0, 255, 255)]

def show_image(image, name, figsize=(10,10)):
    print('image shape:', image.shape)
    cv2.imwrite('./'+name+'.png', image)
    plt.figure(figsize=figsize)
    plt.imshow(image[:,:,::-1])
    plt.axis('off')
    plt.show()
    plt.close()

def read_image(file, isshow=False):
    image = cv2.imread(file)
    if(isshow):
        show_image(image)
    print('read image size', image.shape)
    return image

def save_image(file, image):
    dirname = os.path.dirname(file)
    os.makedirs(dirname, exist_ok=True)
    cv2.imwrite(file, image)

def draw_bbox(image, bbox):
    x1 = round(bbox[0])
    y1 = round(bbox[1])
    x2 = round(bbox[0] + bbox[2])
    y2 = round(bbox[1] + bbox[3])
    cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_LIST[2], 2)
    return image

def draw_point(image, point):
    x1 = round(point[0])
    y1 = round(point[1])
    x2 = round(point[0] + 10)
    y2 = round(point[1] + 10)
    cv2.rectangle(image, (x1, y1), (x2, y2), COLOR_LIST[2], 2)
    return image

def plot_rect3d_on_img(img,
                       num_rects,
                       rect_corners,
                       instance_index=None,
                       show_corner=False,
                       color=(0, 255, 0),
                       thickness=1):
    line_indices = ((0, 1), (0, 3), (0, 4), (1, 2), (1, 5), (3, 2), (3, 7), (4, 5), (4, 7), (2, 6), (5, 6), (6, 7))
    print("image shape", img.shape)
    if np.min(rect_corners) < 0:
        color = RED
        thickness = 2
    
    for i in range(num_rects):
        corners = rect_corners[i].astype(np.int32)
        for start, end in line_indices:
            if (corners[start, 0]<0 or corners[end, 0]<0):
                continue
            cv2.line(img, (corners[start, 0], corners[start, 1]),
                     (corners[end, 0], corners[end, 1]), color, thickness,
                     cv2.LINE_AA)

    if show_corner:
        for i in range(8):
            x = int(rect_corners[0][i][0])
            y = int(rect_corners[0][i][1])
            img = cv2.putText(img, str(i), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, BLUE, 2)
    if instance_index is not None:
        strs_info = f'{instance_index}'
        x = int(rect_corners[0][4][0])
        y = int(rect_corners[0][4][1])
        img = cv2.putText(img, strs_info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1, RED, 2)

    return img.astype(np.uint8)


def get_mono3d_bboxes(curr_anno, proj_mat):

    bbox2d = curr_anno['bbox']
    bbox_tight = curr_anno['bbox_tight']
    center2d = curr_anno['center2d']
    bbox_cam3d = curr_anno['bbox_cam3d']
    center_bbox = [center2d[0], center2d[1], 10, 10]

    tensor = torch.from_numpy(np.array([bbox_cam3d]))
    cam3d_instance = CameraInstance3DBoxes(tensor, origin=(0.5, 0.5, 0.5))
    points_3d = cam3d_instance.corners
    points_3d = points_3d.reshape(-1, 3)
    points_shape = list(points_3d.shape)
    points_shape[-1] = 1
    d1, d2 = proj_mat.shape[:2]
    points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)

    point_2d = points_4 @ proj_mat.T
    uv_origin = point_2d[..., :2] / point_2d[..., 2:3]
    point_2d_res = uv_origin.numpy()[0]
    uv_origin = (uv_origin - 1).round()
    imgfov_pts_2d = uv_origin[..., :2].reshape(1, 8, 2).numpy()

    return points_4, imgfov_pts_2d


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

def plot_rect3d_on_bev(bev_map,
                       num_rects,
                       points_4,
                       instance_index,
                       color=(0, 255, 0),
                       thickness=1):
    # 首先是画校准线 
    # x = 0 和 z = 0
    start = (0, (bev_z_range[1]) * bev_scale)
    end = ((bev_x_range[1]-bev_x_range[0])*bev_scale, (bev_z_range[1]) * bev_scale)
    cv2.line(bev_map, start, end, YELLOW, 2, cv2.LINE_AA)

    start = ((0-bev_x_range[0])*bev_scale, 0)
    end = ((0-bev_x_range[0])*bev_scale, 5000)    
    cv2.line(bev_map, start, end, YELLOW, 2, cv2.LINE_AA)


    point_4_numpy = points_4.numpy()
    # x, y, z 其中 y 代表了高度， x 代表了左右，z 是深度    
    # front_left front_right back_left back_right
    bev_bbox = np.array([
        point_4_numpy[5][[0,2]], 
        point_4_numpy[4][[0,2]],
        point_4_numpy[1][[0,2]],
        point_4_numpy[0][[0,2]],
    ])

    assert np.min(bev_bbox[:, 0]) > bev_x_range_max[0], (bev_bbox, bev_x_range)
    assert np.max(bev_bbox[:, 0]) < bev_x_range_max[1], (bev_bbox, bev_x_range)
    assert np.min(bev_bbox[:, 1]) > bev_z_range_max[0], (bev_bbox, bev_z_range)
    assert np.max(bev_bbox[:, 1]) < bev_z_range_max[1], (bev_bbox, bev_z_range)

    bev_bbox = bev_bbox * bev_scale
    bev_bbox[:, 0] -= bev_x_range[0]*bev_scale
    bev_bbox[:, 1] -= bev_z_range[0]*bev_scale

    # x z 坐标系 to u v 坐标系 
    # x = u, v = H of bev_map - z 
    bev_bbox[:,1] = bev_map.shape[0] - bev_bbox[:,1]
    bev_bbox = bev_bbox.astype(np.int32)

    for start, end in ((1, 3), (3, 2), (2, 0), (0, 1)):
        if start==0 and end == 1:
            color = RED
        else:
            color = GREEN
        cv2.line(bev_map, (bev_bbox[start, 0], bev_bbox[start, 1]),
                 (bev_bbox[end, 0], bev_bbox[end, 1]), color, thickness,
                 cv2.LINE_AA)

    strs_info = f'{instance_index}'

    x = int(np.mean(bev_bbox[:,0]) + 30)
    y = int(np.mean(bev_bbox[:,1]) + 20)

    bev_map = cv2.putText(bev_map, strs_info, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        WHITE, 3)

    return bev_map


#save_file = '/home/qcraft/git/mmdetection3d/data/nuscenes/nuscenes_infos_val_mono3d.coco.json'
input_file = '/mnt/vepfs/Perception/perception-public/PublicDataset/kitti_copy/kitti_object_detection/kitti_infos_train_mono3d.coco.json'
data = json.load(open(input_file))
# data.keys()
# data['images'][0]
# data['annotations'][0]
# data['annotations'] = [data['annotations'][0]]
# data['images'] = [data['images'][0]]

# with open(save_file, 'w') as f:
#     json.dump(data,  f)

from mmdet3d.core import CameraInstance3DBoxes

data_root = '/mnt/vepfs/Perception/perception-public/PublicDataset/kitti_copy/kitti_object_detection'
filename = os.path.join(data_root, 'training/image_2/002297.png')
print(filename)
image = read_image(filename, False)
# bbox2d = data['annotations'][0]['bbox']
# bbox_tight = data['annotations'][0]['bbox_tight']
# center2d = data['annotations'][0]['center2d']
# import ipdb; ipdb.set_trace()

img_dict = {}
for item in data['images']:
    img_dict[item['id']] = item



# for i in range(len(data['annotations'])):
# #bbox_cam3d = data['annotations'][2]['bbox_cam3d']
#     if data['annotations'][i]['image_id'] == 2297:
#         bbox_cam3d = data['annotations'][i]['bbox_cam3d']

#         # center_bbox = [center2d[0], center2d[1], 10, 10]

#         proj_mat = np.array(data['images'][data['annotations'][i]['image_id']]['cam_intrinsic'])
#         #proj_mat = np.array([[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]])
#         # print('bbox2d:', bbox2d)
#         # print('center2d:', center2d)

#         tensor = torch.from_numpy(np.array([bbox_cam3d]))
        
#         #print(tensor)
#         #import ipdb; ipdb.set_trace()
#         ## yaw 
#         tensor[0][6] = -np.arctan2(tensor[0][0],
#                                         tensor[0][2]) + tensor[0][6]


#         cam3d_1 = CameraInstance3DBoxes(tensor , origin=(0.5, 0.5, 0.5))
#         print("camerainstance: ", cam3d_1)
#         print("cam_intrinsic: ", proj_mat)
#         corners = cam3d_1.corners
#         print('corners', corners.shape)
#         points_3d = corners.reshape(-1, 3)

#         points_shape = list(points_3d.shape)
#         points_shape[-1] = 1
#         print(points_shape)
#         d1, d2 = proj_mat.shape[:2]
#         proj_mat = torch.from_numpy(proj_mat).type(torch.float32)
#         if d1 == 3:
#             proj_mat_expanded = torch.eye(
#                 4, device=proj_mat.device, dtype=proj_mat.dtype)
#             proj_mat_expanded[:d1, :d2] = proj_mat
#             proj_mat = proj_mat_expanded
#         points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
#         # print("points_4", points_4.shape, points_4)

#         print(points_4.shape, proj_mat.shape, points_4.dtype, proj_mat.dtype)

        





# for i in range(len(data['annotations'])):
# #bbox_cam3d = data['annotations'][2]['bbox_cam3d']
#     if data['annotations'][i]['image_id'] == 2297:
#         bbox_cam3d = data['annotations'][i]['bbox_cam3d']

#         # center_bbox = [center2d[0], center2d[1], 10, 10]

#         proj_mat = np.array(data['images'][data['annotations'][i]['image_id']]['cam_intrinsic'])
#         #proj_mat = np.array([[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]])
#         # print('bbox2d:', bbox2d)
#         # print('center2d:', center2d)

#         tensor = torch.from_numpy(np.array([bbox_cam3d]))
        
#         #print(tensor)
#         #import ipdb; ipdb.set_trace()
#         ## yaw 
#         tensor[0][6] = -np.arctan2(tensor[0][0],
#                                         tensor[0][2]) + tensor[0][6]


        # cam3d_flip = CameraInstance3DBoxes(tensor , origin=(0.5, 0.5, 0.5))
        # #print("camerainstance: ", cam3d_1)
        # #print("cam_intrinsic: ", proj_mat)
        # corners = cam3d_flip.corners
        # #print('corners', corners.shape)
        # points_3d = corners.reshape(-1, 3)

        # points_shape = list(points_3d.shape)
        # points_shape[-1] = 1

        # proj_mat_flip = proj_mat
        # proj_mat_flip[0, 2] = image.shape[0] - proj_mat_flip[0, 2]
        # cam3d_flip.flip()
        # corners_flip = cam3d_flip.corners
        # point_3d_flip = corners_flip.reshape(-1, 3)
        # proj_mat_flip = torch.from_numpy(proj_mat_flip).type(torch.float32)
        # points_4_flip = torch.cat([point_3d_flip, point_3d_flip.new_ones(points_shape)], dim=-1)

        # point_2d_flip = points_4_flip @ proj_mat_flip.T
        # uv_origin_flip = point_2d_flip[..., :2] / point_2d_flip[..., 2:3]
        # point_2d_res_flip = uv_origin_flip.numpy()[0]
        # #print('point_2d_res', point_2d_res_flip)
        # #uv_origin = (uv_origin - 1).round()
        # imgfov_pts_2d_flip = uv_origin_flip[..., :2].reshape(-1, 8, 2).numpy()
        # image = np.flip(image, 1).copy()
        # image = plot_rect3d_on_img(image, 1, imgfov_pts_2d_flip)
        # # image = plot_rect3d_on_img(image[:, ::-1, :], 1, imgfov_pts_2d_flip)
 
img = np.flip(image, 1).copy()
for i in range(len(data['annotations'])):
#bbox_cam3d = data['annotations'][2]['bbox_cam3d']
    if data['annotations'][i]['image_id'] == 2297:
        bbox_cam3d = data['annotations'][i]['bbox_cam3d']

        # center_bbox = [center2d[0], center2d[1], 10, 10]

        proj_mat = np.array(data['images'][data['annotations'][i]['image_id']]['cam_intrinsic'])
        #proj_mat = np.array([[721.5377, 0.0, 609.5593, 44.85728], [0.0, 721.5377, 172.854, 0.2163791], [0.0, 0.0, 1.0, 0.002745884], [0.0, 0.0, 0.0, 1.0]])
        # print('bbox2d:', bbox2d)
        # print('center2d:', center2d)

        tensor = torch.from_numpy(np.array([bbox_cam3d]))
        
        #print(tensor)
        #import ipdb; ipdb.set_trace()
        ## yaw 
        tensor[0][6] = -np.arctan2(tensor[0][0],
                                        tensor[0][2]) + tensor[0][6]


        cam3d_1 = CameraInstance3DBoxes(tensor , origin=(0.5, 0.5, 0.5))
        print("before flip: ")
        print("camerainstance: ", cam3d_1)
        print("camera intrinsic: ", proj_mat)
        #print("camerainstance: ", cam3d_1)
        #print("cam_intrinsic: ", proj_mat)
        cam3d_1.flip()
        corners = cam3d_1.corners
        #print('corners', corners.shape)
        points_3d = corners.reshape(-1, 3)

        points_shape = list(points_3d.shape)
        points_shape[-1] = 1
        print(points_shape)
        #print("before flip: ", proj_mat)
        proj_mat[0, 2] = image.shape[1] - proj_mat[0, 2]
        d1, d2 = proj_mat.shape[:2]
        proj_mat = torch.from_numpy(proj_mat).type(torch.float32)
        if d1 == 3:
            proj_mat_expanded = torch.eye(
                4, device=proj_mat.device, dtype=proj_mat.dtype)
            proj_mat_expanded[:d1, :d2] = proj_mat
            proj_mat = proj_mat_expanded
        points_4 = torch.cat([points_3d, points_3d.new_ones(points_shape)], dim=-1)
        # print("points_4", points_4.shape, points_4)

        print(points_4.shape, proj_mat.shape, points_4.dtype, proj_mat.dtype)
        point_2d = points_4 @ proj_mat.T
        uv_origin = point_2d[..., :2] / point_2d[..., 2:3]
        point_2d_res = uv_origin.numpy()[0]
        #print('point_2d_res', point_2d_res)
        uv_origin = (uv_origin - 1).round()
        imgfov_pts_2d = uv_origin[..., :2].reshape(1, 8, 2).numpy()
        
        print(imgfov_pts_2d)
        img = plot_rect3d_on_img(img, 1, imgfov_pts_2d)

        
        print("after flip: ", proj_mat)
        print("camerainstance: ", cam3d_1)
        print("camera intrinsic: ", proj_mat)
        print("image shape: ", image.shape)

        

show_image(img, 'flip')

