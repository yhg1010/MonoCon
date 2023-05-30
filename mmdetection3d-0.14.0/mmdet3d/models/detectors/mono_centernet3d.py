import mmcv
import numpy as np
import torch
from mmcv.parallel import DataContainer as DC
from os import path as osp

from mmdet3d.core import (CameraInstance3DBoxes, bbox3d2result,
                          mono_cam_box2vis, show_multi_modality_result)

from mmdet.models.builder import DETECTORS
from mmdet.models.detectors.single_stage import SingleStageDetector


@DETECTORS.register_module()
class CenterNetMono3D(SingleStageDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        assert bbox_head is not None
        super(CenterNetMono3D, self).__init__(backbone, neck, bbox_head, train_cfg,
                                              test_cfg, pretrained)

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_3d,
                      gt_labels_3d,
                      centers2d,
                      depths,
                      gt_kpts_2d=None,
                      gt_kpts_valid_mask=None,
                      attr_labels=None,
                      gt_bboxes_ignore=None,
                      **kwargs):
        x = self.extract_feat(img)
        losses = self.bbox_head.forward_train(x, img_metas, gt_bboxes,
                                              gt_labels, gt_bboxes_3d,
                                              gt_labels_3d, centers2d, depths,
                                              gt_kpts_2d, gt_kpts_valid_mask,
                                              attr_labels, gt_bboxes_ignore,
                                              **kwargs)
        return losses

    def simple_test(self, img, img_metas, rescale=False):
        x = self.extract_feat(img)
        #import ipdb; ipdb.set_trace()
        outs = self.bbox_head(x, img_metas)
        bbox_outputs = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)

        if not self.bbox_head.pred_bbox2d:
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels = bbox_output
                bbox_img.append(bbox3d2result(bboxes, scores, labels))
        else:
            from mmdet.core import bbox2result
            bbox2d_img = []
            bbox_img = []
            for bbox_output in bbox_outputs:
                bboxes, scores, labels, bboxes2d = bbox_output
                bbox2d_img.append(bbox2result(bboxes2d, labels, self.bbox_head.num_classes))
                bbox_img.append(bbox3d2result(bboxes, scores, labels))

        bbox_list = [dict() for i in range(len(img_metas))]
        for result_dict, img_bbox in zip(bbox_list, bbox_img):
            result_dict['img_bbox'] = img_bbox
        if self.bbox_head.pred_bbox2d:
            for result_dict, img_bbox2d in zip(bbox_list, bbox2d_img):
                result_dict['img_bbox2d'] = img_bbox2d
        return bbox_list

    def aug_test(self, imgs, img_metas, rescale=True):
        raise NotImplementedError

    def show_results(self, data, result, out_dir, show=False):
        for batch_id in range(len(result)):
            if isinstance(data['img_metas'][0], DC):
                img_filename = data['img_metas'][0]._data[0][batch_id][
                    'filename']
                #import ipdb; ipdb.set_trace()
                cam2img = data['img_metas'][0]._data[0][batch_id]['cam_intrinsic']
            elif mmcv.is_list_of(data['img_metas'][0], dict):
                img_filename = data['img_metas'][0][batch_id]['filename']
                cam2img = data['img_metas'][0][batch_id]['cam2img']
            else:
                ValueError(
                    f"Unsupported data type {type(data['img_metas'][0])} "
                    f'for visualization!')
            img = mmcv.imread(img_filename)
            file_name = osp.split(img_filename)[-1].split('.')[0]

            assert out_dir is not None, 'Expect out_dir, got none.'

            pred_bboxes = result[batch_id]['img_bbox']['boxes_3d']
            assert isinstance(pred_bboxes, CameraInstance3DBoxes), \
                f'unsupported predicted bbox type {type(pred_bboxes)}'

            show_multi_modality_result(img,
                                       None,
                                       pred_bboxes,
                                       cam2img,
                                       out_dir,
                                       file_name,
                                       'camera',
                                       show=show)    
