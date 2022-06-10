# -*- coding: utf-8 -*-
# author: novauto
# time:  2022.04

import argparse
import os
import numpy as np
import pickle
from kitti_object_eval_python import eval
import warnings
from progressbar import *
warnings.filterwarnings("ignore")

# required files: 1.kitti_object_eval_python & 2.eval_infos.pkl & 3.eval_gt_annos.pkl & 4."prediction_path"
# required module: numpy & progressbar & argparse

def parse_config():
    parser = argparse.ArgumentParser(description='arg parser')
    parser.add_argument('--gt_info_path', type=str, default='eval_infos.pkl', help='kitti val infos')
    parser.add_argument('--gt_label_path', type=str, default='eval_gt_annos.pkl', help='kitti val gt_labels')
    parser.add_argument('--pre_path', type=str, default='./output', help='kitti eval predictions')
    args = parser.parse_args()
    check_path([args.gt_info_path, args.gt_label_path, args.pre_path])
    return args

def check_path(lists):
    for i in lists:
        if not os.path.exists(i):
            print(i, ' is not exists')
            exit(0)

def boxes3d_lidar_to_kitti_camera(boxes3d_lidar, calib):
    boxes3d_lidar_copy = boxes3d_lidar
    xyz_lidar = boxes3d_lidar_copy[:, 0:3]
    l, w, h = boxes3d_lidar_copy[:, 3:4], boxes3d_lidar_copy[:, 4:5], boxes3d_lidar_copy[:, 5:6]
    r = boxes3d_lidar_copy[:, 6:7]

    xyz_lidar[:, 2] -= h.reshape(-1) / 2
    xyz_cam = calib.lidar_to_rect(xyz_lidar)
    r = -r - np.pi / 2
    return np.concatenate([xyz_cam, l, h, w, r], axis=-1)

def boxes3d_to_corners3d_kitti_camera(boxes3d, bottom_center=True):
    boxes_num = boxes3d.shape[0]
    l, h, w = boxes3d[:, 3], boxes3d[:, 4], boxes3d[:, 5]
    x_corners = np.array([l / 2., l / 2., -l / 2., -l / 2., l / 2., l / 2., -l / 2., -l / 2], dtype=np.float32).T
    z_corners = np.array([w / 2., -w / 2., -w / 2., w / 2., w / 2., -w / 2., -w / 2., w / 2.], dtype=np.float32).T
    if bottom_center:
        y_corners = np.zeros((boxes_num, 8), dtype=np.float32)
        y_corners[:, 4:8] = -h.reshape(boxes_num, 1).repeat(4, axis=1)  # (N, 8)
    else:
        y_corners = np.array([h / 2., h / 2., h / 2., h / 2., -h / 2., -h / 2., -h / 2., -h / 2.], dtype=np.float32).T

    ry = boxes3d[:, 6]
    zeros, ones = np.zeros(ry.size, dtype=np.float32), np.ones(ry.size, dtype=np.float32)
    rot_list = np.array([[np.cos(ry), zeros, -np.sin(ry)],
                         [zeros, ones, zeros],
                         [np.sin(ry), zeros, np.cos(ry)]])  # (3, 3, N)
    R_list = np.transpose(rot_list, (2, 0, 1))  # (N, 3, 3)

    temp_corners = np.concatenate((x_corners.reshape(-1, 8, 1), y_corners.reshape(-1, 8, 1),
                                   z_corners.reshape(-1, 8, 1)), axis=2)  # (N, 8, 3)
    rotated_corners = np.matmul(temp_corners, R_list)  # (N, 8, 3)
    x_corners, y_corners, z_corners = rotated_corners[:, :, 0], rotated_corners[:, :, 1], rotated_corners[:, :, 2]

    x_loc, y_loc, z_loc = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]

    x = x_loc.reshape(-1, 1) + x_corners.reshape(-1, 8)
    y = y_loc.reshape(-1, 1) + y_corners.reshape(-1, 8)
    z = z_loc.reshape(-1, 1) + z_corners.reshape(-1, 8)

    corners = np.concatenate((x.reshape(-1, 8, 1), y.reshape(-1, 8, 1), z.reshape(-1, 8, 1)), axis=2)

    return corners.astype(np.float32)

def boxes3d_kitti_camera_to_imageboxes(boxes3d, calib, image_shape=None):
    corners3d = boxes3d_to_corners3d_kitti_camera(boxes3d)
    pts_img, _ = calib.rect_to_img(corners3d.reshape(-1, 3))
    corners_in_image = pts_img.reshape(-1, 8, 2)

    min_uv = np.min(corners_in_image, axis=1)  # (N, 2)
    max_uv = np.max(corners_in_image, axis=1)  # (N, 2)
    boxes2d_image = np.concatenate([min_uv, max_uv], axis=1)
    if image_shape is not None:
        boxes2d_image[:, 0] = np.clip(boxes2d_image[:, 0], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 1] = np.clip(boxes2d_image[:, 1], a_min=0, a_max=image_shape[0] - 1)
        boxes2d_image[:, 2] = np.clip(boxes2d_image[:, 2], a_min=0, a_max=image_shape[1] - 1)
        boxes2d_image[:, 3] = np.clip(boxes2d_image[:, 3], a_min=0, a_max=image_shape[0] - 1)

    return boxes2d_image



def load_preds(path):
    data = np.fromfile(path, dtype=np.float32).reshape(-1, 9)
    num = (data[:, -1] > -0.5).sum()
    if num == 0:
        return np.zeros((0, 7)), np.zeros((0,)), np.zeros((0,)).astype(np.int32)
        
    data = data[:num]
    return data[:, :7], data[:, 7], (data[:, 8]+ 0.01).astype(np.int32) + 1

class Calibration(object):
    def __init__(self, P2, R0, Tr_velo2cam):

        self.P2 = P2  # 3 x 4
        self.R0 = R0  # 3 x 3
        self.V2C = Tr_velo2cam  # 3 x 4

        # Camera intrinsics and extrinsics
        self.cu = self.P2[0, 2]
        self.cv = self.P2[1, 2]
        self.fu = self.P2[0, 0]
        self.fv = self.P2[1, 1]
        self.tx = self.P2[0, 3] / (-self.fu)
        self.ty = self.P2[1, 3] / (-self.fv)

    def cart_to_hom(self, pts):
        """
        :param pts: (N, 3 or 2)
        :return pts_hom: (N, 4 or 3)
        """
        pts_hom = np.hstack((pts, np.ones((pts.shape[0], 1), dtype=np.float32)))
        return pts_hom

    def rect_to_lidar(self, pts_rect):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)  # (N, 4)
        R0_ext = np.hstack((self.R0, np.zeros((3, 1), dtype=np.float32)))  # (3, 4)
        R0_ext = np.vstack((R0_ext, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        R0_ext[3, 3] = 1
        V2C_ext = np.vstack((self.V2C, np.zeros((1, 4), dtype=np.float32)))  # (4, 4)
        V2C_ext[3, 3] = 1

        pts_lidar = np.dot(pts_rect_hom, np.linalg.inv(np.dot(R0_ext, V2C_ext).T))
        return pts_lidar[:, 0:3]

    def lidar_to_rect(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_rect: (N, 3)
        """
        pts_lidar_hom = self.cart_to_hom(pts_lidar)
        pts_rect = np.dot(pts_lidar_hom, np.dot(self.V2C.T, self.R0.T))
        # pts_rect = reduce(np.dot, (pts_lidar_hom, self.V2C.T, self.R0.T))
        return pts_rect

    def rect_to_img(self, pts_rect):
        """
        :param pts_rect: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect_hom = self.cart_to_hom(pts_rect)
        pts_2d_hom = np.dot(pts_rect_hom, self.P2.T)
        pts_img = (pts_2d_hom[:, 0:2].T / pts_rect_hom[:, 2]).T  # (N, 2)
        pts_rect_depth = pts_2d_hom[:, 2] - self.P2.T[3, 2]  # depth in rect camera coord
        return pts_img, pts_rect_depth

    def lidar_to_img(self, pts_lidar):
        """
        :param pts_lidar: (N, 3)
        :return pts_img: (N, 2)
        """
        pts_rect = self.lidar_to_rect(pts_lidar)
        pts_img, pts_depth = self.rect_to_img(pts_rect)
        return pts_img, pts_depth

    def img_to_rect(self, u, v, depth_rect):
        """
        :param u: (N)
        :param v: (N)
        :param depth_rect: (N)
        :return:
        """
        x = ((u - self.cu) * depth_rect) / self.fu + self.tx
        y = ((v - self.cv) * depth_rect) / self.fv + self.ty
        pts_rect = np.concatenate((x.reshape(-1, 1), y.reshape(-1, 1), depth_rect.reshape(-1, 1)), axis=1)
        return pts_rect


def generate_prediction_dicts(batch_dict, pred_dicts, class_names, output_path=None):
    def get_template_prediction(num_samples):
        ret_dict = {
            'name': np.zeros(num_samples), 'truncated': np.zeros(num_samples),
            'occluded': np.zeros(num_samples), 'alpha': np.zeros(num_samples),
            'bbox': np.zeros([num_samples, 4]), 'dimensions': np.zeros([num_samples, 3]),
            'location': np.zeros([num_samples, 3]), 'rotation_y': np.zeros(num_samples),
            'score': np.zeros(num_samples), 'boxes_lidar': np.zeros([num_samples, 7])
        }
        return ret_dict

    def generate_single_sample_dict(batch_index, box_dict):
        pred_scores = box_dict['pred_scores']
        pred_boxes = box_dict['pred_boxes']
        pred_labels = box_dict['pred_labels']
        pred_dict = get_template_prediction(pred_scores.shape[0])
        if pred_scores.shape[0] == 0:
            return pred_dict

        calib_info = batch_dict['calib_info']
        image_shape = batch_dict['image_shape']
        P2 = calib_info['P2']
        R0 = calib_info['R0']
        V2C = calib_info['V2C']

        calib = Calibration(P2, R0, V2C)
        pred_boxes_camera = boxes3d_lidar_to_kitti_camera(pred_boxes, calib)
        pred_boxes_img = boxes3d_kitti_camera_to_imageboxes(
            pred_boxes_camera, calib, image_shape=image_shape
        )

        pred_dict['name'] = np.array(class_names)[pred_labels - 1]
        pred_dict['alpha'] = -np.arctan2(-pred_boxes[:, 1], pred_boxes[:, 0]) + pred_boxes_camera[:, 6]
        pred_dict['bbox'] = pred_boxes_img
        pred_dict['dimensions'] = pred_boxes_camera[:, 3:6]
        pred_dict['location'] = pred_boxes_camera[:, 0:3]
        pred_dict['rotation_y'] = pred_boxes_camera[:, 6]
        pred_dict['score'] = pred_scores
        pred_dict['boxes_lidar'] = pred_boxes

        return pred_dict

    annos = []
    for index, box_dict in enumerate(pred_dicts):
        frame_id = batch_dict['frame_id']
        single_pred_dict = generate_single_sample_dict(index, box_dict)
        single_pred_dict['frame_id'] = frame_id
        annos.append(single_pred_dict)
    return annos

def main():
    
    class_names = ['Car', 'Pedestrian', 'Cyclist']
    kitti_val_nums = 3769
    args = parse_config()

    with open(args.gt_label_path, 'rb') as f:
        eval_gt_annos = pickle.load(f)

    pre_tag = sorted(os.listdir(args.pre_path))[0][6:]
    det_annos = []

    widgets = ['Progress: ', Percentage(), ' ', Bar('#'),' ', Timer(), ' ', ETA(), ' ']
    pbar = ProgressBar(widgets=widgets, maxval=kitti_val_nums).start()

    with open(args.gt_info_path, 'rb') as fi:
        for i in range(kitti_val_nums):
            infos = pickle.load(fi)
            name = infos['frame_id']
            pre_file = os.path.join(args.pre_path, name + pre_tag)
            box_pre, score_pre, cls_pre = load_preds(pre_file)
            pred_dicts = [{'pred_scores':score_pre, 'pred_boxes':box_pre, 'pred_labels':cls_pre}]
            annos = generate_prediction_dicts(infos, pred_dicts, class_names)
            det_annos += annos
            pbar.update(i + 1)
        pbar.finish()
    ap_result_str, ap_dict = eval.get_official_eval_result(eval_gt_annos, det_annos, class_names)
    print('*************** kitti eval results ***************')
    print(ap_result_str)
    print('*************** kitti eval results ***************')

if __name__ == '__main__':
    # python eval_kitti.py --pre_path ./output
    main()