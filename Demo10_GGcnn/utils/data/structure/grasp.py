# -*- coding: UTF-8 -*-
"""===============================================
@Author : wangdx
@Date   : 2020/9/1 21:37
==============================================="""

import numpy as np
import cv2
import math
import scipy.io as scio
from utils.dataset_processing import mmcv


GRASP_WIDTH_MAX = 200.0


class GraspMat:
    def __init__(self, file):
        self.grasp = scio.loadmat(file)['A']   # (3, h, w)

    def height(self):
        return self.grasp.shape[1]

    def width(self):
        return self.grasp.shape[2]

    def crop(self, bbox):
        """
        裁剪 self.grasp

        args:
            bbox: list(x1, y1, x2, y2)
        """
        self.grasp = self.grasp[:, bbox[1]:bbox[3], bbox[0]:bbox[2]]

    def rescale(self, scale, interpolation='nearest'):
        """
        缩放
        """
        ori_shape = self.grasp.shape[1]
        self.grasp = np.stack([
            mmcv.imrescale(grasp, scale, interpolation=interpolation)
            for grasp in self.grasp
        ])
        new_shape = self.grasp.shape[1]
        ratio = new_shape / ori_shape
        # 抓取宽度同时缩放
        self.grasp[2, :, :] = self.grasp[2, :, :] * ratio

    def rotate(self, rota):
        """
        顺时针旋转
        rota: 角度
        """
        self.grasp = np.stack([mmcv.imrotate(grasp, rota) for grasp in self.grasp])
        # 角度旋转
        rota = rota / 180. * np.pi
        self.grasp[1, :, :] -= rota
        self.grasp[1, :, :] = self.grasp[1, :, :] % (np.pi * 2)
        self.grasp[1, :, :] *= self.grasp[0, :, :]

    def _flipAngle(self, angle_mat, confidence_mat):
        """
        水平翻转angle
        Args:
            angle_mat: (h, w) 弧度
            confidence_mat: (h, w) 抓取置信度
        Returns:
        """
        # 全部水平翻转
        angle_out = (angle_mat // math.pi) * 2 * math.pi + math.pi - angle_mat
        # 将非抓取区域的抓取角置0
        angle_out = angle_out * confidence_mat
        # 所有角度对2π求余
        angle_out = angle_out % (2 * math.pi)

        return angle_out

    def flip(self, flip_direction='horizontal'):
        """
        水平翻转
        """
        assert flip_direction in ('horizontal', 'vertical')

        self.grasp = np.stack([
            mmcv.imflip(grasp, direction=flip_direction)
            for grasp in self.grasp
        ])
        # 抓取角翻转，除了位置翻转，角度值也需要翻转
        self.grasp[1, :, :] = self._flipAngle(self.grasp[1, :, :], self.grasp[0, :, :])

    def encode(self):
        """
        (4, H, W) -> (angle_cls+2, H, W)
        """
        self.grasp[1, :, :] = (self.grasp[1, :, :] + 2 * math.pi) % math.pi
        
        self.grasp_point = self.grasp[0, :, :]
        self.grasp_cos = np.cos(self.grasp[1, :, :] * 2)
        self.grasp_sin = np.sin(self.grasp[1, :, :] * 2)
        self.grasp_width = self.grasp[2, :, :]
    
