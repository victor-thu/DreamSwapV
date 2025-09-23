# Openpose
# Original from CMU https://github.com/CMU-Perceptual-Computing-Lab/openpose
# 2nd Edited by https://github.com/Hzzone/pytorch-openpose
# 3rd Edited by ControlNet
# 4th Edited by ControlNet (added face and correct hands)

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

import torch
import numpy as np
from . import util
from .wholebody import Wholebody
import PIL

def draw_pose(pose, H, W, draw_face=False, draw_foot=False, draw_hand=True):
    bodies = pose['bodies']
    faces = pose['faces']
    hands = pose['hands']
    candidate = bodies['candidate']
    subset = bodies['subset']
    canvas = np.zeros(shape=(H, W, 3), dtype=np.uint8)

    canvas = util.draw_bodypose(canvas, candidate, subset)

    if draw_hand:
        if isinstance(hands, dict):
            hands = hands['candidate']
        canvas = util.draw_handpose(canvas, hands)

    if draw_face:
        canvas = util.draw_facepose(canvas, faces)

    # 这边只取foot的一个关节
    if draw_foot:
        foot = pose['foot']['candidate']
        foot = np.concatenate((candidate[10:11,:],foot[0,3:4,:],candidate[13:14,:],foot[0,0:1,:]), axis=0)
        num_person = pose['foot']['subset'].shape[0]
        foot_subset = np.tile([0,1,2,3], (num_person, 1))
        foot_subset[pose['foot']['subset'][:,:4]<0.5] = -1
        canvas = util.draw_footpose(canvas, foot, foot_subset)

    return canvas


class DWposeDetector:
    def __init__(self, cuda_idx=0):

        self.pose_estimation = Wholebody(cuda_idx=cuda_idx)

    def __call__(self, oriImg, return_dict=False):
        if isinstance(oriImg, PIL.Image.Image):
            oriImg = np.array(oriImg)
        oriImg = oriImg.copy()
        H, W, C = oriImg.shape
        with torch.no_grad():
            candidate, subset, has_handbag = self.pose_estimation(oriImg)
            nums, keys, locs = candidate.shape
            candidate[..., 0] /= float(W)
            candidate[..., 1] /= float(H)
            body = candidate[:,:18].copy()
            body = body.reshape(nums*18, locs)
            score = subset[:,:18]
            body_score = np.copy(subset[:,:18])
            for i in range(len(score)):
                for j in range(len(score[i])):
                    if score[i][j] > 0.3:
                        score[i][j] = int(18*i+j)
                    else:
                        score[i][j] = -1

            un_visible = subset<0.3
            candidate[un_visible] = -1

            foot = candidate[:,18:24]
            foot_score = subset[:,18:24]

            faces = candidate[:,24:92]

            hands = candidate[:,92:113]
            hands_score = subset[:, 92:113]
            hands = np.vstack([hands, candidate[:,113:]])
            hands_score = np.vstack([hands_score, subset[:,113:]])
            
            # hands[hands_score < 0.50] = -1
            
            bodies = dict(candidate=body, subset=score, score=body_score)
            hands = dict(candidate=hands, subset=hands_score)
            foot = dict(candidate=foot, subset=foot_score)
            pose = dict(bodies=bodies, hands=hands, faces=faces, foot=foot, has_handbag=has_handbag)
            if return_dict:
                return pose
            return draw_pose(pose, H, W)
