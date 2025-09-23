import os
import cv2
import math
import torch
import random
import skimage
import numpy as np
from glob import glob
from copy import deepcopy
from PIL import Image, ImageDraw, ImageFont, ImageOps
from moviepy.editor import concatenate_videoclips, ImageSequenceClip
from scipy.ndimage import gaussian_filter
from collections import OrderedDict, deque
from skimage import exposure

from .dwpose import draw_pose
from models.attention import BasicTransformerBlock

def modify_and_load_state_dict(model, state_dict):
    # 加载预训练权重
    pretrained_dict = state_dict['state_dict'] if 'state_dict' in state_dict else state_dict
    model_dict = model.state_dict()

    # 过滤出在当前模型中并且形状匹配的层
    matched_parameters = {k: v for k, v in pretrained_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
    
    # 打印出不匹配的层的信息
    unmatched_parameters = {k: v for k, v in pretrained_dict.items() if k not in matched_parameters}
    for k, v in unmatched_parameters.items():
        if k == 'img_in.proj.weight' or k == 'x_embedder.proj.weight' or k == 'patch_embedding.weight':
            c_out, c_in, f, h, w = v.shape
            new_c_in = model_dict[k].shape[1]
            matched_parameters[k] = torch.cat([v, torch.zeros(c_out, new_c_in - c_in, f, h, w)], dim=1)
            print(f"Skipping loading parameter '{k}' due to size mismatch (model: {model_dict[k].size()} vs pretrained: {v.size()})")
        elif k in model_dict:
            print(f"Skipping loading parameter '{k}' due to size mismatch (model: {model_dict[k].size()} vs pretrained: {v.size()})")
        else:
            print(f"Skipping loading parameter '{k}' which is not in the current model")

    # 更新当前模型的状态字典，只加载匹配的层
    model_dict.update(matched_parameters)
    model.load_state_dict(model_dict, strict=True)

    return model

def load_state_dict_safetensors(model, checkpoint_path):
    """
    加载模型权重但跳过那些无法加载的层。

    参数:
    model: 要加载权重的模型实例。
    checkpoint_path: 包含预训练权重的检查点路径。
    """

    state_dict = {}
    checkpoint_path = str(checkpoint_path)
    if os.path.isdir(checkpoint_path):
        for file in glob(checkpoint_path+'/*.safetensors'):
            state_dict.update(load_safetensors(file))
    elif checkpoint_path.endswith(".safetensors"):
        state_dict.update(load_safetensors(checkpoint_path))

    model = modify_and_load_state_dict(model, state_dict) # modified by fxr, strict=False for pose injection
    return model

def load_state_dict(args, model, pretrained_model_path):
    load_key = args.load_key
    pretrained_model_name_or_path = Path(args.pretrained_model_name_or_path)

    if pretrained_model_name_or_path is None:
        model_dir = pretrained_model_path / f"t2v_{args.model_resolution}"
        files = list(model_dir.glob("*.pt"))
        if len(files) == 0:
            raise ValueError(f"No model weights found in {model_dir}")
        if str(files[0]).startswith("pytorch_model_"):
            model_path = pretrained_model_name_or_path / f"pytorch_model_{load_key}.pt"
            bare_model = True
        elif any(str(f).endswith("_model_states.pt") for f in files):
            files = [f for f in files if str(f).endswith("_model_states.pt")]
            model_path = files[0]
            if len(files) > 1:
                logger.warning(
                    f"Multiple model weights found in {pretrained_model_name_or_path}, using {model_path}"
                )
            bare_model = False
        else:
            raise ValueError(
                f"Invalid model path: {pretrained_model_name_or_path} with unrecognized weight format: "
                f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                f"specific weight file, please provide the full path to the file."
            )
    else:
        if pretrained_model_name_or_path.is_dir():
            files = list(pretrained_model_name_or_path.glob("*.pt"))
            if len(files) == 0:
                raise ValueError(f"No model weights found in {pretrained_model_name_or_path}")
            if str(files[0]).startswith("pytorch_model_"):
                model_path = pretrained_model_name_or_path / f"pytorch_model_{load_key}.pt"
                bare_model = True
            elif any(str(f).endswith("_model_states.pt") for f in files):
                files = [f for f in files if str(f).endswith("_model_states.pt")]
                model_path = files[0]
                if len(files) > 1:
                    logger.warning(
                        f"Multiple model weights found in {pretrained_model_name_or_path}, using {model_path}"
                    )
                bare_model = False
            else:
                raise ValueError(
                    f"Invalid model path: {pretrained_model_name_or_path} with unrecognized weight format: "
                    f"{list(map(str, files))}. When given a directory as --dit-weight, only "
                    f"`pytorch_model_*.pt`(provided by HunyuanDiT official) and "
                    f"`*_model_states.pt`(saved by deepspeed) can be parsed. If you want to load a "
                    f"specific weight file, please provide the full path to the file."
                )
        elif pretrained_model_name_or_path.is_file():
            model_path = pretrained_model_name_or_path
            bare_model = "unknown"
        else:
            raise ValueError(f"Invalid model path: {pretrained_model_name_or_path}")

    if not model_path.exists():
        raise ValueError(f"model_path not exists: {model_path}")
    logger.info(f"Loading torch model {model_path}...")
    state_dict = torch.load(model_path, map_location=lambda storage, loc: storage)

    if bare_model == "unknown" and ("ema" in state_dict or "module" in state_dict):
        bare_model = False
    if bare_model is False:
        if load_key in state_dict:
            state_dict = state_dict[load_key]
        else:
            raise KeyError(
                f"Missing key: `{load_key}` in the checkpoint: {model_path}. The keys in the checkpoint "
                f"are: {list(state_dict.keys())}."
            )

    model = modify_and_load_state_dict(model, state_dict) # modified by fxr, strict=False for pose injection
    return model

def f(r, T=0.6, beta=0.1):
    return np.where(r < T, beta + (1 - beta) / T * r, 1)

# Get the bounding box of the mask
def get_bbox_from_mask(mask):
    h,w = mask.shape[0],mask.shape[1]

    if mask.sum() < 10:
        return 0,h,0,w
    rows = np.any(mask,axis=1)
    cols = np.any(mask,axis=0)
    y1,y2 = np.where(rows)[0][[0,-1]]
    x1,x2 = np.where(cols)[0][[0,-1]]
    return (y1,y2,x1,x2)

# Expand the bounding box
def expand_bbox(mask, yyxx, ratio, min_crop=0):
    y1,y2,x1,x2 = yyxx
    H,W = mask.shape[0], mask.shape[1]

    yyxx_area = (y2-y1+1) * (x2-x1+1)
    r1 = yyxx_area / (H * W)
    r2 = f(r1)
    ratio = math.sqrt(r2 / r1)

    xc, yc = 0.5 * (x1 + x2), 0.5 * (y1 + y2)
    h = ratio * (y2-y1+1)
    w = ratio * (x2-x1+1)
    h = max(h,min_crop)
    w = max(w,min_crop)

    x1 = int(xc - w * 0.5)
    x2 = int(xc + w * 0.5)
    y1 = int(yc - h * 0.5)
    y2 = int(yc + h * 0.5)

    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Pad the image to a square shape
def pad_to_square(image, pad_value = 255, random = False):
    H,W = image.shape[0], image.shape[1]
    if H == W:
        return image

    padd = abs(H - W)
    if random:
        padd_1 = int(np.random.randint(0,padd))
    else:
        padd_1 = int(padd / 2)
    padd_2 = padd - padd_1

    if len(image.shape) == 2: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2))
        else:
            pad_param = ((padd_1, padd_2), (0, 0))
    elif len(image.shape) == 3: 
        if H > W:
            pad_param = ((0, 0), (padd_1, padd_2), (0, 0))
        else:
            pad_param = ((padd_1, padd_2), (0, 0), (0, 0))

    image = np.pad(image, pad_param, 'constant', constant_values=pad_value)

    return image

# Expand the image and mask
def expand_image_mask(image, mask, ratio=1.4):
    h,w = image.shape[0], image.shape[1]
    H,W = int(h * ratio), int(w * ratio) 
    h1 = int((H - h) // 2)
    h2 = H - h - h1
    w1 = int((W -w) // 2)
    w2 = W -w - w1

    pad_param_image = ((h1,h2),(w1,w2),(0,0))
    pad_param_mask = ((h1,h2),(w1,w2))
    image = np.pad(image, pad_param_image, 'constant', constant_values=255)
    mask = np.pad(mask, pad_param_mask, 'constant', constant_values=0)
    return image, mask

# Convert the bounding box to a square shape
def box2squre(image, box):
    H,W = image.shape[0], image.shape[1]
    y1,y2,x1,x2 = box
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2
    h,w = y2-y1, x2-x1

    if h >= w:
        x1 = cx - h//2
        x2 = cx + h//2
    else:
        y1 = cy - w//2
        y2 = cy + w//2
    x1 = max(0,x1)
    x2 = min(W,x2)
    y1 = max(0,y1)
    y2 = min(H,y2)
    return (y1,y2,x1,x2)

# Crop the predicted image back to the original image
def crop_back( pred, tar_image,  extra_sizes, tar_box_yyxx_crop):
    H1, W1, H2, W2 = extra_sizes
    y1,y2,x1,x2 = tar_box_yyxx_crop    
    pred = cv2.resize(pred, (W2, H2))
    m = 2 # maigin_pixel

    if W1 == H1:
        if m != 0:
            tar_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
        else:
            tar_image[y1 :y2, x1:x2, :] =  pred[:, :]
        return tar_image

    if W1 < W2:
        pad1 = int((W2 - W1) / 2)
        pad2 = W2 - W1 - pad1
        pred = pred[:,pad1: -pad2, :]
    else:
        pad1 = int((H2 - H1) / 2)
        pad2 = H2 - H1 - pad1
        pred = pred[pad1: -pad2, :, :]

    gen_image = tar_image.copy()
    if m != 0:
        gen_image[y1+m :y2-m, x1+m:x2-m, :] =  pred[m:-m, m:-m]
    else:
        gen_image[y1 :y2, x1:x2, :] =  pred[:, :]
    
    return gen_image

def torch_dfs(model: torch.nn.Module, target_module=BasicTransformerBlock):
    result = OrderedDict()
    for name, module in model.named_modules():
        if isinstance(module, target_module):
            result[name] = module
    return result


class BodyTreeNode:
    def __init__(self, val, children, parent):
        self.val = val
        self.children = children
        self.parent = parent


def call_similarity(pose1, pose2, flip_call=True):
    """
    flip_call: 计算相似度flip后取更大的
    """
    if len(pose1) == 0 or len(pose2) == 0:
        return np.inf

    pose1 -= np.mean(pose1, axis=0)
    pose2 -= np.mean(pose2, axis=0)

    # 归一化尺度
    pose1 /= np.max(np.sqrt(np.sum(pose1**2, axis = 1)))
    pose2 /= np.max(np.sqrt(np.sum(pose2**2, axis = 1)))

    # 计算欧氏距离
    dist = np.sqrt(np.sum((pose1 - pose2)**2))

    dist_flip = np.inf
    if flip_call:
        pose2[:,0] = -pose2[:,0]
        dist_flip = np.sqrt(np.sum((pose1 - pose2)**2))

    # 计算相似度，距离越小，相似度越大
    return max(-dist, -dist_flip)

def face_crop(image, pose, ratio=1/2):
    """
    image: 需要裁剪的图片，可以是list
    pose: 裁剪的基准pose点位
    ratio: 脖子到面部的比例，越接近0越接近脖子，即越往下
    """
    # 支持视频帧crop，但是只根据输入的一个pose来crop，pose的选择由外部逻辑控制
    if isinstance(image, list):
        image_list = []
        for i in image:
            image_list.append(face_crop(i, pose))
        return image_list

    # 图片crop
    w, h = image.size
    body = pose['bodies']['candidate']
    subset = pose['bodies']['subset']
    # 取01关键点的中间位置
    p0 = body[0,:]
    p1 = body[1,:]
    if subset[0,0] == -1 or p0[1] < 0:
        return image
    else:
        # 裁脖子和面部的4 6 开的点
        x, y = ratio*p0 + (1-ratio)*p1
        x, y = int(w * x), int(h * y)

        # 计算目标宽高
        target_h = h - y
        target_w = int(w * target_h / h)

        # 计算裁剪位置
        left = max(x - target_w // 2, 0)
        right = target_w + left
        if right > w:
            right = w
            left = w - target_w
        top = y
        bottom = target_h + y

        image_cropped = image.crop((left, top, right, bottom))
        return image_cropped

def pose_crop(reference_pose, target_pose_list, height, width):
    """
    Modified from mimicmotion
    仅对Pose做裁减缩放对齐到reference image的位置
    reference_pose:DWPose Dict格式结果
    target_pose_list:[DWPose Dict格式结果]
    height: 目标图片高度
    widht: 目标图片宽度
    """
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(reference_pose['bodies']['subset']) > 0 and reference_pose['bodies']['subset'][0][i] >= .0]
    ref_body = reference_pose['bodies']['candidate'][ref_keypoint_id]

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in target_pose_list if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]

    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw =  height, width
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale
    for detected_pose in target_pose_list:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        if isinstance(detected_pose['hands'], dict):
            detected_pose['hands']['candidate'] = detected_pose['hands']['candidate'] * a + b
        else:
            detected_pose['hands'] = detected_pose['hands'] * a + b
        if 'foot' in detected_pose.keys():
            detected_pose['foot']['candidate'] = detected_pose['foot']['candidate'] * a + b
        output_pose.append(detected_pose)

    return output_pose

def pose_crop_v1(reference_pose, target_pose_list, height, width):
    """
    v1 版本额外返回平移和缩放
    Modified from mimicmotion
    仅对Pose做裁减缩放对齐到reference image的位置
    reference_pose:DWPose Dict格式结果
    target_pose_list:[DWPose Dict格式结果]
    height: 目标图片高度
    widht: 目标图片宽度
    """
    ref_keypoint_id = [0, 1, 2, 5, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17]
    ref_keypoint_id = [i for i in ref_keypoint_id \
        if len(reference_pose['bodies']['subset']) > 0 and reference_pose['bodies']['subset'][0][i] >= .0]
    ref_body = reference_pose['bodies']['candidate'][ref_keypoint_id]

    detected_bodies = np.stack(
        [p['bodies']['candidate'] for p in target_pose_list if p['bodies']['candidate'].shape[0] == 18])[:,
                      ref_keypoint_id]

    # compute linear-rescale params
    ay, by = np.polyfit(detected_bodies[:, :, 1].flatten(), np.tile(ref_body[:, 1], len(detected_bodies)), 1)
    fh, fw =  height, width
    ax = ay / (fh / fw / height * width)
    bx = np.mean(np.tile(ref_body[:, 0], len(detected_bodies)) - detected_bodies[:, :, 0].flatten() * ax)
    a = np.array([ax, ay])
    b = np.array([bx, by])
    output_pose = []
    # pose rescale 
    for detected_pose in target_pose_list:
        detected_pose['bodies']['candidate'] = detected_pose['bodies']['candidate'] * a + b
        detected_pose['faces'] = detected_pose['faces'] * a + b
        if isinstance(detected_pose['hands'], dict):
            detected_pose['hands']['candidate'] = detected_pose['hands']['candidate'] * a + b
        else:
            detected_pose['hands'] = detected_pose['hands'] * a + b
        if 'foot' in detected_pose.keys():
            detected_pose['foot']['candidate'] = detected_pose['foot']['candidate'] * a + b
        output_pose.append(detected_pose)

    return output_pose, a, b

def scale_and_transform_image(image, scale=(1.0, 1.0), transform=(0, 0)):
    """
    对图像进行缩放和平移操作，分辨率不变。

    参数:
        image (PIL.Image): 输入图像
        scale (tuple): 缩放比例（scale_x, scale_y）
        transform (tuple): 平移量（dx, dy）

    返回:
        PIL.Image: 处理后的图像
    """
    assert len(scale) == 2, "scale参数应为(scale_x, scale_y)形式的元组"
    assert scale[0] > 0 and scale[1] > 0, "scale必须大于0"
    assert len(transform) == 2, "transform参数应为(dx, dy)形式的元组"

    # 获取输入图像的尺寸
    width, height = image.size

    # 缩放
    new_width = int(scale[0] * width)
    new_height = int(scale[1] * height)
    scaled_image = image.resize((new_width, new_height), Image.LANCZOS)

    # 创建新的空白图像
    new_image = Image.new("RGB", (width, height))

    # 计算平移后的坐标
    dx, dy = int(transform[0] * width), int(transform[1] * height)

    # 将缩放后的图像粘贴到新图像中
    new_image.paste(scaled_image, (dx, dy))

    return new_image

def pose_alignment(reference_pose, target_pose_list, return_img=False, height=-1, width=-1, align_position=False, align_index=-1, align_set='shoulder_leg', draw_face=False, draw_foot=False):
    """
    reference_pose:DWPose Dict格式结果
    target_pose_list:[DWPose Dict格式结果]
    """
    assert (return_img and height > 0 and width > 0) or not return_img, '未提供hw时无法绘制'
    assert align_set in ('full', 'wo_head', 'shoulder_only', 'shoulder_leg')
    if align_set == 'full':
        exclude_set = []
    elif align_set == 'wo_head':
        exclude_set = [(0,14),(0,15),(14,16),(15,17)]
    elif align_set == 'shoulder_only':
        exclude_set = [(0,1),(0,14),(0,15),(14,16),(15,17),(2,3),(3,4),(5,6),(6,7),(1,8),(1,11),(8,9),(11,12),(9,10),(12,13)]
    elif align_set == 'shoulder_leg':
        exclude_set = [(0,14),(0,15),(14,16),(15,17),(2,3),(3,4),(5,6),(6,7)]
    else:
        assert False, f'{align_set} not supported'

    ref_body = reference_pose['bodies']['candidate']
    ref_subset = reference_pose['bodies']['subset'][0]
    ref_hands = reference_pose['hands']
    if isinstance(ref_hands, dict):
        ref_hands = ref_hands['candidate']
    eps = 1e-8

    # 关节链接关系，DWPose代码定义的，有向且无重复的图
    # 身体的索引从1开始，取数时需要-1
    bodyLimbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18]]
    # 手部的索引从0开始，正常取用
    handLimbSeq = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]
    
    # 肢体构建为树结构方便重建人体
    # bfs建树，2号脖子节点作为树根
    body_tree = BodyTreeNode(2,[],None)
    queue = deque([body_tree])
    while queue:
        for i, j in bodyLimbSeq:
            if i == queue[0].val:
                new_node = BodyTreeNode(j,[],queue[0])
                queue[0].children.append(new_node)
                queue.append(new_node)
        queue.popleft()
    # bfs建树，0号手掌根部节点作为树根
    hand_tree = BodyTreeNode(0,[],None)
    queue = deque([hand_tree])
    while queue:
        for i, j in handLimbSeq:
            if i == queue[0].val:
                new_node = BodyTreeNode(j,[],queue[0])
                queue[0].children.append(new_node)
                queue.append(new_node)
        queue.popleft()

    # 对齐人体结构，reference pose提供肢体长度和整体位置（即根节点位置），target pose提供肢体朝向
    aligned_target_pose_list = []
    aligned_target_posesubset_list = []
    aligned_target_hand_list = []
    aligned_target_foot_list = []
    body_limb_ratio = {}
    hand_limb_ratio = {}

    # 找到参考帧中的对齐姿势
    # debug_sim = []
    # ref_img = draw_pose(reference_pose, H=768, W=576)
    if align_index < 0:
        print(f'未指定对齐的帧，自动检测最适合的帧')
        max_similarity = -1
        max_point_cnt = 0
        for idx, target_pose in enumerate(target_pose_list):
            target_body = target_pose['bodies']['candidate']
            target_subset = target_pose['bodies']['subset'][0]
            target_point_cnt = (target_subset > -1).sum()
            # 计算相似度时仅考虑身体不考虑头部
            body_mask = np.array([False]+[True]*13+[False]*4)
            inter_sub = (target_subset > -1) & (ref_subset > -1) & body_mask
            target_similarity = call_similarity(target_body[inter_sub], ref_body[inter_sub], flip_call=True)

            # debug
            # tmp_img = draw_pose(target_pose, H=768, W=576)
            # tmp_img = np.concatenate((ref_img, tmp_img), axis=1)
            # font = cv2.FONT_HERSHEY_SIMPLEX
            # cv2.putText(tmp_img, f'{target_similarity}', (100, 100), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
            # debug_sim.append(tmp_img)
            
            if target_point_cnt > max_point_cnt:
                max_similarity = target_similarity
                max_point_cnt = target_point_cnt
                align_index = idx
            if target_point_cnt >= max_point_cnt and target_similarity > max_similarity:
                max_similarity = target_similarity
                align_index = idx
        

        # cv2.imwrite(f'{idx}.jpg', np.concatenate(debug_sim, axis=0))
    
    # if (ref_subset > -1).sum() != (target_pose_list[align_index]['bodies']['subset'][0] > -1).sum():
    #     print(f"""Warning: 参考帧中仅发现了 {(ref_subset > -1).sum()} 个点位， 
    #             而视频中出现了最多 {(target_pose_list[align_index]['bodies']['subset'][0] > -1).sum()} 个点位，放弃对齐""")
    #     return target_pose_list
    
    align_body = target_pose_list[align_index]['bodies']['candidate']
    size_ratio = (ref_body[:,1].max() - ref_body[:,1].min()) / (align_body[:,1].max() - align_body[:,1].min())

    for idx, target_pose in enumerate(target_pose_list[align_index:align_index+1]+target_pose_list):
        target_body = target_pose['bodies']['candidate']
        target_subset = target_pose['bodies']['subset'][0]
        target_hands = target_pose['hands']
        if isinstance(target_hands, dict):
            target_hands = target_hands['candidate']
        target_foot = target_pose['foot']['candidate']

        aligned_target_body = np.zeros(target_body.shape)
        aligned_target_subset = np.zeros(target_subset.shape)
        aligned_target_hands = np.zeros(target_hands.shape)
        aligned_target_foot = np.zeros(target_foot.shape)

        # bfs处理全部肢体，身体部分的索引需要-1
        node = body_tree
        # 从根节点开始构建新的人体姿态，先把根节点的位置对齐到reference的相同关节
        if align_position:
            aligned_target_body[node.val-1] = ref_body[node.val-1] + \
                                (target_body[node.val-1] - target_pose_list[0]['bodies']['candidate'][node.val-1])
        else:
            aligned_target_body[node.val-1] = target_body[node.val-1]
        queue = deque([node])

        while queue:
            mother_index = queue[0].val - 1
            for child in queue[0].children:
                child_index = child.val - 1
                queue.append(child)

                # 根据subset判断是否有未出的点，身体部分的关节并不是一定能检出的
                target_index = target_subset[np.array([mother_index, child_index])].astype(np.int32)
                ref_index = ref_subset[np.array([mother_index, child_index])].astype(np.int32)

                aligned_target_subset[mother_index] = ref_index[0] if target_index[0] > -1 else target_index[0]
                aligned_target_subset[child_index] = ref_index[1] if target_index[1] > -1 else target_index[1]

                if -1 in ref_index or -1 in target_index:
                    if target_index[0] == -1:
                        if align_position:
                            aligned_target_body[child_index] = ref_body[child_index] + (target_body[child_index] - target_pose_list[0]['bodies']['candidate'][child_index])
                        else:
                            aligned_target_body[child_index] = target_body[child_index]
                    continue

                # 根据向量的模和方向重新定义新的子节点
                ref_vec = ref_body[ref_index][1] - ref_body[ref_index][0]
                target_vec = target_body[target_index][1] - target_body[target_index][0]
                
                if (mother_index, child_index) not in body_limb_ratio.keys() or idx==0:
                    ref_length = np.linalg.norm(ref_vec) + eps
                    target_length = np.linalg.norm(target_vec) + eps
                    ratio = ref_length / target_length * size_ratio
                    body_limb_ratio[(mother_index, child_index)] = ratio
                else:
                    # 跳过 exclude_set 内的关节
                    if (mother_index, child_index) not in exclude_set:
                        # 确保缩放系数对称
                        if (mother_index, child_index) == (14,16) or (mother_index, child_index) == (15,17):
                            ratio = (body_limb_ratio[(14,16)] + body_limb_ratio[(15,17)]) / 2
                            # ratio = max(body_limb_ratio[(14,16)], body_limb_ratio[(15,17)])
                            
                        elif (mother_index, child_index) == (0,14) or (mother_index, child_index) == (0,15):
                            ratio = (body_limb_ratio[(0,14)] + body_limb_ratio[(0,15)]) / 2
                            # ratio = max(body_limb_ratio[(0,14)], body_limb_ratio[(0,15)])
                            
                        elif (mother_index, child_index) == (1,2) or (mother_index, child_index) == (1,5):
                            ratio = (body_limb_ratio[(1,2)] + body_limb_ratio[(1,5)]) / 2
                            # ratio = max(body_limb_ratio[(1,2)], body_limb_ratio[(1,5)])
                            
                        elif (mother_index, child_index) == (2,3) or (mother_index, child_index) == (5,6):
                            ratio = (body_limb_ratio[(2,3)] + body_limb_ratio[(5,6)]) / 2
                            # ratio = max(body_limb_ratio[(2,3)], body_limb_ratio[(5,6)])
                            
                        elif (mother_index, child_index) == (3,4) or (mother_index, child_index) == (6,7):
                            ratio = (body_limb_ratio[(3,4)] + body_limb_ratio[(6,7)]) / 2
                            # ratio = max(body_limb_ratio[(3,4)], body_limb_ratio[(6,7)])
                            
                        elif (mother_index, child_index) == (1,8) or (mother_index, child_index) == (1,11):
                            ratio = (body_limb_ratio[(1,8)] + body_limb_ratio[(1,11)]) / 2
                            # ratio = max(body_limb_ratio[(1,8)], body_limb_ratio[(1,11)])
                            
                        elif (mother_index, child_index) == (8,9) or (mother_index, child_index) == (11,12):
                            ratio = (body_limb_ratio[(8,9)] + body_limb_ratio[(11,12)]) / 2
                            # ratio = max(body_limb_ratio[(8,9)], body_limb_ratio[(11,12)])
                            
                        elif (mother_index, child_index) == (9,10) or (mother_index, child_index) == (12,13):
                            ratio = (body_limb_ratio[(9,10)] + body_limb_ratio[(12,13)]) / 2
                            # ratio = max(body_limb_ratio[(9,10)], body_limb_ratio[(12,13)])
                            
                        elif (mother_index, child_index) == (1,0):
                            ratio = body_limb_ratio[(1,0)]
                        else:
                            assert False, (mother_index, child_index)
                    else:
                        ratio = size_ratio

                # if (mother_index, child_index) not in body_limb_ratio.keys() and idx==0:
                #     ref_length = np.linalg.norm(ref_vec) + eps
                #     target_length = np.linalg.norm(target_vec) + eps
                #     ratio = ref_length / target_length
                #     body_limb_ratio[(mother_index, child_index)] = ratio
                # else:
                #     ratio = sum(body_limb_ratio.values())/len(body_limb_ratio)

                resized_target_vec = target_vec * ratio
                aligned_target_body[child_index] = aligned_target_body[mother_index] + resized_target_vec

            queue.popleft()
            
        aligned_target_pose_list.append(aligned_target_body)
        aligned_target_posesubset_list.append(aligned_target_subset[None,...])

        # 肢体的平均缩放系数
        body_avg_ratio = sum(body_limb_ratio.values())/len(body_limb_ratio)

        # bfs处理全部手掌
        node = hand_tree
        # 根节点初始化，掌根节点需要和重定向后的手腕关节对齐，否则会错位，5 和 8 为手腕关节
        right_wrist_index, left_wrist_index = int(target_subset[5-1]), int(target_subset[8-1])
        aligned_target_hands[:,node.val] = target_hands[:,node.val]

        if left_wrist_index != -1 and int(ref_subset[left_wrist_index]) != -1:
            aligned_target_hands[0,node.val] = aligned_target_body[left_wrist_index]
            aligned_target_hands[0,node.val] = aligned_target_body[left_wrist_index] + \
                                        target_hands[0,node.val] - target_body[left_wrist_index]

        if right_wrist_index != -1 and int(ref_subset[right_wrist_index]) != -1:
            aligned_target_hands[1,node.val] = aligned_target_body[right_wrist_index] + \
                                        target_hands[1,node.val] - target_body[right_wrist_index]

        queue = deque([node])
        while queue:
            mother_index = queue[0].val
            for child in queue[0].children:
                child_index = child.val
                queue.append(child)

                point_index = np.array([mother_index, child_index])

                # 根据向量的模和方向重新定义新的子节点
                ref_vec = ref_hands[:,point_index][:,1] - ref_hands[:,point_index][:,0]
                target_vec = target_hands[:,point_index][:,1] - target_hands[:,point_index][:,0]
                if (mother_index, child_index) not in hand_limb_ratio.keys():

                    # 手部的ratio并不可靠，基于肢体估算ratio
                    ratio = body_avg_ratio
                    hand_limb_ratio[(mother_index, child_index)] = ratio
                else:
                    ratio = hand_limb_ratio[(mother_index, child_index)]
                resized_target_vec = target_vec * ratio
                aligned_target_hands[:,child_index] = aligned_target_hands[:,mother_index] + resized_target_vec
                
                # 如果母节点没有位置，子节点也置为-1，这边可能和原Pose并不一致，掌根节点未检出时直接放弃整个手掌
                if -1 in target_hands[:,point_index][0,0,:]:
                    aligned_target_hands[0,child_index,:] = -1
                if -1 in target_hands[:,point_index][1,0,:]:
                    aligned_target_hands[1,child_index,:] = -1

            queue.popleft()
        aligned_target_hand_list.append(aligned_target_hands)

        # 对齐脚掌点位
        aligned_target_foot[0,0:1,:] = aligned_target_body[13,:] + (target_foot[0,0:1,:] - target_body[13,:]) * body_avg_ratio
        aligned_target_foot[0,3:4,:] = aligned_target_body[10,:] + (target_foot[0,3:4,:] - target_body[10,:]) * body_avg_ratio
        aligned_target_foot_list.append(aligned_target_foot)
    
    aligned_target_pose_list.pop(0)
    aligned_target_posesubset_list.pop(0)
    aligned_target_hand_list.pop(0)
    aligned_target_foot_list.pop(0)

    for target_pose, aligned_target_pose, target_posesubset, aligned_target_hand, aligned_target_foot in \
        zip(target_pose_list, aligned_target_pose_list, aligned_target_posesubset_list, aligned_target_hand_list, aligned_target_foot_list):

        target_pose['bodies']['candidate'] = aligned_target_pose
        target_pose['bodies']['subset'] = target_posesubset
        target_pose['foot']['candidate'] = aligned_target_foot
        if isinstance(target_pose['hands'], dict):
            target_pose['hands']['candidate'] = aligned_target_hand
        else:
            target_pose['hands'] = aligned_target_hand

    if not return_img:
        return target_pose_list
    return [Image.fromarray(draw_pose(i, height, width, draw_face, draw_foot)) for i in target_pose_list]

# def frames_to_video(frames, output_path, fps=15):
#     images_array = [np.array(img) for img in frames]
#     clip = ImageSequenceClip(images_array, fps=fps)
#     clip.write_videofile(output_path, codec='libx264')
#     return output_path

def frames_to_video(frames, output_path, fps=15):
    if not frames:
        raise ValueError("The frames list is empty.")
        
    # 获取帧的尺寸 (height, width, channels)
    frames = [np.array(i) for i in frames]
    height, width, _ = frames[0].shape

    # 定义视频编码器和输出视频参数
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') # 使用适合 QuickTime 的编码器
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # 将帧写入输出视频
    for frame in frames:
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    
    # 释放视频写入器
    out.release()
    return output_path

def video_to_frames(video_path, fps=15):
    # 打开视频文件
    cap = cv2.VideoCapture(video_path)

    # 检查视频是否成功打开
    if not cap.isOpened():
        print("Error: could not open video file")
        return
    
    # 获取视频的帧率
    video_fps = round(cap.get(cv2.CAP_PROP_FPS), 0)
    frame_skip = math.ceil(video_fps / fps)  # 计算每秒需要跳过多少帧以达到15 fps

    # 逐帧读取视频，并将每帧图像保存到文件夹中
    frame_count = 0
    res = []
    while True:
        # 读取一帧
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res.append(frame)
        frame_count += 1

    # 关闭视频文件
    cap.release()
    return res

def pose_split(pose_dict_list, threshold=0.2, sigma=0.5):
    pose_dict_list = deepcopy(pose_dict_list)
    bodies_seq = np.stack([i['bodies']['candidate'] for i in pose_dict_list], axis=0) # n,18,2
    bodies_subset_seq = np.stack([i['bodies']['subset'] for i in pose_dict_list], axis=0)
    split_points = []
    for i in range(1, bodies_seq.shape[0]):
        pre = bodies_seq[i-1]
        cur = bodies_seq[i]
        index = ((bodies_subset_seq[i-1]>0) & (bodies_subset_seq[i]>0))
        index = index[0,:,None].repeat(2,axis=1)
        
        diff = np.linalg.norm(pre[index]-cur[index],2)
        if diff>threshold:
            split_points.append(i)

    res = []
    prev = p = 0
    for p in split_points:
        if sigma > 0:
            pose_dict_list[prev:p] = pose_smooth(pose_dict_list[prev:p], sigma, -1, -1, return_img=False)
        res.append(pose_dict_list[prev:p])
        prev = p
    if sigma > 0:
        pose_dict_list[p:] = pose_smooth(pose_dict_list[p:], sigma, -1, -1, return_img=False)
    res.append(pose_dict_list[p:])
    
    return res

def pose_smooth(pose_dict_list, height=None, width=None, return_img=False):

    bodies_seq = np.stack([i['bodies']['candidate'] for i in pose_dict_list], axis=0) # n,18,2
    bodies_subset_seq = np.stack([i['bodies']['subset'] for i in pose_dict_list], axis=0) # n,1,18
    faces_seq = np.stack([i['faces'] for i in pose_dict_list], axis=0)
    foot_seq = np.stack([i['foot']['candidate'] for i in pose_dict_list], axis=0) # n,2,4,2
    if isinstance(pose_dict_list[0]['hands'], dict):
        hands_seq = np.stack([i['hands']['candidate'] for i in pose_dict_list], axis=0) # n,2,21,2
    else:
        hands_seq = np.stack([i['hands'] for i in pose_dict_list], axis=0) # n,2,21,2

    # smooth body
    n, points, xy = bodies_seq.shape
    for point in range(points):
        for idx in range(xy):
            mask = bodies_seq[:, point, idx] < 0
            bodies_seq[mask, point, idx] = np.nan
            bodies_seq[~mask, point, idx] = gaussian_filter(bodies_seq[~mask, point, idx], sigma=1, mode='nearest')
            bodies_seq[mask, point, idx] = -1

    # smooth hands
    n, hands, points, xy = hands_seq.shape
    for hand in range(hands):
        for point in range(points):
            for idx in range(xy):
                mask = hands_seq[:, hand, point, idx] < 0
                hands_seq[mask, hand, point, idx] = np.nan
                hands_seq[~mask, hand, point, idx] = gaussian_filter(hands_seq[~mask, hand, point, idx], sigma=1, mode='nearest')
                hands_seq[mask, hand, point, idx] = -1
    
    # smooth faces
    n, faces, points, xy = faces_seq.shape
    for face in range(faces):
        for point in range(points):
            for idx in range(xy):
                mask = faces_seq[:, face, point, idx] < 0
                faces_seq[mask, face, point, idx] = np.nan
                faces_seq[~mask, face, point, idx] = gaussian_filter(faces_seq[~mask, face, point, idx], sigma=2, mode='nearest')
                faces_seq[mask, face, point, idx] = -1
    
    # smooth foot
    n, foots, points, xy = foot_seq.shape
    for foot in range(foots):
        for point in range(points):
            for idx in range(xy):
                mask = foot_seq[:, foot, point, idx] < 0
                foot_seq[mask, foot, point, idx] = np.nan
                foot_seq[~mask, foot, point, idx] = gaussian_filter(foot_seq[~mask, foot, point, idx], sigma=1, mode='nearest')
                foot_seq[mask, foot, point, idx] = -1
    
    for idx, pose_dict in enumerate(pose_dict_list):
        pose_dict['bodies']['candidate'] = bodies_seq[idx]
        pose_dict['hands'] = hands_seq[idx]
        pose_dict['faces'] = faces_seq[idx]
        pose_dict['foot']['candidate'] = foot_seq[idx]
    
    if return_img:
        assert height is not None and width is not None, '绘制时必须指定 width height'
        return [Image.fromarray(draw_pose(i, height, width)) for i in pose_dict_list]

    return pose_dict_list

def posehand_threshold(pose_dict, hand_thresh=0.5):
    pose_dict = deepcopy(pose_dict)
    hand_dict = pose_dict['hands']
    hand_dict['candidate'][hand_dict['subset'] < hand_thresh] = -1
    return pose_dict

def add_text_to_image(image, text):
    if isinstance(image, list):
        return [add_text_to_image(i, text) for i in image]
    # 获取图像的尺寸和通道
    width, height = image.size
    channels = len(image.getbands())

    # 创建黑条和绘图对象
    black_bar = Image.new("RGB", (width, 50), (0, 0, 0))
    draw = ImageDraw.Draw(image)

    # 获取字体文件和字体大小
    font = ImageFont.truetype('utils/Arial.ttf', size=30)

    # 计算文本尺寸
    text_size = draw.textsize(text, font)

    # 在图像和黑条上绘制文本
    draw.text(((width - text_size[0]) / 2, height + 10), text, font=font, fill=(255, 255, 255, 255))
    black_bar_draw = ImageDraw.Draw(black_bar)
    black_bar_draw.text(((width - text_size[0]) / 2, 10), text, font=font, fill=(255, 255, 255, 255))

    # 如果图像的通道为1，则将黑条转换为灰度
    if channels == 1:
        black_bar = black_bar.convert("L")

    # 将黑条粘贴到图像底部
    totalimage = Image.new(image.mode, (width, height + 50))
    totalimage.paste(image, (0, 0))
    black_bar = black_bar.crop((0, 0, width, 50))
    totalimage.paste(black_bar, (0, height))

    return totalimage

def pil_to_cv2(pil_image):
    """Convert PIL image to OpenCV format."""
    cv2_image = np.array(pil_image)
    if pil_image.mode == "RGBA":
        cv2_image = cv2_image[:, :, :3]
    return cv2.cvtColor(cv2_image, cv2.COLOR_RGB2BGR)

def cv2_to_pil(cv2_image):
    """Convert OpenCV format image to PIL format."""
    cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(cv2_image)
    return pil_image

def smooth_transition(frame1, frame2, alpha):
    """Smooth transition between two frames based on alpha."""
    return cv2.addWeighted(frame1, 1 - alpha, frame2, alpha, 0)

def color_transfer_lab_ab_only(ref_pil_image, pil_video_frames):
    """
    Transfer color in LAB space (A and B channels only) from a reference image to a list of video frames.
    
    Args:
        ref_pil_image (PIL.Image.Image): Reference image in PIL format.
        pil_video_frames (list of PIL.Image.Image): List of video frames in PIL format.
    
    Returns:
        list of PIL.Image.Image: Processed video frames in PIL format.
    """
    ref_cv2 = pil_to_cv2(ref_pil_image)

    # Convert reference image to LAB color space
    ref_lab = cv2.cvtColor(ref_cv2, cv2.COLOR_BGR2LAB).astype(np.float32)
    l_ref, a_ref, b_ref = cv2.split(ref_lab)
    mean_a_ref, std_a_ref = cv2.meanStdDev(a_ref)
    mean_b_ref, std_b_ref = cv2.meanStdDev(b_ref)

    processed_frames_pil = []

    for frame in pil_video_frames:
        frame_cv2 = pil_to_cv2(frame)
        
        # Convert frame image to LAB color space
        frame_lab = cv2.cvtColor(frame_cv2, cv2.COLOR_BGR2LAB).astype(np.float32)
        l_frame, a_frame, b_frame = cv2.split(frame_lab)
        
        # Perform color transfer on A and B channels
        mean_a_frame, std_a_frame = cv2.meanStdDev(a_frame)
        mean_b_frame, std_b_frame = cv2.meanStdDev(b_frame)
        
        a_frame = (a_frame - mean_a_frame) * (std_a_ref / std_a_frame) + mean_a_ref
        b_frame = (b_frame - mean_b_frame) * (std_b_ref / std_b_frame) + mean_b_ref
        
        a_frame = np.clip(a_frame, 0, 255).astype(np.uint8)
        b_frame = np.clip(b_frame, 0, 255).astype(np.uint8)
        
        lab_result = cv2.merge((l_frame.astype(np.uint8), a_frame, b_frame))
        processed_frame_cv2 = cv2.cvtColor(lab_result, cv2.COLOR_LAB2BGR)
        
        processed_frames_pil.append(cv2_to_pil(processed_frame_cv2))
    
    return processed_frames_pil

def generate_probabilistic_mask(shape, probability):
    """
    生成一个按概率分布的掩码。
    
    Args:
        shape (tuple): 掩码的形状。
        probability (float): 掩码中 `True` 值的概率（0 到 1 之间）。

    Returns:
        np.ndarray: 按概率分布生成的布尔掩码。
    """
    if not (0 <= probability <= 1):
        raise ValueError("Probability must be between 0 and 1.")
    
    # 使用 numpy 的随机模块生成 0 到 1 之间的随机浮点数数组
    random_values = np.random.rand(*shape)
    
    # 生成布尔掩码，随机值小于概率的地方为 True
    mask = random_values < probability
    
    return mask

def pose_aug(target_pose_list, return_img=False, height=-1, width=-1, draw_face=False, draw_foot=False):
    """
    target_pose_list:[DWPose Dict格式结果]
    """
    assert (return_img and height > 0 and width > 0) or not return_img, '未提供hw时无法绘制'

    eps = 1e-8

    # 关节链接关系，DWPose代码定义的，有向且无重复的图
    # 身体的索引从1开始，取数时需要-1
    bodyLimbSeq = [[2, 3], [2, 6], [3, 4], [4, 5], [6, 7], [7, 8], [2, 9], [9, 10], \
               [10, 11], [2, 12], [12, 13], [13, 14], [2, 1], [1, 15], [15, 17], \
               [1, 16], [16, 18]]
    # 手部的索引从0开始，正常取用
    handLimbSeq = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [0, 9], [9, 10], \
             [10, 11], [11, 12], [0, 13], [13, 14], [14, 15], [15, 16], [0, 17], [17, 18], [18, 19], [19, 20]]

    pair_limbs = {
         (14,16):(15,17),
         (0,14):(0,15),
         (1,2):(1,5),
         (2,3):(5,6),
         (3,4):(6,7),
         (1,8):(1,11),
         (8,9):(11,12),
         (9,10):(12,13),
         # reverse
         (15,17):(14,16),
         (0,15):(0,14),
         (1,5):(1,2),
         (5,6):(2,3),
         (6,7):(3,4),
         (1,11):(1,8),
         (11,12):(8,9),
         (12,13):(9,10),
    }
    
    # 肢体构建为树结构方便重建人体
    # bfs建树，2号脖子节点作为树根
    body_tree = BodyTreeNode(2,[],None)
    queue = deque([body_tree])
    while queue:
        for i, j in bodyLimbSeq:
            if i == queue[0].val:
                new_node = BodyTreeNode(j,[],queue[0])
                queue[0].children.append(new_node)
                queue.append(new_node)
        queue.popleft()
    # bfs建树，0号手掌根部节点作为树根
    hand_tree = BodyTreeNode(0,[],None)
    queue = deque([hand_tree])
    while queue:
        for i, j in handLimbSeq:
            if i == queue[0].val:
                new_node = BodyTreeNode(j,[],queue[0])
                queue[0].children.append(new_node)
                queue.append(new_node)
        queue.popleft()

    # 对齐人体结构，reference pose提供肢体长度和整体位置（即根节点位置），target pose提供肢体朝向
    aligned_target_pose_list = []
    aligned_target_posesubset_list = []
    aligned_target_hand_list = []
    aligned_target_foot_list = []
    body_limb_ratio = {}
    hand_limb_ratio = {}
    
    for idx, target_pose in enumerate(target_pose_list):
        target_body = target_pose['bodies']['candidate']
        target_subset = target_pose['bodies']['subset'][0]
        target_hands = target_pose['hands']
        if isinstance(target_pose['hands'], dict):
            target_hands = target_hands['candidate']
        target_foot = target_pose['foot']['candidate']

        aligned_target_body = np.zeros(target_body.shape)
        aligned_target_subset = np.zeros(target_subset.shape)
        aligned_target_hands = np.zeros(target_hands.shape)
        aligned_target_foot = np.zeros(target_foot.shape)

        # bfs处理全部肢体，身体部分的索引需要-1
        node = body_tree
        # 从根节点开始构建新的人体姿态，先把根节点的位置对齐到reference的相同关节
        aligned_target_body[node.val-1] = target_body[node.val-1]
        queue = deque([node])

        while queue:
            mother_index = queue[0].val - 1
            for child in queue[0].children:
                child_index = child.val - 1
                queue.append(child)

                # 根据subset判断是否有未出的点，身体部分的关节并不是一定能检出的
                target_index = target_subset[np.array([mother_index, child_index])].astype(np.int32)

                aligned_target_subset[mother_index] = target_index[0]
                aligned_target_subset[child_index] = target_index[1]

                # 根据向量的模和方向重新定义新的子节点
                target_vec = target_body[target_index][1] - target_body[target_index][0]
                
                if (mother_index, child_index) not in body_limb_ratio.keys():
                    ratio = random.uniform(1.0,1.0)
                    body_limb_ratio[(mother_index, child_index)] = ratio
                    if (mother_index, child_index) in pair_limbs.keys():
                        body_limb_ratio[pair_limbs[(mother_index, child_index)]] = ratio
                else:
                    ratio = body_limb_ratio[(mother_index, child_index)]

                resized_target_vec = target_vec * ratio
                aligned_target_body[child_index] = aligned_target_body[mother_index] + resized_target_vec

            queue.popleft()
            
        aligned_target_pose_list.append(aligned_target_body)
        aligned_target_posesubset_list.append(aligned_target_subset[None,...])

        # 肢体的平均缩放系数
        body_avg_ratio = sum(body_limb_ratio.values())/len(body_limb_ratio)

        # bfs处理全部手掌
        node = hand_tree
        # 根节点初始化，掌根节点需要和重定向后的手腕关节对齐，否则会错位，5 和 8 为手腕关节
        right_wrist_index, left_wrist_index = int(target_subset[5-1]), int(target_subset[8-1])
        aligned_target_hands[:,node.val] = target_hands[:,node.val]

        if left_wrist_index != -1:
            aligned_target_hands[0,node.val] = aligned_target_body[left_wrist_index]
            aligned_target_hands[0,node.val] = aligned_target_body[left_wrist_index] + \
                                        target_hands[0,node.val] - target_body[left_wrist_index]

        if right_wrist_index != -1:
            aligned_target_hands[1,node.val] = aligned_target_body[right_wrist_index] + \
                                        target_hands[1,node.val] - target_body[right_wrist_index]

        queue = deque([node])
        while queue:
            mother_index = queue[0].val
            for child in queue[0].children:
                child_index = child.val
                queue.append(child)

                point_index = np.array([mother_index, child_index])

                # 根据向量的模和方向重新定义新的子节点
                target_vec = target_hands[:,point_index][:,1] - target_hands[:,point_index][:,0]
                if (mother_index, child_index) not in hand_limb_ratio.keys():

                    # 手部的ratio并不可靠，基于肢体估算ratio
                    ratio = body_avg_ratio
                    hand_limb_ratio[(mother_index, child_index)] = ratio
                else:
                    ratio = hand_limb_ratio[(mother_index, child_index)]
                resized_target_vec = target_vec * ratio
                aligned_target_hands[:,child_index] = aligned_target_hands[:,mother_index] + resized_target_vec
                
                # 如果母节点没有位置，子节点也置为-1，这边可能和原Pose并不一致，掌根节点未检出时直接放弃整个手掌
                if -1 in target_hands[:,point_index][0,0,:]:
                    aligned_target_hands[0,child_index,:] = -1
                if -1 in target_hands[:,point_index][1,0,:]:
                    aligned_target_hands[1,child_index,:] = -1

            queue.popleft()
        aligned_target_hand_list.append(aligned_target_hands)

        # 对齐脚掌点位
        aligned_target_foot[0,0:1,:] = aligned_target_body[13,:] + (target_foot[0,0:1,:] - target_body[13,:]) * body_avg_ratio
        aligned_target_foot[0,3:4,:] = aligned_target_body[10,:] + (target_foot[0,3:4,:] - target_body[10,:]) * body_avg_ratio
        aligned_target_foot_list.append(aligned_target_foot)

    # 全局缩放,平移,随机drop
    position_offset_xy = np.random.uniform(0, 0, 2)
    global_rescale = np.random.uniform(1.0, 1.0, 1)
    do_limb_drop_prob = 0.0
    limb_drop_ratio = 0.0
    limb_drop_prob = 0.0 if random.random() < do_limb_drop_prob else 0.0
    drop_cnt = 0
    for target_pose, aligned_target_pose, target_posesubset, aligned_target_hand, aligned_target_foot in \
        zip(target_pose_list, aligned_target_pose_list, aligned_target_posesubset_list, aligned_target_hand_list, aligned_target_foot_list):
        do_limb_drop = random.random() < limb_drop_prob

        if drop_cnt == 0 and do_limb_drop:
            drop_cnt = random.randint(3,15)
        
        target_pose['bodies']['candidate'] = aligned_target_pose*global_rescale + position_offset_xy
        target_posesubset_mask = generate_probabilistic_mask(target_posesubset.shape, limb_drop_ratio if drop_cnt>0 else 0.0)
        # 仅对部分关键点drop
        target_posesubset_mask[0, [0,1,2,5,14,15,16,17,]] = False
        target_posesubset[target_posesubset_mask] = -1
        target_pose['bodies']['subset'] = target_posesubset
        target_pose['foot']['candidate'] = aligned_target_foot*global_rescale + position_offset_xy

        hand_maks = generate_probabilistic_mask((2,), limb_drop_ratio if drop_cnt>0 else 0.0)
        aligned_target_hand[hand_maks,:,:] = -1
        if isinstance(target_pose['hands'], dict):
            target_pose['hands']['candidate'] = aligned_target_hand * global_rescale + position_offset_xy
        else:
            target_pose['hands'] = aligned_target_hand * global_rescale + position_offset_xy
        
        if drop_cnt > 0:
            drop_cnt -= 1
    
    # drop hands and face
    for idx, target_pose in enumerate(target_pose_list):
        # target_hands = target_pose['hands']
        # if isinstance(target_pose['hands'], dict):
        #     target_hands = target_hands['candidate']
        # target_hands.fill_(-1)
        continue
        target_pose_list[idx]['hands']['candidate'][:] = -1
        target_pose_list[idx]['bodies']['subset'][0,[14,15,16,17]] = -1
        

    if not return_img:
        return target_pose_list
    DRAW_HEIGHT = 1280
    DRAW_WIDTH = int(DRAW_HEIGHT / height * width)
    return [Image.fromarray(draw_pose(i, DRAW_HEIGHT, DRAW_WIDTH, draw_face, draw_foot)).resize((width, height)) for i in target_pose_list]

def draw_facepose(canvas, all_lmks):
    eps = 1e-2
    canvas = np.array(canvas)
    DRAW_HEIGHT = 1280
    H, W, C = canvas.shape
    DRAW_WIDTH = int(DRAW_HEIGHT / H * W)
    canvas = cv2.resize(canvas, (DRAW_WIDTH, DRAW_HEIGHT), interpolation=cv2.INTER_LINEAR)
    for lmks in all_lmks:
        lmks = np.array(lmks)
        for lmk in lmks:
            x, y = lmk
            x = int(x * DRAW_WIDTH)
            y = int(y * DRAW_HEIGHT)
            if x > eps and y > eps:
                cv2.circle(canvas, (x, y), 3, (255, 255, 255), thickness=-1)
    canvas = cv2.resize(canvas, (W, H), interpolation=cv2.INTER_LINEAR)
    canvas = Image.fromarray(canvas)
    return canvas

def padding_video(frame_list, hw_ratio):
    """
    frame_list 是一个 PIL 格式图片或 numpy 数组的列表，hw_ratio 是目标宽高比。
    通过 padding 把 frame_list 中的图片都 padding 成目标宽高比。
    
    :param frame_list: List of PIL Images or numpy arrays
    :param hw_ratio: Target height/width ratio (float)
    :return: List of images (same type as input) with the target aspect ratio
    """
    padded_frames = []
    is_numpy = False
    
    for frame in frame_list:
        if isinstance(frame, np.ndarray):
            is_numpy = True
            frame = Image.fromarray(frame)
        
        width, height = frame.size
        current_ratio = height / width
        
        if current_ratio < hw_ratio:
            # Image is wider than target ratio; pad height
            new_height = int(width * hw_ratio)
            vertical_padding = (new_height - height) // 2
            padding = (0, vertical_padding, 0, vertical_padding)
        else:
            # Image is taller than target ratio; pad width
            new_width = int(height / hw_ratio)
            horizontal_padding = (new_width - width) // 2
            padding = (horizontal_padding, 0, horizontal_padding, 0)

        # Adding padding to the image
        padded_frame = ImageOps.expand(frame, padding)
        if is_numpy:
            padded_frame = np.array(padded_frame)
        padded_frames.append(padded_frame)
    
    return padded_frames

def sharpen_frames(frames):
    if isinstance(frames, Image.Image):
        frame = np.array(frames)
        blur = cv2.GaussianBlur(frame.astype(np.float32), (5, 5), 0)
        sharpened_frame = 1.5 * frame.astype(np.float32) - 0.5 * blur.astype(np.float32)
        sharpened_frame = np.clip(sharpened_frame, 0, 255).astype(np.uint8)
        return Image.fromarray(sharpened_frame)
    elif isinstance(frames, list):
        return [sharpen_frames(i) for i in frames]
    else:
        raise NotImplementedError


def bbox_fix(bbox, box_width, box_height):
    left, top, right, bottom = bbox
    if right - left < box_width or bottom - top < box_height:
        return bbox
    if box_width > right-left:
        delta = box_width - (right-left)
        left = max(0, left-delta)
        right = left + box_width
    if box_height > bottom-top:
        delta = box_height - (bottom-top)
        top = max(0, top-delta)
        bottom = top + box_height
    bbox = (left, top, right, bottom)
    return bbox

def find_approximate_crop_box(points, frame_width, frame_height, box_width, box_height, step_size=50, alpha=0.5):
    """
    找到多个包含最多点并且点尽可能远离框边界的固定大小裁剪框的位置。

    参数:
    - points: (1000, 2) 的 NumPy 数组，每行表示一个点 (x, y)。
    - frame_width: 图片宽度。
    - frame_height: 图片高度。
    - box_width: 裁剪框的固定宽度。
    - box_height: 裁剪框的固定高度。
    - step_size: 滑窗的步长，默认为50。
    - alpha: 权重系数，用于调整点数和距离中心的权重，默认为0.5。
    返回:
    - best_box: 包含多个最佳裁剪框坐标 (left, top, right, bottom)
    """
    
    if points.size == 0:
        raise ValueError("点位数组为空")

    point_array = np.array(points)
    point_array = point_array[(point_array[:, 0] >= 0) & (point_array[:, 0] <= frame_width) & (point_array[:, 1] >= 0) & (point_array[:, 1] <= frame_height)]

    best_box = None
    max_score = -9999999999
    for left in range(0, frame_width - box_width + step_size, step_size):
        for top in range(0, frame_height - box_height + step_size, step_size):
            right = left + box_width
            bottom = top + box_height

            # 计算在这个框内的点数
            in_box = (point_array[:, 0] >= left) & (point_array[:, 0] <= right) & \
                        (point_array[:, 1] >= top) & (point_array[:, 1] <= bottom)
            in_count = np.sum(in_box)
            
            if in_count > 0:
                # 计算这些点到框中心的平均距离
                box_center = np.array([left + box_width / 2, top + box_height / 2])
                points_in_box = point_array[in_box]
                distances = np.linalg.norm(points_in_box - box_center, axis=1)
                avg_distance = np.mean(distances)
                
                # 综合评估得分：点数和距离的组合
                score = in_count - alpha * avg_distance
                if score > max_score or best_box is None:
                    max_score = score
                    box = (left, top, right, bottom)
                    best_box = bbox_fix(box, box_width=box_width, box_height=box_height)

    return best_box

def crop_frames_to_target_ratio(frames, pose_points, target_wh_ratio, crop_face=False):
    """
    裁剪帧列表中的帧，使其宽高比与目标宽高比匹配。

    :param frames: 包含帧的列表，每个帧都是一个 PIL 图像。
    :param poses: 包含关键点的列表，每个关键点都是一个字典。
    :param target_ratio: 目标宽高比。
    :return: 裁剪后的帧列表。
    """
    cropped_frames = []
    frame_w, frame_h = frames[0].size

    if crop_face:
        target_x, target_y = 0, 0
        # 计算y轴最靠下的
        for pose_pkl in pose_points:
            body = pose_pkl['bodies']['candidate']
            subset = pose_pkl['bodies']['subset']
            # 取01关键点的中间位置
            p0 = body[0, :]
            p1 = body[1, :]
            x, y = 0.6 * p0 + 0.4 * p1
            if y > target_y:
                target_x, target_y = x, y

        for pose_pkl in pose_points:
            pose_pkl['bodies']['candidate'][:, 1] -= target_y
            pose_pkl['foot']['candidate'][0,:,1] -= target_y
            # pose_pkl['hands']['candidate'][:,:,1] -= target_y
            pose_pkl['bodies']['candidate'][:, 1] /= 1 - target_y
            pose_pkl['foot']['candidate'][0,:,1] /= 1 - target_y
            # pose_pkl['hands']['candidate'][:,:,1] /= 1 - target_y

        crop_bbox = (0, int(16*round(target_y*frame_h/16)), frame_w, frame_h)
        frames = [frame.crop(crop_bbox) for frame in frames]

        frame_w, frame_h = frames[0].size

    if frame_w / frame_h < target_wh_ratio:
        # 裁高度
        target_w = int(16 * round(frame_w / 16))
        target_h = int(16 * round(target_w / target_wh_ratio / 16))
    else:
        # 裁宽度
        target_h = int(16 * round(frame_h / 16))
        target_w = int(16 * round(target_h * target_wh_ratio / 16))
    
    all_points = []
    for pose_pkl in pose_points:
        all_points.append(pose_pkl['bodies']['candidate'])
        all_points.append(pose_pkl['foot']['candidate'][0,3:4,:])
        all_points.append(pose_pkl['foot']['candidate'][0,0:1,:])
        if pose_pkl['bodies']['candidate'][14,0] > 0 and pose_pkl['bodies']['candidate'][15,0] < 1:
            height = abs(pose_pkl['bodies']['candidate'][14,0] - pose_pkl['bodies']['candidate'][15,0]) * 2
            point_0 = pose_pkl['bodies']['candidate'][0,:]
            midpoint_14_15 = (pose_pkl['bodies']['candidate'][14,:] + pose_pkl['bodies']['candidate'][15,:]) / 2
            vec = midpoint_14_15 - point_0
            target_vec = vec * 6
            target_point = point_0 + target_vec
            target_point = target_point.clip(0, 1)
            for i in range(5):
                all_points.append(target_point[np.newaxis,:])

    all_points = np.concatenate(all_points, axis=0)
    all_points = all_points[((all_points >= 0.0) & (all_points <= 1.0)).all(axis=1)]
    all_points[:, 0] = (all_points[:, 0] * frame_w)
    all_points[:, 1] = (all_points[:, 1] * frame_h)
    target_box = find_approximate_crop_box(
        all_points, 
        frame_w, 
        frame_h, 
        target_w, 
        target_h, 
        step_size=int(frame_w*0.025), 
        alpha=0.5
    )
    cropped_frames = [frame.crop(target_box) for frame in frames]

    return cropped_frames