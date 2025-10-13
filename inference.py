# // Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //     http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.

# Single-video DreamSwapV 推理脚本
# Author: victor-thu
# Date: 2025/9/20

import os
import cv2
import argparse
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
import pickle as pkl
import logging
import re

# -------- diffusers / wanx --------
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video
from wanx.wan_pipeline import WanImageToVideoPipeline
from wanx.wan_transformers_pose import WanAttnProcessorFlash, get_wanx_diffusers

# -------- DWPose / Hamer 相关 --------
from utils.dwpose import DWposeDetector, draw_pose
from utils.hamer.draw_hand import Hamer, overlay_images
from utils import resize_and_centercrop, pose_smooth, posehand_threshold

# -------- TrackingSAM 相关 --------
from utils.sam.seg_track_anything import aot_model2ckpt, tracking_objects_in_video
from utils.sam.model_args          import segtracker_args, sam_args, aot_args
from utils.sam.SegTracker          import SegTracker

# -------------------------- 常量与工具 --------------------------

def setup_logger():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("single_infer")

def ensure_4n_plus_1_leq(max_len:int) -> int:
    """将给定 max_len 调整为 <= max_len 的最大 4n+1 数"""
    if max_len <= 0:
        raise ValueError("max_len must be positive")
    # 寻找不超过 max_len 的 4n+1
    L = max_len - ((max_len - 1) % 4)
    if L < 1:
        L = 1
    return L

def split_segments(total, max_length):
    """
    将序列分为若干段，段长满足 ≤ max_length 且 (len % 4 == 1)，
    相邻段重叠 1 帧，以便跨段衔接 first_inpainted_frame。
    返回 [(start, end), ...]，end 为开区间。
    """
    segments = []
    start = 0
    while start < total:
        end = min(start + max_length, total)
        if end - start == 1:
            break
        while (end - start) % 4 != 1:
            end -= 1
            if end - start <= 0:
                raise ValueError("Cannot find a valid segment size")
        segments.append((start, end))
        if end - start == max_length:
            start = end - 1  # 与下一段重叠 1 帧
        else:
            break
    return segments

def init_SegTracker(origin_frame_bgr, gpu_id=0,
                    aot_model="r50_deaotl",
                    long_term_mem=9999, max_len_long_term=9999,
                    sam_gap=9999, max_obj_num=255, points_per_side=16):
    """新建并重置 SegTracker"""
    if origin_frame_bgr is None:
        return None
    aot_args["model"]                = aot_model
    aot_args["model_path"]           = aot_model2ckpt[aot_model]
    aot_args["long_term_mem_gap"]    = long_term_mem
    aot_args["max_len_long_term"]    = max_len_long_term
    aot_args["gpu_id"]               = gpu_id

    segtracker_args["sam_gap"]       = sam_gap
    segtracker_args["max_obj_num"]   = max_obj_num

    sam_args["generator_args"]["points_per_side"] = points_per_side
    sam_args["gpu_id"]               = gpu_id

    tracker = SegTracker(segtracker_args, sam_args, aot_args)
    tracker.restart_tracker()
    return tracker

def segtracker_add_first_frame(tracker, frame_bgr, mask_np):
    """塞入首帧与其二值 mask"""
    tracker.restart_tracker()
    tracker.add_reference(frame_bgr, mask_np, 0)
    tracker.first_frame_mask = mask_np
    return tracker

def tracking_masks(tracker, video_path, total_len, fps):
    """返回长度为 total_len 的 mask 列表（np.uint8）"""
    return tracking_objects_in_video(tracker, video_path, total_len, fps)

def load_ref_with_gray_bg(path: str, gray=(128, 128, 128)) -> Image.Image:
    """将带 Alpha 的参考图转为灰底 RGB（若无 Alpha 也可正常读入）"""
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (*gray, 255))
    return Image.alpha_composite(bg, img).convert("RGB")

def center_image_to_canvas(image: Image.Image, canvas_size, padding=20,
                           target_area_ratio=0.4, mask_color=(128,128,128)) -> Image.Image:
    """把参考图按比例居中铺到 canvas（RGB）"""
    image = np.array(image)
    h_orig, w_orig = image.shape[:2]
    canvas_w, canvas_h = canvas_size
    effective_w = canvas_w - 2 * padding
    effective_h = canvas_h - 2 * padding
    if effective_w <= 0 or effective_h <= 0:
        raise ValueError("Padding too large.")

    effective_area = effective_w * effective_h
    target_area    = effective_area * target_area_ratio
    scale_area = np.sqrt(target_area / (w_orig * h_orig)) if (w_orig*h_orig) > 0 else 1.0
    scale_fit  = min(effective_w / w_orig, effective_h / h_orig) if max(w_orig,h_orig)>0 else 1.0
    final_scale = min(scale_area, scale_fit)
    new_w, new_h = int(w_orig * final_scale), int(h_orig * final_scale)

    canvas = np.full((canvas_h, canvas_w, 3), mask_color, dtype=np.uint8)
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_offset = padding + (effective_w - new_w) // 2
    y_offset = padding + (effective_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
    return Image.fromarray(canvas).convert("RGB")

def mask_augmentation(image: Image.Image, mask: Image.Image, mask_color=(128,128,128), bounding=False):
    """格子状 mask 增强，返回 (new_mask, agnostic_image)"""
    image_array = np.array(image)
    mask_np = np.array(mask) > 128
    h, w = mask_np.shape
    if not bounding:
        # 使用全图网格（与原脚本保持一致的 kh/kw 计算）
        kh = h // 60
        kw = w // 60
        row_indices = np.linspace(0, h, kh+1, dtype=int, endpoint=True)
        col_indices = np.linspace(0, w, kw+1, dtype=int, endpoint=True)
        new_mask = np.zeros_like(mask_np)
        for i in range(kh):
            for j in range(kw):
                r0, r1 = row_indices[i], row_indices[i+1]
                c0, c1 = col_indices[j], col_indices[j+1]
                block = mask_np[r0:r1, c0:c1]
                if np.any(block):
                    new_mask[r0:r1, c0:c1] = 1
    else:
        coords = np.argwhere(mask_np)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            new_mask = np.zeros_like(mask_np)
            new_mask[y_min:y_max+1, x_min:x_max+1] = 1
        else:
            new_mask = mask_np

    mask_color = np.array(mask_color, dtype=np.uint8).reshape(1,1,3)
    agnostic_image = np.where(new_mask[..., None] != 1, image_array, mask_color)
    return Image.fromarray((new_mask * 255).astype(np.uint8)), Image.fromarray(agnostic_image.astype(np.uint8))

def read_video_to_pil(path: str, max_frames: int = None):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, first = [], None
    i = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if first is None:
            first = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
        i += 1
        if max_frames is not None and i >= max_frames:
            break
    cap.release()
    return frames, first, fps

def concat_imgs_h(imgs):
    """横向拼接 PIL 图片，按最小高度等比缩放并底对齐"""
    # 1. 找到最小高度
    min_h = min(img.height for img in imgs)

    # 2. 缩放所有图片到相同高度
    scaled_imgs = []
    for img in imgs:
        if img.height != min_h:
            # 按比例缩放宽度
            new_w = int(img.width * min_h / img.height)
            img = img.resize((new_w, min_h), Image.LANCZOS)
        scaled_imgs.append(img)

    # 3. 计算总宽度并新建画布
    total_w = sum(im.width for im in scaled_imgs)
    dst = Image.new("RGB", (total_w, min_h))

    # 4. 拼接
    x = 0
    for im in scaled_imgs:
        dst.paste(im, (x, 0))
        x += im.width

    return dst

# ------------------------------ 主流程 ------------------------------
def run_single(args):
    logger = setup_logger()

    # 解析基本参数
    video_path      = Path(args.video).expanduser().resolve()
    first_mask_path = Path(args.first_mask).expanduser().resolve()
    ref_path        = Path(args.ref).expanduser().resolve()
    out_dir         = Path(args.output_dir).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    assert video_path.exists(), f"Video not found: {video_path}"
    assert first_mask_path.exists(), f"First mask not found: {first_mask_path}"
    assert ref_path.exists(), f"Reference image not found: {ref_path}"

    device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # prompt embeds（不需要 prompt，wanx 原接口用零向量）
    prompt_embeds = torch.zeros((1, 6, 4096))

    # 模型
    transformer, _ = get_wanx_diffusers(None, args.checkpoint, use_text_encoder=False, use_fa3=True)
    transformer = transformer.to(device, dtype=torch.bfloat16)
    transformer.set_attn_processor(WanAttnProcessorFlash())

    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers",  
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    scheduler = UniPCMultistepScheduler(
        prediction_type='flow_prediction',
        use_flow_sigmas=True,
        num_train_timesteps=1000,
        flow_shift=5
    )
    dreamswapv_pipe = WanImageToVideoPipeline(transformer, vae, scheduler, None)

    # DWPose / Hamer
    gpu_idx = 0 if "cuda" in args.device else -1
    dwpose_detector = DWposeDetector(cuda_idx=gpu_idx if gpu_idx>=0 else 0)
    hand_drawer     = Hamer(gpu_idx if gpu_idx>=0 else 0)

    # 视频与尺寸
    # 缩放：短边≤720 且对齐16
    orig_h, orig_w = args.height, args.width
    min_dim = min(orig_h, orig_w)
    scale   = 720.0 / min_dim if min_dim > 720 else 1.0
    scaled_h, scaled_w = int(orig_h * scale), int(orig_w * scale)
    height = ((scaled_h + 15) // 16) * 16
    width  = ((scaled_w  + 15) // 16) * 16

    video_list, first_frame_bgr, fps = read_video_to_pil(str(video_path), max_frames=args.total_length)

    # 参考图准备（灰底转 RGB → canvas 居中）
    refimg_list = [load_ref_with_gray_bg(str(ref_path))]
    refimg_list = [center_image_to_canvas(img, (width, height)) for img in refimg_list]

    # SegTracker：用首帧 mask 初始化并跟踪
    tracker = init_SegTracker(first_frame_bgr, gpu_id=gpu_idx if gpu_idx>=0 else 0)
    fm_pil = Image.open(first_mask_path).convert("L")

    tracker = segtracker_add_first_frame(tracker, first_frame_bgr, np.array(fm_pil))
    tracked = tracking_masks(tracker, str(video_path), len(video_list), fps)
    masks_np = [np.array(fm_pil)] + [np.array(m) for m in tracked[1:]]
    mask_list = [Image.fromarray(m.astype(np.uint8)) for m in masks_np]
    # mask_list = [resize_and_centercrop(i, height, width) for i in mask_list]

    # 分段（满足 4n+1，且段长≤frame_length）
    frame_length = ensure_4n_plus_1_leq(args.frame_length)
    segments = split_segments(len(video_list), frame_length)
    logger.info(f"Segments: {segments} (count={len(segments)})")

    # 输出命名
    vid_stem   = video_path.stem.split("-")[0]
    ref_stem   = ref_path.stem
    output_name = f"{vid_stem}_{ref_stem}"
    output_path = out_dir / f"{output_name}.mp4"
    debug_path  = out_dir / f"{output_name}_debug.mp4"

    final_outputs, final_video, final_mask, final_pose, final_agnostic = [], [], [], [], []
    first_inpainted_frame = None

    for seg_idx, (s, e) in enumerate(tqdm(segments, desc="Segments")):
        seg_videos = video_list[s:e]
        seg_masks  = mask_list[s:e]

        # 生成 pose / agnostic / new_mask
        pose_list, agnostic_list, new_mask_list = [], [], []
        for img, msk in zip(seg_videos, seg_masks):
            mask_new, agnostic = mask_augmentation(img, msk)
            agnostic_list.append(resize_and_centercrop(agnostic, height, width))
            new_mask_list.append(resize_and_centercrop(mask_new, height, width))

            v_h, v_w = img.size[1], img.size[0]
            pose_dict = posehand_threshold(dwpose_detector(img, return_dict=True))
            pose_dict = pose_smooth([pose_dict], height=v_h, width=v_w, return_img=False)[0]
            pose_img  = Image.fromarray(
                draw_pose(pose_dict, H=v_h, W=v_w, draw_face=True, draw_foot=True, draw_hand=False)
            )
            hand_img  = hand_drawer.draw_hand(Image.new("RGB", (v_w, v_h), (0,0,0)), pose_dict, img.resize((v_w, v_h)))
            pose_final = overlay_images(pose_img, hand_img)
            pose_list.append(resize_and_centercrop(pose_final, height, width))

        seg_videos = [resize_and_centercrop(i, height, width) for i in seg_videos]
        seg_masks  = [resize_and_centercrop(i, height, width) for i in seg_masks]

        # call pipeline
        with torch.cuda.amp.autocast():
            seg_out = dreamswapv_pipe.__mask_call__(
                prompt=None,
                prompt_embeds=prompt_embeds,
                height=height, width=width,
                video_length=len(seg_videos),
                seed=args.seed,
                num_inference_steps=args.num_steps,
                guidance_scale=args.guidance_scale,
                ref_img=refimg_list,
                mask_image=new_mask_list,
                agnostic_image=agnostic_list,
                pose_image=pose_list,
                num_videos_per_prompt=1,
                device=device,
                first_inpainted_frame=first_inpainted_frame
            ).frames[0]

        # 更新跨段首帧
        first_inpainted_frame = Image.fromarray((seg_out[-1] * 255).astype(np.uint8))

        # 去重叠帧
        if seg_idx > 0:
            seg_out       = seg_out[1:]
            pose_list     = pose_list[1:]
            agnostic_list = agnostic_list[1:]
            seg_masks     = seg_masks[1:]
            seg_videos    = seg_videos[1:]

        final_outputs.extend(seg_out)
        final_pose.extend(pose_list)
        final_agnostic.extend(agnostic_list)
        final_mask.extend(seg_masks)
        final_video.extend(seg_videos)

        del pose_list, agnostic_list, new_mask_list, seg_out
        torch.cuda.empty_cache()

    # 导出结果
    export_to_video(final_outputs, str(output_path), fps=fps)
    logger.info(f"Saved: {output_path}")

    # 可选：debug 横向拼接导出
    if args.save_debug:
        debug_frames = []
        for i, out_frame in enumerate(final_outputs):
            debug_frames.append(
                concat_imgs_h(
                    refimg_list + [
                        final_video[i],
                        final_mask[i],
                        final_agnostic[i],
                        final_pose[i],
                        Image.fromarray((out_frame * 255).astype(np.uint8))
                    ]
                )
            )
        export_to_video(debug_frames, str(debug_path), fps=fps)
        logger.info(f"Saved debug: {debug_path}")
    
    del video_list, mask_list
    torch.cuda.empty_cache()

# ------------------------------ CLI ------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Single-video DreamSwapV inference with first-frame mask")
    p.add_argument("--video", required=True, help="输入视频路径（.mp4）")
    p.add_argument("--first_mask", required=True, help="首帧二值 mask（L 模式，白=前景），尺寸自动对齐首帧")
    p.add_argument("--ref", required=True, help="参考图路径（要求透明 PNG）")
    p.add_argument("--checkpoint", required=True, help="DreamSwapV checkpoint 路径")

    p.add_argument("--output_dir", default="./outputs", help="输出目录")
    p.add_argument("--device", default="cuda:0", help="设备，如 cuda:0 / cpu")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--num_steps", type=int, default=50, help="推理步数")
    p.add_argument("--save_debug", action="store_true", help="保存 debug 横拼视频")
    p.add_argument("--total_length", type=int, default=None, help="生成视频的长度，置空则默认生成全长视频")
    p.add_argument("--frame_length", type=int, default=69, help="分段长度，默认为训练长度69，要求为4n+1，一般情况不建议调整")
    p.add_argument("--height", type=int, default=1280, help="生成视频的高度")
    p.add_argument("--width", type=int, default=720, help="生成视频的宽度")

    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run_single(args)
