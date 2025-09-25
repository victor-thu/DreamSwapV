#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Benchmark-based DreamSwapV 推理脚本
Author: victor-thu
Date: 2025/9/20
"""
import os
import cv2
import argparse
from tqdm import tqdm
import torch
import numpy as np
from glob import glob
from pathlib import Path
from datetime import datetime
from PIL import Image
import logging
import json
import pickle as pkl
import torch.multiprocessing as mp
import re

from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from diffusers import AutoencoderKLWan
from diffusers.utils import export_to_video

from utils.dwpose import DWposeDetector, draw_pose
from utils.hamer.draw_hand import Hamer, overlay_images
from utils import (
    resize_and_centercrop,
    pose_smooth,
    posehand_threshold,
)

from wanx.wan_pipeline import WanImageToVideoPipeline
from wanx.wan_transformers_pose import WanAttnProcessorFlash, get_wanx_diffusers

# ------ SAM / SegTracker 相关 ------
from utils.sam.seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from utils.sam.model_args          import segtracker_args, sam_args, aot_args
from utils.sam.SegTracker          import SegTracker
# ------ SAM / SegTracker 相关 ------
from utils.sam.seg_track_anything import aot_model2ckpt, tracking_objects_in_video, draw_mask
from utils.sam.model_args          import segtracker_args, sam_args, aot_args
from utils.sam.SegTracker          import SegTracker


# -------------------------- 工具 / 辅助函数 --------------------------

# ---------------------- 分段：overlap = 1 帧 ----------------------
def split_segments(total, max_length):
    """
    Splits lists of frames and masks into overlapping segments,
    where each segment has length <= max_length and
    overlaps the next by one frame.
    Returns list of (segment_frames, segment_masks).
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
            start = end - 1  # overlap last frame
        else:
            break
    return segments

def setup_logger(name, log_file, level=logging.INFO):
    handler   = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.addHandler(handler)
    return logger

# ---------------- SegTracker 封装 ----------------
def init_SegTracker(origin_frame, gpu_id=0,
                    aot_model="r50_deaotl",
                    long_term_mem=9999, max_len_long_term=9999,
                    sam_gap=9999, max_obj_num=255, points_per_side=16):
    """新建并重置 SegTracker"""
    if origin_frame is None:
        return None
    # 重设 AOT/SAM 参数
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

def SegTracker_add_first_frame(tracker, frame_bgr, mask_np):
    """把首帧 + mask 塞进 SegTracker"""
    tracker.restart_tracker()
    tracker.add_reference(frame_bgr, mask_np, 0)
    tracker.first_frame_mask = mask_np
    return tracker

def gd_detect(tracker, gpu_id, frame_bgr, caption,
              box_th=0.25, text_th=0.25):
    """
    grounding-dino → SAM → AOT，一次性在首帧自动检测+分割
    返回 (tracker, 首帧mask_np, 绘制后首帧)
    """
    if tracker is None:
        tracker = init_SegTracker(frame_bgr, gpu_id)
    mask_np, annotated = tracker.detect_and_seg(frame_bgr, caption,
                                                box_th, text_th)
    tracker = SegTracker_add_first_frame(tracker, frame_bgr, mask_np)
    return tracker, mask_np, annotated

def tracking_objects(tracker, video_path, total_len, fps):
    """从第 2 帧开始一路跟踪，返回 [mask_np...] 长度 = total_len"""
    return tracking_objects_in_video(tracker, video_path, total_len, fps)


def load_ref_with_gray_bg(path: str, gray=(128, 128, 128)) -> Image.Image:
    """将带 Alpha 的参考图转为灰底 RGB"""
    img = Image.open(path).convert("RGBA")
    bg  = Image.new("RGBA", img.size, (*gray, 255))
    return Image.alpha_composite(bg, img).convert("RGB")

def build_tasks_from_benchmark(bench_root: str) -> list[dict]:
    """
    bench_root/
        lands_mask/ | verti_mask/
            {video_name}/
                {subject}/
                    masks/00000.png-00068.png
                    ref.png  (透明)
        lands_video/ | verti_video/
            {video_name}.mp4
    """
    tasks = []

    for t in ("lands", "verti"):
        mask_root  = Path(bench_root) / f"{t}_mask"
        video_root = Path(bench_root) / f"{t}"
        if not mask_root.exists() or not video_root.exists():
            continue

        for video_dir in mask_root.iterdir():
            if not video_dir.is_dir():
                continue
            video_name = video_dir.name

            # 可选，指定 benchmark 中想要专门跑的一些视频
            # def num_pref(name: str) -> str | None:
            #     """捕获文件/文件夹开头的连续数字，如 '852122-hd...' → '852122'"""
            #     m = re.match(r"^(\d+)", name)
            #     return m.group(1) if m else None
            # ALLOW_LIST = {
            #     "14071724", "4623570", "5057325", "6520307", "7187515", "8970369", "9017890", "855564"
            # }
            # vid_id = num_pref(video_name)           # ← 提取数字前缀
            # if vid_id not in ALLOW_LIST:            # ← 不在名单，跳过
            #     continue

            video_path = video_root / f"{video_name}.mp4"
            if not video_path.exists():
                print(f"[WARN] MISSING VIDEO: {video_path}")
                continue

            for subject_dir in video_dir.iterdir():
                if not subject_dir.is_dir():
                    continue
                mask_folder = subject_dir / "masks"
                if not mask_folder.exists():
                    print(f"[WARN] MISSING MASK FOLDER: {mask_folder}")
                    continue

                ref_candidates = sorted(
                    [p for p in subject_dir.glob("*") 
                     if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".webp")
                     and p.parent.name != "masks"]
                )
                if not ref_candidates:
                    print(f"[WARN] MISSING REF IMG: {subject_dir}")
                    continue
                ref_path = ref_candidates[1]
                first_mask_path = ref_candidates[0]

                print(ref_path, first_mask_path)

                tasks.append({
                    "video_path"   : str(video_path),
                    "mask_folder"  : str(mask_folder),
                    "refimg_path"  : [str(ref_path)],
                    "first_mask_path": str(first_mask_path),   # 兼容旧字段
                    "caption"        : subject_dir.name
                })
    return tasks

def center_image_to_canvas(
        image: np.ndarray,
        canvas_size,
        padding: int = 20,
        target_area_ratio: float = 0.4,
        mask_color: tuple = (128, 128, 128)
    ) -> np.ndarray:

    image = np.array(image)
    h_orig, w_orig = image.shape[:2]
    canvas_w, canvas_h = canvas_size
    effective_w = canvas_w - 2 * padding
    effective_h = canvas_h - 2 * padding
    if effective_w <= 0 or effective_h <= 0:
        raise ValueError("Padding too large.")

    effective_area = effective_w * effective_h
    target_area    = effective_area * target_area_ratio
    scale_area = np.sqrt(target_area / (w_orig * h_orig)) if (w_orig * h_orig) > 0 else 1.0
    scale_fit  = min(effective_w / w_orig, effective_h / h_orig) if max(w_orig, h_orig) > 0 else 1.0
    final_scale = min(scale_area, scale_fit)
    new_w, new_h = int(w_orig * final_scale), int(h_orig * final_scale)

    canvas = np.full((canvas_h, canvas_w, 3), mask_color, dtype=np.uint8)
    resized_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    x_offset = padding + (effective_w - new_w) // 2
    y_offset = padding + (effective_h - new_h) // 2
    canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_img
    return Image.fromarray(canvas).convert("RGB")

def mask_augmentation(image, mask, mask_color=(128, 128, 128), bounding=False):
    image_array = np.array(image)
    mask = np.array(mask) > 128
    h, w = mask.shape
    if not bounding:
        bbox_h, bbox_w = (np.max(np.where(mask), axis=1) - np.min(np.where(mask), axis=1) + 1) if np.any(mask) else (0, 0)
        '''kh = bbox_h // 20
        kw = bbox_w // 20'''
        kh = h // 60
        kw = w // 60
        row_indices = np.linspace(0, h, kh+1, dtype=int, endpoint=True)
        col_indices = np.linspace(0, w, kw+1, dtype=int, endpoint=True)
        new_mask = np.zeros_like(mask)
        for i in range(kh):
            for j in range(kw):
                r0, r1 = row_indices[i], row_indices[i+1]
                c0, c1 = col_indices[j], col_indices[j+1]
                block = mask[r0:r1, c0:c1]
                if np.any(block):
                    new_mask[r0:r1, c0:c1] = 1
    else:
        coords = np.argwhere(mask)
        if coords.size > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            new_mask = np.zeros_like(mask)
            new_mask[y_min:y_max+1, x_min:x_max+1] = 1
        else:
            new_mask = mask
    mask_color = np.array(mask_color, dtype=np.uint8).reshape(1, 1, 3)
    agnostic_image = np.where(
        new_mask[..., np.newaxis] != 1,
        image_array,
        mask_color
    )
    return Image.fromarray((new_mask * 255).astype(np.uint8)), Image.fromarray(agnostic_image.astype(np.uint8))

def read_video_to_pil(path: str, max_frames=69):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise IOError(f"Cannot open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frames, first = [], None
    for i in range(max_frames):
        success, frame = cap.read()
        if not success:
            break
        if first is None:
            first = frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(Image.fromarray(frame_rgb))
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

# ------------------------------ main 逻辑 ------------------------------
def main(args):
    (tasks, output_dir, checkpoint_path, guidance_scale,
     frame_length, seed, num_steps, 
     save_debug, total_length, cuda_idx) = args
    
    os.makedirs(output_dir, exist_ok=True)

    rank   = int(os.environ.get("RANK", 0))
    logger = setup_logger(f"gpu{cuda_idx}", f"./log/RANK_{rank}_cuda{cuda_idx}.log")
    device = torch.device(f"cuda:{cuda_idx}")
    logger.info(f"Device set: {device}")

    # prompt embeds（不需要 prompt，wanx 原接口用零向量）
    prompt_embeds = torch.zeros((1, 6, 4096))

    # model
    transformer, _ = get_wanx_diffusers(None, checkpoint_path, use_text_encoder=False, use_fa3=True)
    transformer = transformer.to(device, dtype=torch.bfloat16)
    transformer.set_attn_processor(WanAttnProcessorFlash())

    vae = AutoencoderKLWan.from_pretrained(
        "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers", 
        subfolder="vae",
        torch_dtype=torch.float16
    ).to(device)
    scheduler = UniPCMultistepScheduler(prediction_type='flow_prediction', use_flow_sigmas=True,
                                        num_train_timesteps=1000, flow_shift=5)
    dreamswapv_pipe = WanImageToVideoPipeline(transformer, vae, scheduler, None)

    dwpose_detector = DWposeDetector(cuda_idx=cuda_idx)
    hand_drawer     = Hamer(cuda_idx)

    for task in tqdm(tasks, desc=f"GPU{cuda_idx} processing"):
        vid        = Path(task["video_path"]).stem.split("-")[0]
        ref_stem   = Path(task["refimg_path"][0]).stem
        output_name       = f"{vid}_{ref_stem}"
        output_path       = f"{output_dir}/{output_name}.mp4"
        debug_output_path = f"{output_dir}/{output_name}_debug.mp4"
        if os.path.exists(output_path):
            continue

        # reference
        refimg_list = [load_ref_with_gray_bg(p) for p in task["refimg_path"]]

        # video frames
        video_list, first_frame, fps = read_video_to_pil(task["video_path"], max_frames=total_length)
        frame_length = len(video_list)

        # mask list (69 png)
        # -------- 在线检测+跟踪，生成与 video_list 等长的 mask_list --------
        # 首帧 BGR（cv2 格式）已在 read_video_to_pil() 里拿到 first_frame
        tracker = init_SegTracker(first_frame, gpu_id=cuda_idx)

        # 加载首帧 mask
        first_mask_path = task.get("first_mask_path")
        fm_pil = Image.open(first_mask_path).convert("L")
        tracker = SegTracker_add_first_frame(tracker, first_frame, np.array(fm_pil))
        masks_np = [np.array(fm_pil)] + [np.array(m)                   
                                for m in tracking_objects(
                                    tracker, task["video_path"],
                                    len(video_list), fps)[1:]]

        mask_list = [Image.fromarray(m.astype(np.uint8)) for m in masks_np]

        if len(mask_list) < frame_length:
            logger.warning(f"Mask < {frame_length}: {task['mask_folder']}")


        # 原分辨率 → 缩到短边 ≤ 720，然后对齐 16
        orig_h, orig_w = first_frame.shape[:2]
        min_dim = min(orig_h, orig_w)
        scale   = 720.0 / min_dim if min_dim > 720 else 1.0
        scaled_h, scaled_w = int(orig_h * scale), int(orig_w * scale)
        height = ((scaled_h + 15) // 16) * 16
        width  = ((scaled_w + 15) // 16) * 16

        # ----------- (A) 准备收集全片输出 / debug -----------
        final_outputs, final_video, final_mask, final_pose, final_agnostic = [], [], [], [], []
        first_inpainted_frame = None      # 核心衔接变量

        # 分段
        segments = split_segments(len(video_list), frame_length)

        for seg_idx, (s, e) in enumerate(segments):
            seg_videos = video_list[s:e]
            seg_masks  = mask_list[s:e]

            # ---------- (B) 针对本段生成 pose / agnostic / mask ----------
            pose_list, agnostic_list, new_mask_list = [], [], []
            for img, msk in zip(seg_videos, seg_masks):
                mask_new, agnostic = mask_augmentation(img, msk)
                agnostic_list.append(resize_and_centercrop(agnostic, height, width))
                new_mask_list.append(resize_and_centercrop(mask_new, height, width))

                v_h, v_w = img.size[1], img.size[0]
                pose_dict = posehand_threshold(dwpose_detector(img, return_dict=True))
                pose_dict = pose_smooth([pose_dict], height=v_h, width=v_w, return_img=False)[0]
                pose_img = Image.fromarray(
                    draw_pose(pose_dict, H=v_h, W=v_w,
                            draw_face=True, draw_foot=True, draw_hand=False)
                )
                hand_img = hand_drawer.draw_hand(
                    Image.new("RGB", (v_w, v_h), (0, 0, 0)), pose_dict,
                    img.resize((v_w, v_h)))
                pose_final = overlay_images(pose_img, hand_img)
                pose_list.append(resize_and_centercrop(pose_final, height, width))

            seg_videos = [resize_and_centercrop(i, height, width) for i in seg_videos]
            seg_masks  = [resize_and_centercrop(i, height, width) for i in seg_masks]

            refimg_list = [center_image_to_canvas(i, (width, height)) for i in refimg_list]

            # ---------- (C) call pipeline ----------
            with torch.cuda.amp.autocast():
                seg_out = dreamswapv_pipe.__mask_call__(
                    prompt=None,
                    prompt_embeds=prompt_embeds,
                    height=height, width=width,
                    video_length=len(seg_videos),
                    seed=seed,
                    num_inference_steps=num_steps,
                    guidance_scale=guidance_scale,
                    ref_img=refimg_list,
                    mask_image=new_mask_list,
                    agnostic_image=agnostic_list,
                    pose_image=pose_list,
                    num_videos_per_prompt=1,
                    device=device,
                    first_inpainted_frame=first_inpainted_frame
                ).frames[0]

            # 更新 first_inpainted_frame (= 本段最后一帧, PIL 格式)
            first_inpainted_frame = Image.fromarray(
                (seg_out[-1] * 255).astype(np.uint8))

            # ---------- (D) 去掉重叠帧并汇总 ----------
            if seg_idx > 0:      # 不是首段 → 丢掉第一帧
                seg_out        = seg_out[1:]
                pose_list      = pose_list[1:]
                agnostic_list  = agnostic_list[1:]
                seg_masks = seg_masks[1:]
                seg_videos = seg_videos[1:]

            final_outputs.extend(seg_out)
            final_pose.extend(pose_list)
            final_agnostic.extend(agnostic_list)
            final_mask.extend(seg_masks)
            final_video.extend(seg_videos)


            del pose_list, agnostic_list, new_mask_list, seg_out
            torch.cuda.empty_cache()
        # === 循环结束 ===

        # ----------- (E) 导出完整视频 -----------
        export_to_video(final_outputs, output_path, fps=fps)
        logger.info(f"Saved: {output_path}")

        # ----------- (F) debug 拼接 -----------
        if save_debug:
            debug_frames = []
            for i, out_frame in enumerate(final_outputs):
                debug_frames.append(concat_imgs_h(
                    refimg_list + [final_video[i], final_mask[i], final_agnostic[i], final_pose[i],
                                Image.fromarray((out_frame * 255).astype(np.uint8))]))
            export_to_video(debug_frames, debug_output_path, fps=fps)
            logger.info(f"Saved debug: {debug_output_path}")


        del video_list, mask_list
        torch.cuda.empty_cache()

# ------------------------------ CLI ------------------------------
def parse_args():
    os.makedirs("./log", exist_ok=True)
    p = argparse.ArgumentParser(description="Benchmark-based DreamSwapV inference with first-frame mask")
    p.add_argument("--bench_root", required=True, help="benchmark 路径，用于读取 video, first-frame mask 和 reference")
    p.add_argument("--checkpoint", required=True, help="DreamSwapV checkpoint 路径")

    p.add_argument("--output_dir", default="./outputs", help="输出目录")
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--guidance_scale", type=float, default=1.5)
    p.add_argument("--num_steps", type=int, default=50, help="推理步数")
    p.add_argument("--save_debug", action="store_true", help="保存 debug 横拼视频")
    p.add_argument("--total_length", type=int, default=69, help="生成视频的长度，置空则默认生成全长视频，DreamSwapV-benchmark 中使用69帧比较")
    p.add_argument("--frame_length", type=int, default=69, help="分段长度，默认为训练长度69，要求为4n+1，一般情况不建议调整")

    return p.parse_args()


# --------------------------- 入口：多进程切片 ---------------------------
if __name__ == "__main__":
    args = parse_args()
    try:
        mp.set_start_method("spawn", force=True)
    except Exception:
        pass

    all_tasks = build_tasks_from_benchmark(args.bench_root)
    if not all_tasks:
        raise RuntimeError("No tasks found in benchmark path!")

    num_gpu = int(os.environ.get("ARNOLD_WORKER_GPU", "1"))
    tasks_split = [
        (all_tasks[i::num_gpu], args.output_dir, args.checkpoint, args.guidance_scale,
         args.frame_length, args.seed, args.num_steps, 
         args.save_debug, args.total_length, i) 
        for i in range(num_gpu)
    ]

    processes = []
    for i in range(num_gpu):
        p = mp.Process(target=main, args=(tasks_split[i],))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
