
import cv2
import json
from pathlib import Path
from ultralytics import YOLO
from ultralytics.utils import LOGGER
import os
LOGGER.setLevel('ERROR')
from multiprocessing import Pool, Manager
from tqdm import tqdm
import torch
import time

VIDEO_DIR = '/mnt/bn/lyl/wwt/humanvid'  # 替换为你的视频文件夹路径
OUTPUT_JSON = '/mnt/bn/lyl/wwt/counted_data.json'
MODEL_NAME = 'yolov8n.pt'
CONFIDENCE_THRESHOLD = 0.4  # 提高置信度阈值
IOU_THRESHOLD = 0.3         # 加强NMS抑制
MIN_CONTINUOUS_FRAMES = 10  # 至少连续15帧检测到多人
MAX_TRACKING_DISTANCE = 50  # 像素级轨迹跟踪阈值

def process_video(args):
    video_path, gpu_idx = args
    try:
        # 设置GPU
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_idx)
        device = torch.device('cuda:0')
        model = YOLO(MODEL_NAME).to(device)
        
        cap = cv2.VideoCapture(str(video_path))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0
        continuous_count = 0
        tracking_boxes = {}
        max_people = 0
        
        # 视频预处理
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.resize(frame, (1024, 576))
            
            # 关键改进：多参数预测
            results = model.predict(
                frame,
                conf=CONFIDENCE_THRESHOLD,
                iou=IOU_THRESHOLD,
                classes=[0],     # 仅检测person类
                verbose=False,
                device=device
            )
            
            # 获取检测结果
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            
            # 轨迹跟踪与误检过滤
            current_people = 0
            current_boxes = []
            for box, conf in zip(boxes, confs):
                if conf < CONFIDENCE_THRESHOLD:
                    continue
                # 计算与跟踪框的重叠度
                matched = False
                for track_id, track_box in tracking_boxes.items():
                    if bbox_iou(box, track_box) > 0.5:
                        tracking_boxes[track_id] = box
                        matched = True
                        break
                if not matched:
                    tracking_boxes[len(tracking_boxes)] = box
                current_boxes.append(box)
            
            # 计算当前帧人数
            current_people = len(current_boxes)
            max_people = max(max_people, current_people)
            
            # 连续帧验证机制
            if current_people >= 2:
                continuous_count += 1
            else:
                continuous_count = 0
            
            # 提前终止条件
            if continuous_count >= MIN_CONTINUOUS_FRAMES:
                cap.release()
                return (video_path.name, max_people)
            
            frame_count += 1
        
        cap.release()
        
        # 综合判断逻辑
        if (continuous_count >= MIN_CONTINUOUS_FRAMES or 
            (max_people >= 2 and frame_count / total_frames > 0.2)):
            return (video_path.name, max_people)
        return None
    except Exception as e:
        print(f"Error on GPU {gpu_idx}: {str(e)}")
        return None

def bbox_iou(box1, box2):
    # 计算两个边界框的IoU
    x1, y1, x2, y2 = box1
    x3, y3, x4, y4 = box2
    inter_x1 = max(x1, x3)
    inter_y1 = max(y1, y3)
    inter_x2 = min(x2, x4)
    inter_y2 = min(y2, y4)
    
    if inter_x2 <= inter_x1 or inter_y2 <= inter_y1:
        return 0.0
    
    inter_area = (inter_x2 - inter_x1) * (inter_y2 - inter_y1)
    box1_area = (x2 - x1) * (y2 - y1)
    box2_area = (x4 - x3) * (y4 - y3)
    return inter_area / (box1_area + box2_area - inter_area)

def main():
    # 创建结果文件（如果不存在）
    if not os.path.exists(OUTPUT_JSON):
        with open(OUTPUT_JSON, 'w') as f:
            json.dump({}, f)
    
    # 加载已有结果避免重复处理
    with open(OUTPUT_JSON, 'r') as f:
        result_dict = json.load(f)
    
    video_paths = [vp for vp in Path(VIDEO_DIR).glob('*.*') if vp.name not in result_dict]
    print(f"待处理视频数量：{len(video_paths)}")
    
    if not video_paths:
        print("所有视频已处理完成！")
        return
    
    video_paths = list(Path(VIDEO_DIR).glob('*.*'))
    if not video_paths:
        print("No videos found!")
        return

    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUs")
    
    # 创建GPU索引循环分配
    gpu_indices = list(range(num_gpus)) * len(video_paths)
    video_gpu_pairs = list(zip(video_paths, gpu_indices[:len(video_paths)]))
    
    with Pool(processes=num_gpus) as pool, \
         tqdm(total=len(video_paths), desc="总进度", unit="视频") as pbar:
        
        for res in pool.imap_unordered(process_video, video_gpu_pairs):
            if res:
                # 原子操作更新文件
                with open(OUTPUT_JSON, 'r+') as f:
                    data = json.load(f)
                    data[res[0]] = res[1]
                    f.seek(0)
                    json.dump(data, f, indent=4)
                    f.truncate()
            pbar.update(1)

if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')
    torch.cuda.empty_cache()
    main()