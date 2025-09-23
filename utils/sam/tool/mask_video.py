import cv2
import numpy as np
import subprocess

def apply_mask_to_video(video_path, mask_path, output_path, gray_color=(128, 128, 128)):
    # 打开视频和掩码文件
    video_cap = cv2.VideoCapture(video_path)
    mask_cap = cv2.VideoCapture(mask_path)

    # 检查视频是否成功打开
    if not video_cap.isOpened() or not mask_cap.isOpened():
        print("无法打开视频或掩码文件，请检查路径是否正确！")
        return

    # 获取视频的基本信息
    width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = video_cap.get(cv2.CAP_PROP_FPS)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    # 创建输出视频文件
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    while True:
        # 逐帧读取视频和掩码
        ret_video, frame_video = video_cap.read()
        ret_mask, frame_mask = mask_cap.read()

        # 如果任意一个视频结束，则退出循环
        if not ret_video or not ret_mask:
            break

        # 确保帧大小一致
        if frame_video.shape != frame_mask.shape:
            print("视频和掩码的分辨率不一致，请检查输入文件！")
            break

        # 将掩码转换为灰度图
        gray_mask = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)

        # 找到掩码中白色区域（值接近 255）
        white_mask = (gray_mask > 200)  # 允许一定的容差

        # 将白色区域的像素替换为灰色
        frame_video[white_mask] = gray_color

        # 写入处理后的帧到输出视频
        out.write(frame_video)

    # 释放资源
    video_cap.release()
    mask_cap.release()
    out.release()
    print(f"处理完成，结果已保存到 {output_path}")
# 示例用法
if __name__ == "__main__":
    video_path = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/mask_videos/5837589_5837589-hd_1920_1080_24fps_video.mp4"  # 输入的原始视频文件
    mask_path = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/mask_videos/5837589_Basketball.mp4"  # 输入的 mask 视频文件
    output_path = "/mnt/bn/lyl/wwt/Segment-and-Track-Anything/mask_videos/5837589_Basketball_agnostic.mp4"  # 输出的处理后视频文件

    apply_mask_to_video(video_path, mask_path, output_path)