import os
import numpy as np
from moviepy.editor import VideoFileClip, clips_array, ImageClip
from PIL import Image, ImageDraw, ImageFont

# ====== 配置区 ======
folder1    = r"/mnt/bn/lyl/wwt/wanx_video_pose/results/attntest_40000_khreverse_cfg2.0_test"
folder2    = r"/mnt/bn/lyl/wwt/wanx_video_pose/results/poseimgmixtest_40100_khreverse_cfg2.0_legacymask_test"
output_dir = r"/mnt/bn/lyl/wwt/wanx_video_pose/results/40000_main_pose_test"
# 三列的文字标签，顺序对应左→中→右
labels     = ["ref & ori", "mask & pose", "w/o & w/ pose"]
text_height = 50        # 文字行高度
font_path   = None      # 指定 .ttf 字体文件可提高美观
# ====================

def create_label_clip(text, size, duration, font_path=None, fontsize=40):
    """用 PIL 生成黑底白字的 ImageClip"""
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, fontsize) if font_path else ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    tw, th = draw.textsize(text, font=font)
    draw.text(((size[0]-tw)//2,(size[1]-th)//2), text, font=font, fill='white')
    return ImageClip(np.array(img)).set_duration(duration)

def process_two_folders_with_labels(f1, f2, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    vids1 = sorted([v for v in os.listdir(f1) if v.lower().endswith('.mp4')])
    vids2 = sorted([v for v in os.listdir(f2) if v.lower().endswith('.mp4')])
    n = min(len(vids1), len(vids2))

    for i in range(n):
        # 只处理 1-based 奇数 → 下标偶数
        if i % 2 != 1:
            continue

        p1 = os.path.join(f1, vids1[i])
        p2 = os.path.join(f2, vids2[i])
        c1 = VideoFileClip(p1)
        c2 = VideoFileClip(p2)
        w1, h1 = c1.size
        w2, h2 = c2.size
        dur = c1.duration  # 假设等长

        # —— 第一行（folder1） ——
        c11 = c1.crop(x1=0,       x2=w1/4,   y1=0, y2=h1).set_duration(dur)
        c12 = c1.crop(x1=2*w1/4,       x2=3*w1/4, y1=0, y2=h1).set_duration(dur)
        c13 = c1.crop(x1=3*w1/4,       x2=w1,     y1=0, y2=h1).set_duration(dur)

        # —— 第二行 ——
        c21 = c1.crop(x1=w1/4,       x2=2*w1/4,   y1=0, y2=h1).set_duration(dur)
        c22 = c2.crop(x1=2*w2/4,       x2=3*w2/4, y1=0, y2=h2).set_duration(dur)
        c23 = c2.crop(x1=3*w2/4,       x2=w2,     y1=0, y2=h2).set_duration(dur)

        # —— 第三行文字 ——  
        # 文字宽度应与上方列宽一致，我们这里用第一行的三块宽度：
        col_widths = [w1/4, w1/4, w1/4]
        label_clips = []
        for lbl, cw in zip(labels, col_widths):
            label_clips.append(
                create_label_clip(lbl, (int(cw), text_height), dur, font_path)
            )
        # clips_array 支持直接把一行 ImageClip 放到最下面：
        final = clips_array([
            [c11,    c12,    c13],
            [c21,    c22,    c23],
            label_clips
        ])

        name = os.path.splitext(vids1[i])[0]
        outp = os.path.join(out_dir, f"{name}_2x3_with_labels.mp4")
        final.write_videofile(outp, codec='libx264', audio_codec='aac')

        # 释放资源
        for clip in (c1, c2, c11, c12, c13, c21, c22, c23, final, *label_clips):
            clip.close()

if __name__ == '__main__':
    process_two_folders_with_labels(folder1, folder2, output_dir)
