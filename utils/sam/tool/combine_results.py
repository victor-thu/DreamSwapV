import os
import numpy as np
from moviepy.editor import VideoFileClip, clips_array, ImageClip
from PIL import Image, ImageDraw, ImageFont

# ====== 配置区域，请根据需求修改 ======
folders = [
    r"/mnt/bn/lyl/wwt/wanx_video_pose/results/attntest_40000_khreverse_cfg2.0",
    r"/mnt/bn/lyl/wwt/wanx_video_pose/results/attntest_40000_khreverse_cfg2.0_imginpaint",
    r"/mnt/bn/lyl/wwt/wanx_video_pose/results/attntest_40100_khreverse_cfg2.0_imagemix",
    r"/mnt/bn/lyl/wwt/wanx_video_pose/results/attntest_40100_khreverse_cfg2.0_maskaug",
    r"/mnt/bn/lyl/wwt/wanx_video_pose/results/attntest_40100_khreverse_cfg2.0_refcrop0.7",
]
output_dir = r"/mnt/bn/lyl/wwt/wanx_video_pose/results/40000_main_imginpaint_imagemix_maskaug_refcrop"
labels = ["img_inpainting", "img_mix", "mask_dilate", "ref_aug"]  # 对应第二行四个视频的标签
text_height = 50  # 下方文字行高度
font_path = None
# ===================================

def create_label_clip(text, size, duration, font_path=None, fontsize=60):
    """
    使用 PIL 创建指定尺寸的图片，再转为 ImageClip，其中 fontsize 可调整文字大小。
    """
    img = Image.new('RGB', size, color='black')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(font_path, fontsize) if font_path else ImageFont.load_default()
    except Exception:
        font = ImageFont.load_default()

    text_w, text_h = draw.textsize(text, font=font)
    x = (size[0] - text_w) // 2
    y = (size[1] - text_h) // 2
    draw.text((x, y), text, font=font, fill='white')

    frame = np.array(img)
    clip = ImageClip(frame).set_duration(duration)
    return clip


def process_videos(folders, output_dir):
    """
    对五个文件夹中的奇数序号视频（1-based 奇数，即 0-based 偶数）进行：
    1. 第一行：folder1 的全视频
    2. 第二行：folder2-5 视频的右 1/4 区域，裁剪后水平拼接
    3. 第三行：文字标签行，对应 labels 列表
    最终将三行垂直拼接，输出高度为 2*h + text_height、宽度为 w 的新视频。
    """
    os.makedirs(output_dir, exist_ok=True)

    all_files = [[f for f in os.listdir(folder) if f.lower().endswith('.mp4')] for folder in folders]
    count = len(all_files[0])

    for idx in range(count):
        if idx % 2 != 1:
            continue

        paths = [os.path.join(folders[i], all_files[i][idx]) for i in range(5)]

        clip1 = VideoFileClip(paths[0])
        w, h = clip1.size
        duration = clip1.duration
        quarter = int(w / 4)

        # 第二行裁剪与拼接
        quarter_clips = []
        for p in paths[1:]:
            c = VideoFileClip(p).crop(x1=3*quarter, x2=w, y1=0, y2=h).set_duration(duration)
            quarter_clips.append(c)
        row2 = clips_array([quarter_clips])

        # 第三行文字标签
        label_clips = []
        for lbl in labels:
            clip = create_label_clip(lbl, (quarter, text_height), duration, font_path)
            label_clips.append(clip)
        row3 = clips_array([label_clips])

        # 三行垂直组合
        final = clips_array([[clip1], [row2], [row3]])

        base = os.path.splitext(os.path.basename(paths[0]))[0]
        out_path = os.path.join(output_dir, f"{base}_combined.mp4")
        final.write_videofile(out_path, codec='libx264', audio_codec='aac')

        final.close()
        clip1.close()
        row2.close()
        row3.close()
        for c in quarter_clips + label_clips:
            c.close()

if __name__ == '__main__':
    process_videos(folders, output_dir)