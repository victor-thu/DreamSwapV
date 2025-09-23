from PIL import Image

def add_gray_background(image_path):
    # 打开图片
    img = Image.open(image_path).convert("RGBA")  # 确保图片是RGBA格式
    width, height = img.size

    # 创建灰色背景
    gray_color = (128, 128, 128)
    background = Image.new("RGB", (width, height), gray_color)

    # 将原图合成到灰色背景上
    background.paste(img, mask=img.split()[3])  # 使用img的Alpha通道作为mask

    # 转换为RGB格式（去掉Alpha通道）
    result = background.convert("RGB")

    # 保存结果
    result.save(image_path, format="PNG")
    print(f"处理完成，结果已保存到 {image_path}")

# 示例调用
add_gray_background("/mnt/bn/lyl/wwt/sd_video_pose/test_images/potion.png")