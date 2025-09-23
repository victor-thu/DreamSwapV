import random
import torch
from torchvision import transforms
from PIL import Image

class CropToMultipleOf16(transforms.Lambda):
    def __init__(self):
        super().__init__(self.crop_image_to_multiple_of_16)
    
    def crop_image_to_multiple_of_16(self, image):
        """
        裁剪图像，确保裁剪后的图像宽度和高度都能被16整除。

        参数:
        image (PIL.Image): 输入的PIL图像。

        返回:
        PIL.Image: 裁剪后的图像。
        """
        # 获取图像的原始尺寸
        w, h = image.size
        
        # 计算裁剪后的宽度和高度
        new_w = w - (w % 16)  # 确保宽度能被16整除
        new_h = h - (h % 16)  # 确保高度能被16整除

        # 计算裁剪的起始坐标 (保证居中裁剪)
        x_start = (w - new_w) // 2
        y_start = (h - new_h) // 2

        # 裁剪图像
        cropped_image = image.crop((x_start, y_start, x_start + new_w, y_start + new_h))
        
        return cropped_image

class KeepratioAdaptiveResize(transforms.Lambda):
    '''保持长宽比后resize, resize后的图片一定大于等于目标尺寸, 确保可以从中crop出target尺寸的图'''
    def __init__(self, size, interpolation, ensure_divisible=False, divisible_value=16):
        super(KeepratioAdaptiveResize, self).__init__(self._resize)
        # Ensure output_size is a tuple
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.interpolation = interpolation
        self.ensure_divisible = ensure_divisible
        self.divisible_value = divisible_value
    
    def _resize(self, image):
        # 输入图片尺寸
        image_width, image_height = image.size
        aspect_ratio = image_width / image_height
        # 目标尺寸
        target_height, target_width = self.size
        target_aspect_ratio = target_width / target_height
        if aspect_ratio > target_aspect_ratio:
            new_height = int(target_height)
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = int(target_width)
            new_height = int(new_width / aspect_ratio)
        
        # 确保新宽度和高度可以被指定的值整除
        if self.ensure_divisible:
            new_width = (new_width + self.divisible_value - 1) // self.divisible_value * self.divisible_value
            new_height = (new_height + self.divisible_value - 1) // self.divisible_value * self.divisible_value

        return transforms.functional.resize(image, (new_height, new_width), self.interpolation)

class KeepratioAdaptiveResizeV2(transforms.Lambda):
    '''保持长宽比后resize, resize后的图片一定小于等于目标尺寸'''
    def __init__(self, size, interpolation, ensure_divisible=False, divisible_value=16):
        super(KeepratioAdaptiveResizeV2, self).__init__(self._resize)
        if isinstance(size, int):
            size = (size, size)
        self.size = size
        self.interpolation = interpolation
        self.ensure_divisible = ensure_divisible
        self.divisible_value = divisible_value
    
    def _resize(self, image):
        # 输入图片尺寸
        image_width, image_height = image.size
        aspect_ratio = image_width / image_height
        # 目标尺寸
        target_height, target_width = self.size
        target_aspect_ratio = target_width / target_height
        if aspect_ratio < target_aspect_ratio:
            new_height = int(target_height)
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = int(target_width)
            new_height = int(new_width / aspect_ratio)
        
        # 确保新宽度和高度可以被指定的值整除
        if self.ensure_divisible:
            new_width = (new_width + self.divisible_value - 1) // self.divisible_value * self.divisible_value
            new_height = (new_height + self.divisible_value - 1) // self.divisible_value * self.divisible_value

        return transforms.functional.resize(image, (new_height, new_width), self.interpolation)

class KeepratioRandomResizeCrop(transforms.Lambda):
    '''保持长宽的前提下，随机缩放并裁减出指定尺寸的图片，
       size:输出图片的尺寸(h,w)
       scale:缩放比例(scale_1,scale_2)，例如原图(500,1000),输出size为(384,512)
             首先原图会被缩放到(384*scale_1,768*scale_1)~(384*scale_2,768*scale_2)之间,再随机crop出(384,512)的图片
             这里scale必需都大于0
    '''
    def __init__(self, size, scale, height_offset, interpolation):
        super(KeepratioRandomResizeCrop, self).__init__(self._resize)
        # Ensure output_size is a tuple
        if isinstance(size, int):
            size = (size, size)
        if isinstance(scale, int):
            scale = (scale, scale)
        assert scale[0] >= 1 and scale[1] >= 1
        self.size = size
        self.scale = scale
        self.height_offset = height_offset
        self.interpolation = interpolation
    
    def _resize(self, image):
        
        # 输入图片尺寸
        image_width, image_height = image.size
        aspect_ratio = image_width / image_height
        # 目标尺寸
        target_height, target_width = self.size
        target_aspect_ratio = target_width / target_height
        if aspect_ratio > target_aspect_ratio:
            new_height = int(target_height)
            new_width = int(new_height * aspect_ratio)
        else:
            new_width = int(target_width)
            new_height = int(new_width / aspect_ratio)
        
        if self.scale[0] < self.scale[1]:
            scale = torch.Tensor(1).uniform_(self.scale[0], self.scale[1]).item()
        else:
            scale = self.scale[0]
        new_height = int(new_height * scale)
        new_width = int(new_width * scale)
        resized_image = transforms.functional.resize(image, (new_height, new_width), self.interpolation)

        if new_height == target_height and new_width == target_width:
            return resized_image

        if 0 < new_width - target_width:
            x1 = torch.randint(0, new_width - target_width, (1,)).item()
        else:
            x1 = new_width - target_width
        # 垂直方向增加offset，对于tt数据人物普遍出现在偏下方可能有用
        y_offset = int(self.height_offset * new_height)
        if  y_offset < new_height - target_height:
            y1 = torch.randint(y_offset, new_height - target_height, (1,)).item()
        else:
            y1 = new_height - target_height

        return transforms.functional.crop(resized_image, y1, x1, target_height, target_width)


class CenterBottomCrop(transforms.Lambda):
    def __init__(self, size):
        super(CenterBottomCrop, self).__init__(self._crop)
        # Ensure output_size is a tuple
        if isinstance(size, int):
            size = (size, size)
        self.size = size

    def _crop(self, image):
        # Calculate the area to be cropped
        image_width, image_height = image.size
        th, tw = self.size
        i = (image_height - th)  # start from the bottom
        j = (image_width - tw) // 2  # center horizontally
        # Crop the image
        return transforms.functional.crop(image, i, j, th, tw)


class CenterFractionalCrop(transforms.Lambda):
    def __init__(self, output_size):
        super(CenterFractionalCrop, self).__init__(self._crop)
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        self.output_size = output_size

    def _crop(self, image):
        image_width, image_height = image.size
        th, tw = self.output_size
        total_crop_height = image_height - th
        top_crop = total_crop_height * 3 // 4
        i = top_crop  # start from top_crop from top
        j = (image_width - tw) // 2  # center horizontally
        return transforms.functional.crop(image, i, j, th, tw)

class PaddingResize:
    def __init__(self, size, padding_color=(0, 0, 0)):
        """
        初始化 PaddingResize 类。

        参数:
        - size: 目标大小 (width, height)。
        - padding_color: 填充颜色，默认为黑色 (0, 0, 0)。
        """
        self.size = size
        self.padding_color = padding_color

    def __call__(self, img):
        """
        对图像进行填充和调整大小。

        参数:
        - img: 输入图像 (PIL Image)。

        返回:
        - 处理后的图像 (PIL Image)。
        """
        # 获取输入图像的宽度和高度
        width, height = img.size

        # 计算缩放比例
        scale = min(self.size[0] / width, self.size[1] / height)

        # 计算调整大小后的宽度和高度
        new_width = int(width * scale)
        new_height = int(height * scale)

        # 调整图像大小
        img = img.resize((new_width, new_height), Image.ANTIALIAS)

        # 创建新的图像并填充背景颜色
        new_img = Image.new("RGB", self.size, self.padding_color)

        # 计算填充位置
        top = (self.size[1] - new_height) // 2
        left = (self.size[0] - new_width) // 2

        # 将调整大小后的图像粘贴到新图像上
        new_img.paste(img, (left, top))

        return new_img

def resize_and_centercrop(image, height, width):
    trans_fn = transforms.Compose(
        [
            KeepratioAdaptiveResize((height, width), interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop((height, width)),
            
        ]
    )
    return trans_fn(image)

def resize_and_padding(image, height, width):
    trans_fn = transforms.Compose(
        [
            PaddingResize((height, width)),
            
        ]
    )
    return trans_fn(image)


def resize_and_center_bottom_crop(image, height, width):
    trans_fn = transforms.Compose(
        [
            KeepratioAdaptiveResize((height, width), interpolation=transforms.InterpolationMode.LANCZOS),
            CenterBottomCrop((height, width)),
        ]
    )
    return trans_fn(image)

def center_bottom_crop(image, height, width):
    trans_fn = transforms.Compose(
        [
            CenterBottomCrop((height, width)),
        ]
    )
    return trans_fn(image)

def random_resize(min_width, max_width, images):
    prop = 1.0 * images[0].size[0] / images[0].size[1]
    w = int(random.uniform(min_width, max_width)) 
    h = int(w/prop)
    for idx, img in enumerate(images):
        images[idx] = transforms.functional.resize(img, (h, w), transforms.InterpolationMode.LANCZOS)
    return images

def random_crop(tw, th, images):
    w, h = images[0].size

    x1 = random.randint(0, w - tw)
    y1 = random.randint(0, h - th)

    for idx, img in enumerate(images):
        images[idx] = img.crop((x1, y1, x1 + tw, y1 + th))
    return images

def color_jitter(*images):

    p1 = random.uniform(0.7, 1.3)
    p2 = random.uniform(0.7, 1.3)
    p3 = random.uniform(0.7, 1.3)
    p4 = random.uniform(-0.05, 0.05)
    
    parameters = (p1, p2, p3, p4)
    functions = (transforms.functional.adjust_brightness, 
                transforms.functional.adjust_contrast, 
                transforms.functional.adjust_saturation,
                transforms.functional.adjust_hue
            )
    
    images = list(images)
    for p, f in zip(parameters, functions):
        for idx, img in enumerate(images):
            images[idx] = f(img, p)

    return images

def weak_color_jitter(*images):

    p1 = random.uniform(0.9, 1.1)
    p2 = random.uniform(0.9, 1.1)
    p3 = random.uniform(0.9, 1.1)
    p4 = random.uniform(-0.01, 0.01)
    
    parameters = (p1, p2, p3, p4)
    functions = (transforms.functional.adjust_brightness, 
                transforms.functional.adjust_contrast, 
                transforms.functional.adjust_saturation,
                transforms.functional.adjust_hue
            )
    
    images = list(images)
    for p, f in zip(parameters, functions):
        for idx, img in enumerate(images):
            images[idx] = f(img, p)

    return images