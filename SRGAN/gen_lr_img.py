import sys
import os
from PIL import Image

from fileinput import filename


import os
from PIL import Image

class gen_lr_img:
    # 将当前文件夹下所有图片移动到 'hr' 文件夹下，并新建 'lr' 文件夹，利用 'hr' 中的高分辨图片生成相应的低分辨图片并保存到 'lr' 文件夹下
    def __init__(self, dir_path):
        self.dir_path = dir_path      # 文件夹路径
        self.image_extensions = ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG']
        os.makedirs(os.path.join(self.dir_path, 'hr', 'hr'), exist_ok=True)
        os.makedirs(os.path.join(self.dir_path, 'lr', 'lr'), exist_ok=True)

    def is_any_img(self):
        # 判断当前一级目录下是否有图片存在
        for filename in os.listdir(self.dir_path):
            if any(filename.endswith(ext) for ext in self.image_extensions):
                return True
        return False

    def __move_img__(self):
        # 如果当前一级目录存在图片，则将其都移动到 hr 文件夹下
        if self.is_any_img():
            for filename in os.listdir(self.dir_path):
                if any(filename.endswith(ext) for ext in self.image_extensions):
                    os.rename(os.path.join(self.dir_path, filename), os.path.join(self.dir_path, 'hr', 'hr', filename))

    def __gen_lr_img__(self, upscale_factor):
        # 利用 'hr' 中的高分辨图片生成相应的低分辨图片并保存到 'lr' 文件夹下
        for filename in os.listdir(os.path.join(self.dir_path, 'hr', 'hr')):
            if any(filename.endswith(ext) for ext in self.image_extensions):
                # 打开图片
                img = Image.open(os.path.join(self.dir_path, 'hr', 'hr', filename))

                # 获取图片尺寸
                width, height = img.size

                # 计算新图片尺寸
                new_width = width // upscale_factor
                new_height = height // upscale_factor

                # 使用双三次插值法进行图片缩放
                lr_img = img.resize((new_width, new_height), Image.BICUBIC)

                # 保存图片
                lr_img.save(os.path.join(self.dir_path, 'lr', 'lr', filename))

        print('低分辨图片生成成功！')
        print('低分辨图片保存在：', os.path.join(self.dir_path, 'lr', 'lr'))


if  __name__ == '__main__':
    dir_path = '/home/cxmd/文档/data_for_AI_train/VOC2012/train/'
    upscale_factor = 4
    gen_lr_img(dir_path).__move_img__()
    gen_lr_img(dir_path).__gen_lr_img__(upscale_factor)

