from PIL import Image
import torch
from torch.utils.data import Dataset
from traditional_features import *
import matplotlib.pyplot as plt

class MyDataSet(Dataset):
    """自定义数据集"""

    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item]).convert('RGB')

        # RGB为彩色图片，L为灰度图片
        if img.mode != 'RGB':
            raise ValueError("image: {} isn't RGB mode.".format(self.images_path[item]))
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        c, h, w = img.shape
        rgb_img = img.clone()

        # img 是一个归一化的图像，其形状为 (C, H, W)
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # 对 img 执行逆归一化操作
        rgb_img = ((rgb_img * std) + mean) * 255

        rgb_img = rgb_img.permute(1, 2, 0).numpy()
        # BGR2RGB
        rgb_img = rgb_img[:, :, ::-1]
        rgb_img = rgb_img.astype(np.uint8)

        # path_save = r"E:\workspace\PycharmProjects\FeaturesFuseProject\FuseClass\test\\"+str(item)+".jpg"
        # cv2.imwrite(path_save, rgb_img)

        # 提取canny特征
        cannymap = calculate_canny(rgb_img)
        h1, w1 = cannymap.shape
        cannymap = torch.tensor(cannymap, dtype=torch.float32).view(1, h1, w1).contiguous()

        # 提取hog特征
        # hog_features = calculate_hog(rgb_img)
        # print(hog_features, hog_features.shape)

        return img, label, cannymap

    @staticmethod
    def collate_fn(batch):
        # 官方实现的default_collate可以参考
        # https://github.com/pytorch/pytorch/blob/67b7e751e6b5931a9f45274653f4f653a4e6cdf6/torch/utils/data/_utils/collate.py
        # zip()会把图片和图片放在一起，标签和标签放在一起
        images, labels, cannymaps = tuple(zip(*batch))
        # 将这批图片堆叠起来
        images = torch.stack(images, dim=0)
        # 转换为tensor格式
        labels = torch.as_tensor(labels)
        # 将canny特征堆叠起来
        cannymaps = torch.stack(cannymaps, dim=0)
        # 将hog特征堆叠起来
        # hog_features = torch.stack(hog_features, dim=0)
        return images, labels, cannymaps
