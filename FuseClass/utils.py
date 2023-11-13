import os
import sys
import json
import pickle
import random

import torch
from tqdm import tqdm
import torch.nn.functional as F

import matplotlib.pyplot as plt

def read_data(root: str):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)
    # 遍历文件夹，一个文件夹对应一个类别
    img_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    img_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(img_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)
    
    images_path = []
    images_label = []
    every_class_num = []  # 存储每个类别的样本总数
    for cla in img_class:
        cla_path = os.path.join(root, cla)
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        for img_path in images:
            images_path.append(img_path)
            images_label.append(image_class)
    
    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training/vailding.".format(len(images_path)))

    return images_path, images_label

def read_split_data(root: str, val_rate: float = 0.2):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    # 遍历文件夹，一个文件夹对应一个类别
    airplane_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    airplane_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(airplane_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    for cla in airplane_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))
        # 按比例随机采样验证样本
        val_path = random.sample(images, k=int(len(images) * val_rate))

        for img_path in images:
            if img_path in val_path:  # 如果该路径在采样的验证集样本中则存入验证集
                val_images_path.append(img_path)
                val_images_label.append(image_class)
            else:  # 否则存入训练集
                train_images_path.append(img_path)
                train_images_label.append(image_class)

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(airplane_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(airplane_class)), airplane_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def read_split_three_data(root: str, train_val_rate: float = 0.8, train_rate: float = 0.75):
    random.seed(0)  # 保证随机结果可复现
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    test_path = r"./test"
    if os.path.exists(test_path) is False:
        os.makedirs(test_path)
    # 存放训练，验证，测试数据
    f_train = open(os.path.join(test_path, 'train.txt'), 'w')
    f_val = open(os.path.join(test_path, 'val.txt'), 'w')
    f_test = open(os.path.join(test_path, 'test.txt'), 'w')

    # 遍历文件夹，一个文件夹对应一个类别
    airplane_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]
    # 排序，保证顺序一致
    airplane_class.sort()
    # 生成类别名称以及对应的数字索引
    class_indices = dict((k, v) for v, k in enumerate(airplane_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []  # 存储训练集的所有图片路径
    train_images_label = []  # 存储训练集图片对应索引信息
    val_images_path = []  # 存储验证集的所有图片路径
    val_images_label = []  # 存储验证集图片对应索引信息
    test_images_path = []
    test_images_label = []
    every_class_num = []  # 存储每个类别的样本总数
    supported = [".jpg", ".JPG", ".png", ".PNG"]  # 支持的文件后缀类型
    # 遍历每个文件夹下的文件
    # 按每一个类别划分数据，分别放入训练集，验证集，测试集
    for cla in airplane_class:
        cla_path = os.path.join(root, cla)
        # 遍历获取supported支持的所有文件路径
        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]
        # 获取该类别对应的索引
        image_class = class_indices[cla]
        # 记录该类别的样本数量
        every_class_num.append(len(images))

        # 先获得训练验证集
        train_val_path = random.sample(images, k=int(len(images) * train_val_rate))
        # 不在训练验证集中的样本放入测试集
        for img_path in images:
            if img_path not in train_val_path:
                test_images_path.append(img_path)
                test_images_label.append(image_class)
                in_str = img_path + ", " + str(image_class)
                f_test.write(in_str + '\n')

        train_path = random.sample(train_val_path, int(len(train_val_path) * train_rate))
        for img_path in train_val_path:
            if img_path in train_path:  # 如果该路径在采样的验证集样本中则存入验证集
                train_images_path.append(img_path)
                train_images_label.append(image_class)
                train_str = img_path + ', ' + str(image_class)
                f_train.write(train_str + '\n')
            else:
                val_images_path.append(img_path)
                val_images_label.append(image_class)
                val_str = img_path + ', ' + str(image_class)
                f_val.write(val_str + '\n')

    f_test.close()

    print("{} images were found in the dataset.".format(sum(every_class_num)))
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))
    print("{} images for test.".format(len(test_images_path)))

    plot_image = False
    if plot_image:
        # 绘制每种类别个数柱状图
        plt.bar(range(len(airplane_class)), every_class_num, align='center')
        # 将横坐标0,1,2,3,4替换为相应的类别名称
        plt.xticks(range(len(airplane_class)), airplane_class)
        # 在柱状图上添加数值标签
        for i, v in enumerate(every_class_num):
            plt.text(x=i, y=v + 5, s=str(v), ha='center')
        # 设置x坐标
        plt.xlabel('image class')
        # 设置y坐标
        plt.ylabel('number of images')
        # 设置柱状图的标题
        plt.title('flower class distribution')
        plt.show()

    return train_images_path, train_images_label, val_images_path, val_images_label

def load_data(text):
    with open(text, 'r') as f:
        dataset = f.readlines()

    img_paths = []
    labels = []
    for data in dataset:
        data = data.split(', ')
        img_paths.append(data[0])
        labels.append(int(data[1]))

    return img_paths, labels

def plot_data_loader_image(data_loader):
    batch_size = data_loader.batch_size
    plot_num = min(batch_size, 4)

    json_path = './class_indices.json'
    assert os.path.exists(json_path), json_path + " does not exist."
    json_file = open(json_path, 'r')
    class_indices = json.load(json_file)

    for data in data_loader:
        images, labels = data
        for i in range(plot_num):
            # [C, H, W] -> [H, W, C]
            img = images[i].numpy().transpose(1, 2, 0)
            # 反Normalize操作
            img = (img * [0.229, 0.224, 0.225] + [0.485, 0.456, 0.406]) * 255
            label = labels[i].item()
            plt.subplot(1, plot_num, i+1)
            plt.xlabel(class_indices[str(label)])
            plt.xticks([])  # 去掉x轴的刻度
            plt.yticks([])  # 去掉y轴的刻度
            plt.imshow(img.astype('uint8'))
        plt.show()


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list

def plt_result(x, y, title):

    plt.xlabel("epochs")
    plt.ylabel(title)

    plt.plot(x, y, color='blue')

    plt.savefig(os.path.join("./logs/line", title + ".jpg"))
    # plt.show()

# 读取测试集
def load_test_data(test_path):
    with open(test_path, "r") as f:
        items = f.readlines()

    test_images_path = []
    test_images_label = []
    for item in items:
        data = item.split()
        test_images_path.append(data[0])
        test_images_label.append(int(data[1]))

    return test_images_path, test_images_label


def plot_bar(x, y, title):
    import matplotlib.pyplot as plt

    # 设置柱状图的位置和宽度
    bar_width = 0.3

    # 创建柱状图
    plt.bar(x, y, width=bar_width, edgecolor='k', alpha=0.7)

    # 设置标题和坐标轴标签
    plt.title(title)
    plt.xlabel('Category')
    plt.ylabel('Value')

    # 显示图形
    plt.savefig(os.path.join("./logs/bar", title + ".jpg"))
    plt.show()

if __name__ == "__main__":
    import numpy as np
    x = np.array(list(range(10)))
    y = np.sin(x)
    print(x)
    print(y)
    plt.plot(x, y)
    plt.show()

    # plt_result(x, y, "sin")

