import os
from utils import read_split_three_data,load_data

root = r"/root/autodl-tmp/MTARSI/airplane-datasets-new"

train_images_path, train_images_label, val_images_path, val_images_label = read_split_three_data(root, train_val_rate=0.9, train_rate=0.8)

train_text = r"./test/train.txt"
val_text = r"./test/val.txt"
test_text  =r"./test/test.txt"

train_img_paths, train_labels = load_data(train_text)
val_img_paths, val_labels = load_data(val_text)
test_img_paths, test_labels = load_data(test_text)

sum_dataset = len(train_img_paths) + len(val_img_paths) + len(test_img_paths)
print(f"数据集总数量：{sum_dataset}")
print(f"训练集数量：{len(train_img_paths)}, 占比：{len(train_img_paths) / sum_dataset}")
print(f"验证集数量：{len(val_img_paths)}, 占比：{len(val_img_paths) / sum_dataset}")
print(f"测试集数量：{len(test_img_paths)}, 占比：{len(test_img_paths) / sum_dataset}")