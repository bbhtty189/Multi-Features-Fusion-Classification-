from utils import read_split_three_data

if __name__ == "__main__":
    root = r"/root/autodl-tmp/MTARSI/airplane-datasets-new"
    train_val_rate = 0.8
    train_rate = 0.75
    train_images_path, train_images_label, val_images_path, val_images_label = read_split_three_data(root, train_val_rate, train_rate)
