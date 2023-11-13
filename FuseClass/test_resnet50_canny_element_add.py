import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import math

from tqdm.auto import tqdm as tqdm_auto
from my_dataset_new import MyDataSet
from diffusers.optimization import get_cosine_schedule_with_warmup, get_linear_schedule_with_warmup
from models.feature_extract_model_easy_method_element_add import extractmodel
from dataclasses import dataclass
from torchvision import datasets
from utils import plt_result, read_split_three_data, load_data

import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

@dataclass
class test_cfg:
    # data
    img_size = 224
    root = r"/root/autodl-tmp/MTARSI/airplane-datasets-new"
    # 在resume中有checkpoints保存的epoch，model_weights，optimizer
    resume = ""
    weight_path = r"/root/autodl-tmp/weights/weights_resnet50_element_add_gate_unit/epoch_97.pth"
    dataset_type = ""
    # 类别要与数据集一致
    num_classes = 20
    # model
    model_type = "resnet50"
    include_top = True
    easy_method = "gate-unit"
    DROPOUT = 0.5
    # dataloader
    batch_size = 160
    val_batch_size = 160
    shuffle = False
    num_workers = 8
    # optim
    # 经过实验发现, 对于16batch, lr设置为8e-6比较合适, 保险起见, 32batch设为5e-6
    lr = 1e-2
    # train
    start_epoch = 0
    epochs = 100
    lr_warmup_steps = 1500
    # eval
    bs = 160
    in_channels = 1
    # save
    save_path = r""

def train(local_rank):

    if local_rank == 0:
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

        if os.path.exists("./logs") is False:
            os.makedirs("./logs")

        if os.path.exists("./data") is False:
            os.makedirs("./data")

    cfg = test_cfg()

    logging.info(f'''\nthe configuration of training:
                 dataset: imgsize={cfg.img_size}, datasetroot={cfg.root}, resume={cfg.resume}, weight_path={cfg.weight_path}, dataset_type={cfg.dataset_type}, num_class={cfg.num_classes}.
                 model: model_type={cfg.model_type}, include_top={cfg.include_top}, easy_method={cfg.easy_method}, DROPOUT={cfg.DROPOUT}.
                 dataloader: batchsize={cfg.batch_size}, val_batch_size={cfg.val_batch_size}, shuffle={cfg.shuffle}, num_workers={cfg.num_workers}.
                 optim: lr={cfg.lr}.
                 train: start_epoch={cfg.start_epoch}, epochs={cfg.epochs}, lr_warmup_steps={cfg.lr_warmup_steps}.
                 eval: bs={cfg.bs}, in_channels={cfg.in_channels}.
                 save: save_path={cfg.save_path}''')

    test_transform = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    if cfg.dataset_type == "":
        assert cfg.root != "", "please input a dataset root"

        # train_images_path, train_images_label, val_images_path, val_images_label = read_split_three_data(cfg.root)

        test_path = r"./test/test.txt"
        test_images_path, test_images_label = load_data(test_path)

        import json
        json_str = r"class_indices.json"
        with open(json_str,'r') as load_f:
            all_class = json.load(load_f)
        num_class = len(all_class)
        assert int(cfg.num_classes) == int(num_class), f"数据集的类别数量{num_class}与cfg中的类别数量{cfg.num_classes}不一样"

        test_dataset = MyDataSet(test_images_path, test_images_label, test_transform)


    # dataloader
    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                   batch_size=cfg.bs,
                                                   num_workers=cfg.num_workers,
                                                   shuffle=cfg.shuffle,
                                                   )

    # model
    model = extractmodel(cfg=cfg)
    # print(f"\nmodel = {model}\n")

    #  装载权重
    if cfg.weight_path != "":
        logging.warning(f"model loading from {cfg.weight_path} on cpu")
        checkpoint = torch.load(cfg.weight_path, map_location=torch.device("cpu"))  # 可以是cpu,cuda,cuda:index
        model.load_state_dict(checkpoint['model_state_dict'])

    # 将模型装入单卡GPU上
    model = model.cuda(local_rank)

    # loss
    criterion = nn.CrossEntropyLoss()

    # 查看model使用的设备
    model_device = next(model.parameters()).device
    print(f"model_device {model_device}\n")

    with tqdm_auto(range(len(test_dataloader))) as pbar:
        model = model.eval()
        loss = 0
        sample_num = 0
        accu_num = torch.zeros(1).cuda(local_rank)
        for test_step, batch in zip(pbar, test_dataloader):
            with torch.no_grad():
                img, label, canny = batch
                sample_num += img.shape[0]

                img = img.cuda(local_rank)
                label = label.cuda(local_rank)
                canny = canny.cuda(local_rank)

                pred = model(img, canny)

                pred_cls = torch.max(pred, dim=1)[1]

                accu_num += torch.eq(pred_cls, label).sum()

                loss_value = criterion(pred, label)

                pbar.update(1)
                loss += loss_value.item()

                logs = {"rank": local_rank, "TestLoss": loss / (test_step + 1), "accu":accu_num.item() / sample_num}
                # str_log = "ValLoss: {}, Epoch: {} Lr: {}, accu: {}".format(loss / (global_step + 1), epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
                # logging.info(str_log)
                pbar.set_postfix(**logs)

                test_step += 1

        pbar.close()
        loss = loss / (test_step)

        # str_log = "rank: {} TrainLoss: {}, Epoch: {} Lr: {}, accu: {}".format(local_rank, loss, epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
        # logging.info(str_log)

        # 只在主进程执行记录日志和更新进度条
        str_log = "rank: {} TestLoss: {}, accu: {}".format(local_rank, loss, accu_num.item() / sample_num)
        logging.info(str_log)

# init seed
def init_seeds(seed=0, cuda_deterministic=False):

    np.random.seed(seed)
    torch.manual_seed(seed)

    if cuda_deterministic:  # slower, more reproducible
         torch.backends.cudnn.deterministic = True
         torch.backends.cudnn.benchmark = False
    else:  # faster, less reproducible
         torch.backends.cudnn.deterministic = False
         torch.backends.cudnn.benchmark = True

if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0, help="local device id on current node")
    args = parser.parse_args()

    print(args.local_rank)

    if args.local_rank == 0:
        # Get current time
        current_time = datetime.now()

        # Format time
        formatted_time = current_time.strftime('%Y-%m-%d_%H-%M-%S')

        logfile = "./logs/" + "test_" + str(formatted_time) +".log"
        print(logfile)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
            handlers=[logging.FileHandler(logfile, mode='w'),
                                    logging.StreamHandler()]
        )

    if args.local_rank == 0:
        if torch.cuda.is_available():
            logging.warning("Cuda is available!")
            if torch.cuda.device_count() > 1:
                logging.warning(f"Find {torch.cuda.device_count()} GPUs!")
            else:
                logging.warning("Too few GPU!")
        else:
            logging.warning("Cuda is not available! Exit!")

    # 设置随机种子，保证试验的可复现性
    seed = 3047
    init_seeds(seed+args.local_rank, cuda_deterministic=False)

    train(args.local_rank)