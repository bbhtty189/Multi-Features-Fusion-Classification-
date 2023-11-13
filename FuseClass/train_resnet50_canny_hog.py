import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import numpy as np
import math

from tqdm.auto import tqdm as tqdm_auto
from my_dataset_new import MyDataSet
from diffusers.optimization import get_cosine_schedule_with_warmup
from models.resnet_origin import resnet50
from dataclasses import dataclass
from torchvision import datasets
from utils import plt_result, read_split_three_data

import os
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

@dataclass
class train_cfg:
    # data
    img_size = 224
    root = r"E:\datasets\MTARSI\airplane-datasets"
    # 在resume中有checkpoints保存的epoch，model_weights，optimizer
    resume = ""
    weight_path = r"./weights/resnet50.pth"
    dataset_type = ""
    # 类别要与数据集一致
    num_classes = 33
    # model
    include_top = True
    DROPOUT = 0.5
    # dataloader
    batch_size = 1
    val_batch_size = 192
    shuffle = True
    num_workers = 8
    # optim
    # 经过实验发现, 对于16batch, lr设置为8e-6比较合适, 保险起见, 32batch设为5e-6
    lr = 1e-4
    # train
    start_epoch = 0
    epochs = 100
    lr_warmup_steps = 1500
    # eval
    bs = 1
    in_channels = 1

def train(local_rank):

    if local_rank == 0:
        if os.path.exists("./weights") is False:
            os.makedirs("./weights")

        if os.path.exists("./logs") is False:
            os.makedirs("./logs")

        if os.path.exists("./data") is False:
            os.makedirs("./data")

    cfg = train_cfg()

    train_transform = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    val_transform = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    train_dataset = None
    if cfg.dataset_type == "MNIST":
        train_dataset = datasets.MNIST("./data", train=True, download=False, transform=train_transform)
        val_dataset = datasets.MNIST("./data", train=False, download=False, transform=val_transform)
    elif cfg.dataset_type == "CIFAR-10":
        train_dataset = datasets.CIFAR10("./data", train=True, download=False, transform=train_transform)
        val_dataset = datasets.CIFAR10("./data", train=False, download=False, transform=val_transform)
    elif cfg.dataset_type == "CIFAR-100":
        train_dataset = datasets.CIFAR100("/root/autodl-tmp/CIFAR-100", train=True, download=False, transform=train_transform)
        val_dataset = datasets.CIFAR100("/root/autodl-tmp/CIFAR-100", train=False, download=False, transform=val_transform)
    elif cfg.dataset_type == "":
        assert cfg.root != "", "please input a dataset root"

        train_images_path, train_images_label, val_images_path, val_images_label = read_split_three_data(cfg.root)

        import json
        json_str = r"class_indices.json"
        with open(json_str,'r') as load_f:
            all_class = json.load(load_f)
        num_class = len(all_class)
        assert int(cfg.num_classes) == int(num_class), f"数据集的类别数量{num_class}与cfg中的类别数量{cfg.num_classes}不一样"

        train_dataset = MyDataSet(train_images_path, train_images_label, train_transform)
        val_dataset = MyDataSet(val_images_path, val_images_label, val_transform)


    # dataloader
    train_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=cfg.batch_size,
                                                   num_workers=cfg.num_workers,
                                                   shuffle=cfg.shuffle,
                                                   )

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=cfg.val_batch_size,
                                                 num_workers=cfg.num_workers,
                                                 shuffle=cfg.shuffle,
                                                 )

    # model
    model = resnet50(num_classes=cfg.num_classes, include_top=True)
    # print(f"\nmodel = {model}\n")

    # 使用resnet的预训练权重
    if cfg.weight_path != "":
        logging.warning(f"model.res loading from {cfg.weight_path} on cpu")
        layers_False = ["fc.weight", "fc.bias"]
        
        model_dict = model.state_dict()

        weights_dict = torch.load(cfg.weight_path, map_location=torch.device("cpu"))
        # 删除有关分类类别的权重
        for k in list(weights_dict.keys()):
            if k in layers_False:
                if np.shape(model_dict[k]) == np.shape(weights_dict[k]):
                    continue
                else:
                    del weights_dict[k]
        print(model.load_state_dict(weights_dict, strict=False))

    #  断点处重启训练
    if cfg.resume != "":
        logging.warning(f"model loading from {cfg.resume} on cpu")
        checkpoint = torch.load(cfg.resume, map_location=torch.device("cpu"))  # 可以是cpu,cuda,cuda:index
        model.load_state_dict(checkpoint['model_state_dict'])
        # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # cfg.start_epoch = checkpoint['epoch'] + 1

    # 将模型装入单卡GPU上
    model = model.cuda(local_rank)

    # optimizer
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)

    #  断点处重启训练
    if cfg.resume != "":
        logging.warning(f"optimizer & start_epoch loading from {cfg.resume} on {local_rank}")
        checkpoint = torch.load(cfg.resume, map_location=torch.device(local_rank))  # 可以是cpu,cuda,cuda:index
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        cfg.start_epoch = checkpoint['epoch'] + 1


    # loss
    criterion = nn.CrossEntropyLoss()

    # 查看model和optimizer使用的设备
    model_device = next(model.parameters()).device
    optimizer_device = optimizer.param_groups[0]['params'][0].device
    print(f"model_device {model_device}, optimizer_device {optimizer_device} \n")


    # 学习率调整策略，warmup的轮次
    cfg.lr_warmup_steps = int(len(train_dataloader) * 10)

    # 使用cosine策略时, 轮次要长一些，否则学习率增长太快导致梯度发散
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer=optimizer,
        num_warmup_steps=cfg.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * cfg.epochs),
    )

     #  断点处重启训练
    if cfg.resume != "":
        logging.warning(f"lr_scheduler loading from {cfg.resume} on {local_rank}")
        checkpoint = torch.load(cfg.resume, map_location=torch.device(local_rank))  # 可以是cpu,cuda,cuda:index
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        lr_scheduler.step()  # 立即更新学习率

    if local_rank == 0:
        # tensorboard
        experiment_name = "runs/" + str(formatted_time) + "CIFAR Experiment"
        writer = SummaryWriter(experiment_name)
    


    train_loss_list = []
    val_loss_list = []
    train_accu_list = []
    val_accu_list = []
    for epoch in range(cfg.start_epoch, cfg.epochs):

        with tqdm_auto(range(len(train_dataloader))) as pbar:
            model = model.train()
            loss = 0
            sample_num = 0
            accu_num = torch.zeros(1).cuda(local_rank)
            for global_step, batch in zip(pbar, train_dataloader):


                img, label, canny, hog_features = batch
                sample_num += img.shape[0]

                img = img.cuda(local_rank)
                label = label.cuda(local_rank)
                # print(label, label.shape)

                pred = model(img)

                pred_cls = torch.max(pred, dim=1)[1]
                accu_num += torch.eq(pred_cls, label).sum()

                loss_value = criterion(pred, label)

                for name, param in model.named_parameters():
                    if param.device == "cpu":
                        print(name, param.device)

                loss_value.backward()

                for name, param in model.named_parameters():
                    if param.device == "cpu":
                        if param.grad is not None:
                            print(f"{name} gradient is on {param.grad.device}")
                        else:
                            print(f"{name} has no gradient yet")
                
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param.device == "cpu":
                            print(f"Parameter is on {param.device}")

                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                torch.cuda.synchronize()
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

                pbar.update(1)
                loss += loss_value.item()

                logs = {"rank": local_rank, "TrainLoss": loss / (global_step + 1), "Epoch": epoch, "Lr": lr_scheduler.get_last_lr()[0], "accu":accu_num.item() / sample_num}
                # str_log = "TrainLoss: {}, Epoch: {} Lr: {}, accu: {}".format(loss / (global_step + 1), epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
                # logging.info(str_log)
                pbar.set_postfix(**logs)

                # 只在主进程执行记录日志和更新进度条
                writer.add_scalar('train loss', loss, epoch * len(train_dataloader) + global_step)
                writer.add_scalar('train accuracy', accu_num.item() / sample_num, epoch * len(train_dataloader) + global_step)

                global_step += 1

            pbar.close()
            loss = loss / (global_step)

            # # 打印python数据类型，而不是tensor，方式是加.item()
            # str_log = "rank: {} TrainLoss: {}, Epoch: {} Lr: {}, accu: {}".format(local_rank, loss, epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
            # logging.info(str_log)

            # 只在主进程执行记录日志和更新进度条
            # 打印python数据类型，而不是tensor，方式是加.item()
            str_log = "rank: {} TrainLoss: {}, Epoch: {} Lr: {}, accu: {}".format(local_rank, loss, epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
            logging.info(str_log)
            train_accu_list.append(math.floor((accu_num.item() / sample_num) * 1000) / 1000)
            train_loss_list.append(math.floor(loss * 1000) / 1000)

        with tqdm_auto(range(len(val_dataloader))) as pbar:
            model = model.eval()
            loss = 0
            sample_num = 0
            accu_num = torch.zeros(1).cuda(local_rank)
            for val_step, batch in zip(pbar, val_dataloader):
                with torch.no_grad():
                    img, label = batch
                    sample_num += img.shape[0]

                    img = img.cuda(local_rank)
                    label = label.cuda(local_rank)

                    pred = model(img)

                    pred_cls = torch.max(pred, dim=1)[1]

                    accu_num += torch.eq(pred_cls, label).sum()

                    loss_value = criterion(pred, label)

                    pbar.update(1)
                    loss += loss_value.item()

                    logs = {"rank": local_rank, "ValLoss": loss / (val_step + 1), "Epoch": epoch, "Lr": lr_scheduler.get_last_lr()[0], "accu":accu_num.item() / sample_num}
                    # str_log = "ValLoss: {}, Epoch: {} Lr: {}, accu: {}".format(loss / (global_step + 1), epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
                    # logging.info(str_log)
                    pbar.set_postfix(**logs)

                    # 只在主进程执行记录日志和更新进度条
                    writer.add_scalar('valid loss', loss, epoch * len(val_dataloader) + global_step)
                    writer.add_scalar('valid accuracy', accu_num.item() / sample_num, epoch * len(val_dataloader) + global_step)

                    val_step += 1

            pbar.close()
            loss = loss / (val_step)

            # str_log = "rank: {} TrainLoss: {}, Epoch: {} Lr: {}, accu: {}".format(local_rank, loss, epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
            # logging.info(str_log)

            # 只在主进程执行记录日志和更新进度条
            str_log = "rank: {} ValLoss: {}, Epoch: {} Lr: {}, accu: {}".format(local_rank, loss, epoch, lr_scheduler.get_last_lr()[0], accu_num.item() / sample_num)
            logging.info(str_log)
            val_accu_list.append(math.floor((accu_num.item() / sample_num) * 1000) / 1000)
            val_loss_list.append(math.floor(loss * 1000) / 1000)

        # 保存断点数据
        if (epoch + 1) % 1 == 0:
            save_path = r"/root/autodl-tmp/weights_resnet50"
            os.makedirs(save_path, exist_ok=True)
            save_file = os.path.join(save_path, f"epoch_{epoch}.pth")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
            }, save_file)
            logging.warning(f"checkpoint has been saved in {save_file}")


    # 关闭tensorboard
    writer.close()

    plt_result(np.arange(cfg.epochs), train_loss_list, "Train loss")
    plt_result(np.arange(cfg.epochs), val_loss_list, "Val loss")
    plt_result(np.arange(cfg.epochs), train_accu_list, "Train accu")
    plt_result(np.arange(cfg.epochs), val_accu_list, "Val accu")

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

        logfile = "./logs/" + str(formatted_time) +".log"
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