import torch
import torch.nn as nn
from models.resnet import resnet50, resnet101
from models.VGG_cond import vgg

"""
    (B,N,C)中，N是特征的大小或数量，C是特征的种类
    2023-6-3[待做]:
        (1)对于特征图谱类型的人工特征：
            使用Xception等网络从多种人工特征图谱中提取更为细致的空间语义信息和空间关系信息，
            用于主网络进行分类，可以进一步扩展到检测任务。
            
            也可以切割人工特征图谱，切分为跟深度学习特征图谱一样大小的小特征图谱，然后通道合并。
        
        (2)对于特征向量类型的人工特征：
            对于高维度特征向量，先使用PCA降维到固定维度的特征向量，对于低纬度特征向量，可以使用全连接层升维到固定维度的特征向量，得到(B,N1,C)形式的特征向量。
            将融合后的特征图谱改变为(B,N2,C)的形式，通过2个全连接层学习权重因子，之后进行权重相乘并线性串联得到(B,N1+N2,C)的特征向量，再经过一个全连接层降维到
            (B,N2,C)，之后根据将特征向量转换成特征图谱，并进行分类预测或检测预测。
    2023-6-4[待做]:
        人工特征的有效性验证，及其组合方案对于效果的影响验证。
    2023-6-9[完成]:
        对于特征向量，使用全连接层将其调整到相同的特征空间(统一的维度)；对于特征图谱，使用卷积层将其调整到到相同的特征空间(统一的维度）；之后先验证简单操作中的拼接，逐元素相加
        和加权求和的特征融合方法。
    2023-6-10[待做]:
        验证基于注意力的融合方法
    2023-6-11-12[待做]:
        验证基于张量的方法
"""

class extractmodel(nn.Module):
    def __init__(self, cfg):
        super(extractmodel, self).__init__()
        self.num_classes = cfg.num_classes
        self.include_top = cfg.include_top
        self.easy_method = cfg.easy_method
        # self.hog_dim = cfg.hog_dim
        self.basic_dim = 512
        self.total_dim = 0

        # resnet50
        if cfg.model_type == "resnet50":
            self.total_dim = self.basic_dim * 4
            self.model = resnet50(self.num_classes, self.include_top)
        elif cfg.model_type == "resnet101":
            self.total_dim = self.basic_dim * 4
            self.model = resnet101(self.num_classes, self.include_top)
        # vgg
        elif cfg.model_type == "vgg16":
            self.total_dim = self.basic_dim
            self.model = vgg("vgg16")
        elif cfg.model_type == "vgg19":
            self.total_dim = self.basic_dim
            self.model = vgg("vgg19")

        # extract deep feature maps from shallow feature maps
        self.extract = nn.Sequential(
            nn.Conv2d(1, (self.total_dim // 16), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.total_dim // 16),
            nn.ReLU(),
            nn.Conv2d((self.total_dim // 16), (self.total_dim // 8), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.total_dim // 8),
            nn.ReLU(),
            nn.Conv2d((self.total_dim // 8), (self.total_dim // 4), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.total_dim // 4),
            nn.ReLU(),
            nn.Conv2d((self.total_dim // 4), (self.total_dim // 2), kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.total_dim // 2),
            nn.ReLU(),
            nn.Conv2d((self.total_dim // 2), self.total_dim, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # extract deep features from shallow features
        # self.extract_hog = nn.Linear(self.hog_dim, self.total_dim)


        # weight-sum
        self.weight_cond = nn.Parameter(torch.ones(self.total_dim, 7, 7))
        self.weight_canny = nn.Parameter(torch.ones(self.total_dim, 7, 7))
        # self.weight_hog = nn.Parameter(torch.ones(2048))

        # gate unit
        self.cond_gate = nn.Linear(self.total_dim, self.total_dim)
        self.canny_gate = nn.Linear(self.total_dim, self.total_dim)

        # class-head
        # cat
        self.dim_reduction = nn.Conv2d(self.total_dim*2, self.total_dim, kernel_size=1, stride=1)
        self.dim_reduction_linear = nn.Linear(self.total_dim*2, self.total_dim)
        self.pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(self.total_dim, self.num_classes)

    def forward(self, img, canny):
        """
            cond: (B, C, H, W)
            conny_features = (B, C, H, W)
            C = 2048, H=7, W=7
        """

        cond, _ = self.model(img)
        canny_features = self.extract(canny)

        fuse_features = None
        if self.easy_method == "cat":
            # fuse canny on channels
            fuse_features = torch.cat((cond, canny_features), dim=1)
            fuse_features = self.dim_reduction(fuse_features)
            fuse_features = self.pooling(fuse_features)
            # (B,C,1,1) -> (B,C)
            fuse_features = torch.flatten(fuse_features, 1)

            # fuse hog
            # hog = self.extract_hog(hog)
            # fuse_features = torch.cat((fuse_features, hog), dim=1)
            # fuse_features = self.dim_reduction_linear(fuse_features)

            x = self.fc(fuse_features)
            return x

        elif self.easy_method == "element-add":
            # fuse canny by adding element one by one.
            fuse_features = cond + canny_features
            fuse_features = self.pooling(fuse_features)
            # (B,C,1,1) -> (B,C)
            fuse_features = torch.flatten(fuse_features, 1)

            # fuse hog
            # hog = self.extract_hog(hog)
            # fuse_features = fuse_features + hog

            x = self.fc(fuse_features)
            return x

        elif self.easy_method == "weighted-sum": # 效果不太理想，感觉是parameter的问题，换成linear再试试
            # fuse canny by multiply weights
            weighted_cond = torch.mul(cond, self.weight_cond)
            # 求和池化 (B,C,H,W) -> (B,C,N) -> (B,C,1)
            weighted_cond = weighted_cond.view(weighted_cond.shape[0], weighted_cond.shape[1], -1).contiguous()
            weighted_cond = torch.sum(weighted_cond, dim=2)

            weighted_canny = torch.mul(canny_features, self.weight_canny)
            # 求和池化 (B,C,H,W) -> (B,C,N) -> (B,C,1)
            weighted_canny = weighted_canny.view(weighted_canny.shape[0], weighted_canny.shape[1], -1).contiguous()
            weighted_canny = torch.sum(weighted_canny, dim=2)

            # fuse_features = torch.cat((weighted_cond, weighted_canny), dim=1)
            # fuse_features = fuse_features.view(fuse_features.shape[0], fuse_features.shape[1], 1, 1).contiguous()
            # fuse_features = self.dim_reduction(fuse_features)

            fuse_features = weighted_cond + weighted_canny
            
            # (B,C,1,1) 或者 (B,C,1) -> (B,C)
            fuse_features = torch.flatten(fuse_features, 1)

            # fuse hog
            # hog = self.extract_hog(hog)
            # hog = torch.mul(hog, self.weight_hog)
            # fuse_features = torch.cat((fuse_features, hog), dim=1)
            # fuse_features = self.dim_reduction_linear(fuse_features)

            x = self.fc(fuse_features)
            return x

        elif self.easy_method == "gate-unit":
            # (B, 2048, 7, 7) -> (B, 2048, 1, 1) -> (B, 2048)
            cond_pooling = self.pooling(cond)
            cond_pooling = torch.flatten(cond_pooling, 1)

            # (B, 2048, ,7, 7) -> (B, 2048, 1, 1) -> (B, 2048)
            canny_pooling = self.pooling(canny_features)
            canny_pooling = torch.flatten(canny_pooling, 1)

            # cond gate unit
            cond_weights = self.cond_gate(cond_pooling)
            canny_weights = self.canny_gate(canny_pooling)
            cond_weights = cond_weights.view(cond_weights.shape[0], cond_weights.shape[1], 1, 1).contiguous()
            canny_weights = canny_weights.view(canny_weights.shape[0], canny_weights.shape[1], 1, 1).contiguous()

            # print(cond.shape)
            # print(canny_features.shape)
            # print(cond_weights.shape)
            # print(canny_weights.shape)

            # fuse_features = torch.cat((cond_weights * cond, canny_weights * canny_features), dim=1)
            # fuse_features = self.dim_reduction(fuse_features)
            # fuse_features = self.pooling(fuse_features)

            fuse_features = cond_weights * cond + canny_weights * canny_features
            fuse_features = self.pooling(fuse_features)

            # (B,C,1,1) 或者 (B,C,1) -> (B,C)
            fuse_features = torch.flatten(fuse_features, 1)

            # fuse hog
            # hog = self.extract_hog(hog)
            # fuse_features = torch.cat((fuse_features, hog), dim=1)
            # fuse_features = self.dim_reduction_linear(fuse_features)

            x = self.fc(fuse_features)
            return x

if __name__ == "__main__":

    x = torch.randint(0, 10, (1, 2, 4))
    print(x, x.shape)
    x = torch.sum(x, dim=2)
    print(x, x.shape)

    from dataclasses import dataclass
    from torchsummary import torchsummary
    from utils import load_data
    from torchvision import transforms
    from my_dataset_new import MyDataSet

    @dataclass
    class train_cfg:
        # model
        num_classes = 20
        include_top = True
        easy_method = 'weighted-sum'
        hog_dim = 20736

        # train
        model_type = "vgg16"
        img_size = 224
        test_batch_size = 1
        num_workers = 8
        shuffle = True


    cfg = train_cfg()

    model = extractmodel(cfg=cfg).cuda(0)

    test_transform = transforms.Compose([transforms.Resize((cfg.img_size, cfg.img_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    test_path = r"../test/test.txt"
    test_images_path, test_images_label = load_data(test_path)
    test_dataset = MyDataSet(test_images_path, test_images_label, test_transform)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                 batch_size=cfg.test_batch_size,
                                                 num_workers=cfg.num_workers,
                                                 shuffle=cfg.shuffle,
                                                 )

    for img, label, canny in test_dataloader:

        img = img.cuda(0)
        label = label.cuda(0)
        canny = canny.cuda(0)

        pred = model(img, canny)
        print(pred.shape, pred)

        break













