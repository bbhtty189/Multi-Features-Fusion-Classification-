from PIL import Image
import torch
from traditional_features import *
from torchvision import transforms

img_path = r"E:\datasets\MTARSI\airplane-datasets-new\B-52\3-111.jpg"

img = Image.open(img_path).convert("RGB")

img_size = 224
train_transform = transforms.Compose([transforms.Resize((img_size, img_size)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

img = train_transform(img)
c, h, w = img.shape
rgb_img = img.clone()

# img 是一个归一化的图像，其形状为 (C, H, W)
mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

# 对 img 执行逆归一化操作
rgb_img = ((rgb_img * std) + mean) * 255
# C,H,W -> H,W,C
rgb_img = rgb_img.permute(1, 2, 0).numpy()
# BGR2RGB
rgb_img = rgb_img[:, :, ::-1]
rgb_img = rgb_img.astype(np.uint8)

# 直接提取关键点效果很不好
# keypoints, descriptors = extract_sift(rbg_img)

# lbp_img = convert_lab(rgb_img) //先转换成lab，之后提取比较详细的关键点
# lab_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2Lab)
# cv2.imshow("111", lab_img)
# cv2.waitKey(0)

# 二值化图像，之后提取关键点效果很不好
# h, w, _ = rgb_img.shape
# # 将图像转换为灰度图像
# gray_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
# # 对灰度图像进行二值化操作
# _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)
# cv2.imshow("111", binary_image)
# cv2.waitKey(0)

# rgb转换成HLS, 提取出细节最丰富的关键点
hls_image = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HLS)
cv2.imshow("111", hls_image)
cv2.waitKey(0)

# rgb转换成HSV, 关键点提取效果效果一般
# hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2HSV)
# cv2.imshow("111", hsv_img)
# cv2.waitKey(0)

# 将图像转换为灰度图
gray_img = cv2.cvtColor(hls_image, cv2.COLOR_RGB2GRAY)

# 创建SIFT对象
sift = cv2.SIFT_create()

# 提取关键点和SIFT特征
keypoints, descriptors = sift.detectAndCompute(gray_img, None)

print("Number of keypoints:", len(keypoints))
# keypoints, x:kp.pt[0]; y:kp.pt[1]
keypoints_array = np.array([(kp.pt[0], kp.pt[1]) for kp in keypoints])
print("keypoints:", keypoints_array)
# descriptors
print("Descriptors shape:", descriptors.shape)
print("Descriptors: ", descriptors)

def expandDescriptors(keypoints, descriptors, total_num):
    num_descriptors = descriptors.shape[0]
    print(num_descriptors)
    if num_descriptors < total_num:
        need_num = total_num - num_descriptors
        i = 0
        add_keypoints = []
        add_descriptors = []
        while(i < need_num):
            mid_x = (keypoints[i][0] + keypoints[i+1][0]) / 2
            mid_y = (keypoints[i][1] + keypoints[i+1][1]) / 2
            mid_descriptors = (descriptors[i] + descriptors[i+1]) / 2
            add_keypoints.append((mid_x, mid_y))
            add_descriptors.append(mid_descriptors)
            i += 1

        add_keypoints = np.array(add_keypoints)
        add_descriptors = np.array(add_descriptors)

        total_keypoints = np.concatenate((keypoints, add_keypoints), axis=0)
        total_descriptors = np.concatenate((descriptors, add_descriptors), axis=0)

        return total_keypoints, total_descriptors

    elif num_descriptors > total_num:
        keypoints = keypoints[:total_num]
        descriptors = descriptors[:total_num]
        return keypoints, descriptors

    else:
        return keypoints, descriptors

total_num = 500

new_keypoints, new_descriptors = expandDescriptors(keypoints_array, descriptors, total_num)

print(new_keypoints.shape)
print(new_descriptors.shape)

# 绘制关键点
# image_with_keypoints = cv2.drawKeypoints(gray_img, new_keypoints, None, flags=cv2.DRAW_MATCHES_FLAGS_DEFAULT)
# 设置画圆的半径和颜色
radius = 1
color = (0, 0, 255)  # 红色

color_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
# 在图像上绘制每个点
for point in new_keypoints:
    # print(point)
    cv2.circle(color_img, (int(point[0]), int(point[1])), radius, color, -1)  # -1 表示填充整个圆

cv2.imshow("111", color_img)
cv2.waitKey(0)
cv2.imwrite(r"E:\result\sift_img.png", color_img)





