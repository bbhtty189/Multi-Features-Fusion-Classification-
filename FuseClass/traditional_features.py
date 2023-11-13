import numpy as np
import cv2
import skimage.feature as skft
import mahotas
from scipy.stats import kurtosis, skew

def calculate_lbp(img):
    '''
    :param img: RGB img (H,W,3)
    :return: LBP map (H,W,1)
    '''

    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 获取图像的边缘特征
    lbpmap = skft.local_binary_pattern(gray_img, 8, 1.0, method="var")
    lbpmap = lbpmap.astype(np.uint8)

    # cv2.imshow("111", lbpmap)
    # cv2.waitKey(0)
    return lbpmap

def calculate_canny(img):
    '''
    :param img: RGB img (H,W,3)
    :return: Canny map (H,W,1)
    '''

    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 获取图像的边缘特征
    cannymap = cv2.Canny(gray_img, 100, 200, L2gradient=True)

    # cv2.imshow("111", cannymap)
    # cv2.waitKey(0)

    return cannymap

def calculate_hog(img, cell_size=(8, 8), block_size=(2, 2), nbins=9):
    '''
    :param img: RGB img (H,W,3)
    :param cell_size: cell size
    :param block_size: block size
    :param nbins: nbins
    :return: hog features
    '''

    # 计算灰度图的HOG特征
    features, hog_image = skft.hog(img, orientations=nbins, pixels_per_cell=cell_size,
                              cells_per_block=block_size, visualize=True, channel_axis=2)
    features = np.array(features)

    # cv2.imshow("111", hog_image)
    # cv2.waitKey(0)

    return features

def calculate_child_hog(img, cell_size=(8, 8), block_size=(2, 2), nbins=9, num=3):
    '''
    :param img: RGB img (H,W,3)
    :param cell_size: cell size
    :param block_size: block size
    :param nbins: nbins
    :return: hog features
    '''

    h, w, _ = img.shape
    child_imgs = []
    x = np.linspace(0, w, num+1).astype(np.longlong)
    y = np.linspace(0, h, num+1).astype(np.longlong)
    print(img.shape)
    print(len(x), x)
    print(len(y), y)
    # 当num=3时, 那么将图像分为3x3共9个子图
    for i in range(num):
        for j in range(num):
            child_img = img[y[i]:y[i + 1], x[j]:x[j + 1], :]
            child_imgs.append(child_img)

    all_features = []
    for child_img in child_imgs:
        # 计算HOG特征
        features, hog_image = skft.hog(child_img, orientations=nbins, pixels_per_cell=cell_size,
                                  cells_per_block=block_size, visualize=True, channel_axis=2)
        all_features.append(features)

    all_features = np.array(all_features)
    all_features = all_features.reshape(-1)

    # cv2.imshow("111", hog_image)
    # cv2.waitKey(0)

    return all_features

def extract_sift(img):
    '''
    :param img: RGB img (H,W,3)
    :return: sift features
    '''
    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 创建SIFT对象
    sift = cv2.SIFT_create()

    # 提取关键点和SIFT特征
    keypoints, descriptors = sift.detectAndCompute(gray_img, None)

    return keypoints, descriptors

def calculate_zernike(img, degree=8):
    '''
    :param img: RGB img (H,W,3)
    :param degree: degree
    :return: zernike features
    '''

    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 二值化图像
    _, threshold_image = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # 计算Zernike矩
    zernike_moments = mahotas.features.zernike_moments(threshold_image, degree)

    return zernike_moments

def calculate_hu(img):
    '''
    :param img: RGB img (H,W,3)
    :return: hu moments
    '''

    # 将图像转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 二值化图像
    _, threshold_image = cv2.threshold(gray_img, 127, 255, cv2.THRESH_BINARY)

    # 计算图像的矩
    moments = cv2.moments(threshold_image)

    # 计算Hu矩
    hu_moments = cv2.HuMoments(moments)

    # 归一化 Hu 矩
    hu_moments_normalized =  -np.sign(hu_moments) * np.log10(np.abs(hu_moments))

    return hu_moments_normalized

def image_binary_entropy(img):
    '''
    :param img: RGB img (H,W,3)
    :return: binary entropy of image
    '''

    h, w,_ = img.shape
    # 将图像转换为灰度图像
    gray_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    # 对灰度图像进行二值化操作
    _, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

    white_pixels = np.count_nonzero(binary_image == 255)
    entropy = np.log2(white_pixels) if white_pixels > 0 else 0

    return entropy

def convert_lab(img):
    '''
    :param img: RGB img (H,W,3)
    :return: lab img
    '''

    # 将图像从BGR格式转换为LAB格式
    lab_img = cv2.cvtColor(img, cv2.COLOR_RGB2Lab)

    # cv2.imshow("111", lab_img)
    # cv2.waitKey(0)

    return lab_img

def convert_hls(img):
    '''
    :param img: RGB img (H,W,3)
    :return: hue, light, saturation of img
    '''

    # 将图像从BGR格式转换为HLS格式
    hls_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)

    # Hue
    hue = hls_image[:, :, 0]

    # light
    light = hls_image[:, :, 1]

    # Saturation
    saturation = hls_image[:, :, 2]

    cv2.imshow("111", hue)
    cv2.waitKey(0)

    cv2.imshow("112", light)
    cv2.waitKey(0)

    cv2.imshow("113", saturation)
    cv2.waitKey(0)

    return hue, light, saturation

def calculate_color_hist(img, num_bins=8):
    '''
    :param img: RGB img (H,W,3)
    :param num_bins: num bins
    :return:
    '''

    # 计算直方图
    hist = cv2.calcHist([img], [0, 1, 2], None, [num_bins, num_bins, num_bins], [0, 256, 0, 256, 0, 256])

    # 将三维直方图归一化并转换为一维数组
    hist = cv2.normalize(hist, hist).flatten()

    return hist

def image_stats(img):
    '''
    :param img: RGB img (H,W,3)
    :return: mean, variance, std_deviation, kurt, skewness of img
    '''

    # 计算均值
    mean = np.mean(img)

    # 计算方差
    variance = np.var(img)

    # 计算标准差
    std_deviation = np.std(img)

    # 计算峰度
    kurt = kurtosis(img.flatten())

    # 计算偏度
    skewness = skew(img.flatten())

    return mean, variance, std_deviation, kurt, skewness

