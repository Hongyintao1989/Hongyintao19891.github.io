import cv2
import numpy as np
import torch


def crop_to_shape(data, shape):
    """
    将数组裁剪为给定图像形状，通过移除边界实现
    (期望输入是形状为[batch, nx, ny, channels]的张量)

    :param data: 要裁剪的数组
    :param shape: 目标形状
    """
    # 计算偏移量
    offset0 = (data.shape[1] - shape[1]) // 2
    offset1 = (data.shape[2] - shape[2]) // 2
    
    # 根据偏移量情况进行裁剪
    if offset0 == 0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:data.shape[1], offset1:(-offset1 - 1)]
        else:
            return data[:, offset0:data.shape[1], offset1:(-offset1)]
    elif offset1 == 0:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:data.shape[2]]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
        else:
            return data[:, offset0:-offset0, offset1:data.shape[2]]
    else:
        if data.shape[1] % 2 == 1 or shape[1] % 2 == 1:
            return data[:, offset0:(-offset0 - 1), offset1:(-offset1)]
        elif data.shape[2] % 2 == 1 or shape[2] % 2 == 1:
            return data[:, offset0:-offset0, offset1:(-offset1-1)]
        else:
            return data[:, offset0:-offset0, offset1:(-offset1)]


def img_augmentation(x_train, y_train):
    """
    对训练数据进行图像增强
    
    :param x_train: 输入图像数组
    :param y_train: 输入标签数组
    :return: 增强后的数据集
    """
    x_rotat = []
    y_rotat = []
    x_flip = []
    y_flip = []

    for idx in range(len(x_train)):
        # 随机旋转
        x, y = random_rotation(x_train[idx], y_train[idx])
        x_rotat.append(x)
        y_rotat.append(y)
        
        # 水平垂直翻转
        x, y = hv_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
        
        # 水平翻转
        x, y = h_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
        
        # 垂直翻转
        x, y = v_flip(x_train[idx], y_train[idx])
        x_flip.append(x)
        y_flip.append(y)
    
    return np.array(x_rotat), np.array(y_rotat), np.array(x_flip), np.array(y_flip)


def random_rotation(x_image, y_image):
    """
    随机旋转图像和标签
    
    :param x_image: 输入图像
    :param y_image: 输入标签
    :return: 旋转后的图像和标签
    """
    rows_x, cols_x, chl_x = x_image.shape
    rows_y, cols_y = y_image.shape
    rand_num = np.random.randint(-40, 40)
    
    M1 = cv2.getRotationMatrix2D((cols_x/2, rows_x/2), rand_num, 1)
    M2 = cv2.getRotationMatrix2D((cols_y/2, rows_y/2), rand_num, 1)
    
    x_image = cv2.warpAffine(x_image, M1, (cols_x, rows_x))
    y_image = cv2.warpAffine(y_image, M2, (cols_y, rows_y))
    
    return np.array(x_image), np.array(y_image)


def h_flip(x_image, y_image):
    """
    水平翻转图像和标签
    
    :param x_image: 输入图像
    :param y_image: 输入标签
    :return: 翻转后的图像和标签
    """
    x_image = cv2.flip(x_image, 1)
    y_image = cv2.flip(y_image, 1)
    return x_image, y_image


def v_flip(x_image, y_image):
    """
    垂直翻转图像和标签
    
    :param x_image: 输入图像
    :param y_image: 输入标签
    :return: 翻转后的图像和标签
    """
    x_image = cv2.flip(x_image, 0)
    y_image = cv2.flip(y_image, 0)
    return x_image, y_image


def hv_flip(x_image, y_image):
    """
    水平垂直翻转图像和标签
    
    :param x_image: 输入图像
    :param y_image: 输入标签
    :return: 翻转后的图像和标签
    """
    x_image = cv2.flip(x_image, -1)
    y_image = cv2.flip(y_image, -1)
    return x_image, y_image


# 添加PyTorch特有的工具函数
def numpy_to_torch(x):
    """
    将NumPy数组转换为PyTorch张量
    
    :param x: NumPy数组
    :return: PyTorch张量
    """
    if len(x.shape) == 4:  # 处理批量图像数据 [B, H, W, C]
        x = x.transpose(0, 3, 1, 2)  # 转为[B, C, H, W]
        return torch.from_numpy(x).float()
    elif len(x.shape) == 3:  # 处理单张图像 [H, W, C]
        x = x.transpose(2, 0, 1)  # 转为[C, H, W]
        return torch.from_numpy(x).float()
    else:
        return torch.from_numpy(x).float()


def torch_to_numpy(x):
    """
    将PyTorch张量转换为NumPy数组
    
    :param x: PyTorch张量
    :return: NumPy数组
    """
    if x.dim() == 4:  # 处理批量图像数据 [B, C, H, W]
        x = x.cpu().numpy().transpose(0, 2, 3, 1)  # 转为[B, H, W, C]
    elif x.dim() == 3:  # 处理单张图像 [C, H, W]
        x = x.cpu().numpy().transpose(1, 2, 0)  # 转为[H, W, C]
    else:
        x = x.cpu().numpy()
    return x