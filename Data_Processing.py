import cv2
import numpy as np
from torchvision import transforms
from PIL import Image
import random

#归一化处理
def normalized(image):
    #创建相应大小的result矩阵存储归一化后的样本数据
    result = np.zeros(image.shape, dtype=np.float32)
    #归一化处理，样本值在[-1，1]之间
    result = image / 127.5 - 1
    print(result)
    return result

#颜色失真，s是颜色失真的强度
def get_color_distortion(image,s=1.0):
    #改变图像的属性：亮度、对比度、饱和度和色调
    color_jitter = transforms.ColorJitter(0.8*s, 0.8*s, 0.8*s, 0.2*s)
    #给一个transform加上概率，以一定的概率执行该操作
    rnd_color_jitter = transforms.RandomApply([color_jitter], p=0.8)
    #依概率p将图片转换为灰度图
    rnd_gray = transforms.RandomGrayscale(p=0.2)
    #整合多个步骤
    color_distort = transforms.Compose([
    rnd_color_jitter,
    rnd_gray])
    target_image=color_distort(image)
    return target_image

#点变换
def point_move(img):
    # 获取样本图像长与宽
    h, w = img.shape[0:2]
    # N对基准控制点
    N = 5
    # 边框长度
    pad_pix = 50
    points = []
    # 每对基准点的x坐标差值
    dx = int(w/(N - 1))
    # 加边框后每个基准点的坐标，存于points列表中
    for i in range(N):
        points.append((dx * i,  pad_pix))
        points.append((dx * i, pad_pix + h))
    # 三通道同时加边框
    img = cv2.copyMakeBorder(img, pad_pix, pad_pix, 0, 0, cv2.BORDER_CONSTANT,
                             value=(int(img[0][0][0]), int(img[0][0][1]), int(img[0][0][2])))
    # 原点
    source = np.array(points, np.int32)
    # 一维两列n行
    source = source.reshape(1, -1, 2)
    # 随机扰动幅度20到30或-20到-30
    rand_num_pos = random.uniform(0, 30)
    rand_num_neg = -1 * rand_num_pos
    newpoints = []
    for i in range(N):
        # 百分之五十向上抖动，百分之五十向下抖动
        nx_up = points[2 * i][0] + np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        ny_up = points[2 * i][1] + np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        nx_down = points[2 * i + 1][0] + np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        ny_down = points[2 * i + 1][1] + np.random.choice([rand_num_neg, rand_num_pos], p=[0.5, 0.5])
        # 保存新的点
        newpoints.append((nx_up, ny_up))
        newpoints.append((nx_down, ny_down))
    # target点，将新的点矩阵化，一维两列n行
    target = np.array(newpoints, np.int32)
    target = target.reshape(1, -1, 2)
    # 计算matches
    matches = []
    for i in range(1, 2*N + 1):
        matches.append(cv2.DMatch(i, i, 0))

    return source, target, matches, img

if __name__=='__main__':
    for i in range(10):
        img = cv2.imread('data/test.png', cv2.IMREAD_COLOR)
        # 返回变换点的信息
        source, target, matches, img = point_move(img)
        # 薄板样条插值
        tps = cv2.createThinPlateSplineShapeTransformer()
        tps.estimateTransformation(source, target, matches)
        # 局部纠正
        img = tps.warpImage(img)
        # cv2.imshow("norm", img)
        #cv2.waitKey(0)
        # CV格式转PIL
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        # 颜色失真
        img = get_color_distortion(img)
        # PIL转CV格式
        img = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
        # 高斯模糊，正态分布，1.5
        img = cv2.GaussianBlur(img, (3, 3), 1.5)
        #norm=normalized(img)
        # cv2.imshow("norm", img)
        # cv2.waitKey(0)
        cv2.imwrite('data_set/train_pic/train_'+str(i)+'.img',img)
