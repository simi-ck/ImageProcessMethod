import numpy as np
from matplotlib import pyplot as plt
import math
from PIL import Image
import cv2 as cv
import skimage
from skimage import data, color, draw, transform, feature, util


# 圆检测 opencv和scipy

# image- 8位，单通道，灰度输入图像。
# circles- 找到的圆的输出向量。每个向量被编码为3元素的浮点向量 （x，y，半径）。
# circle_storage - 在C函数中，这是一个将包含找到的圆的输出序列的内存存储。
# method- 使用检测方法。目前，唯一实现的方法是 CV_HOUGH_GRADIENT，基本上是 21HT，在[Yuen90]中有描述 。
# dp - 累加器分辨率与图像分辨率的反比。例如，如果 dp = 1，则累加器具有与输入图像相同的分辨率。如果 dp = 2，则累加器的宽度和高度都是一半。
# minDist -检测到的圆的中心之间的最小距离。如果参数太小，除了真正的参数外，可能会错误地检测到多个邻居圈。如果太大，可能会错过一些圈子。
# param1 - 第一个方法特定的参数。在CV_HOUGH_GRADIENT的情况下， 两个传递给Canny（）边缘检测器的阈值较高（较小的两个小于两倍）。
# param2 - 第二种方法参数。在CV_HOUGH_GRADIENT的情况下
# ，它是检测阶段的圆心的累加器阈值。越小，可能会检测到越多的虚假圈子。首先返回对应于较大累加器值的圈子。
# minRadius -最小圆半径。
# maxRadius - 最大圆半径。

def opencv_hough_circle(image):
    dst = cv.pyrMeanShiftFiltering(image, 10, 100)   #边缘保留滤波EPF
    grayImage = cv.cvtColor(dst, cv.COLOR_RGB2GRAY)
    cv.Hough
    circles = cv.HoughCircles(grayImage, cv.HOUGH_GRADIENT, 1, 100, param1=60, param2=10, minRadius=5, maxRadius=500)
    for circle in circles[0]:
        x = int(circle[0])
        y = int(circle[1])
        r = int(circle[2])
        img = cv.circle(image, (x, y), r, (0, 0, 255), -1)
    cv.imshow("circles", image)

def skimage_hough_circle():
    image = util.img_as_ubyte(data.coins()[0:95, 70:370])  # 裁剪原图片
    edges = feature.canny(image, sigma=3, low_threshold=10, high_threshold=50)  # 检测canny边缘

    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(8, 5))

    ax0.imshow(edges, cmap=plt.cm.gray)  # 显示canny边缘
    ax0.set_title('original iamge')

    hough_radii = np.arange(15, 30, 2)  # 半径范围
    hough_res = transform.hough_circle(edges, hough_radii)  # 圆变换

    centers = []  # 保存中心点坐标
    accums = []  # 累积值
    radii = []  # 半径

    for radius, h in zip(hough_radii, hough_res):
        # 每一个半径值，取出其中两个圆
        num_peaks = 2
        peaks = feature.peak_local_max(h, num_peaks=num_peaks)  # 取出峰值
        centers.extend(peaks)
        accums.extend(h[peaks[:, 0], peaks[:, 1]])
        radii.extend([radius] * num_peaks)

    # 画出最接近的5个圆
    image = color.gray2rgb(image)
    for idx in np.argsort(accums)[::-1][:5]:
        center_x, center_y = centers[idx]
        radius = radii[idx]
        cx, cy = draw.circle_perimeter(center_y, center_x, radius)
        image[cy, cx] = (255, 0, 0)

    ax1.imshow(image)
    ax1.set_title('detected image')
    plt.show()

# 椭圆检测 使用scipy

# img: 待检测图像。
# accuracy: 使用在累加器上的短轴二进制尺寸，是一个double型的值，默认为1
# thresh: 累加器阈值，默认为4
# min_size: 长轴最小长度，默认为4
# max_size: 短轴最大长度，默认为None,表示图片最短边的一半。
# 返回一个 [(accumulator, y0, x0, a, b, orientation)] 数组，accumulator表示累加器，（y0,x0)表示椭圆中心点，（a,b)分别表示长短轴，orientation表示椭圆方向
def scipy_ellipse():
    image_rgb = data.coffee()[0:220, 160:420]  # 裁剪原图像，不然速度非常慢
    image_gray = color.rgb2gray(image_rgb)
    edges = feature.canny(image_gray, sigma=2.0, low_threshold=0.55, high_threshold=0.8)

    # 执行椭圆变换
    result = transform.hough_ellipse(edges, accuracy=20, threshold=250, min_size=100, max_size=120)
    result.sort(order='accumulator')  # 根据累加器排序

    # 估计椭圆参数
    best = list(result[-1])  # 排完序后取最后一个
    yc, xc, a, b = [int(round(x)) for x in best[1:5]]
    orientation = best[5]

    # 在原图上画出椭圆
    cy, cx = draw.ellipse_perimeter(yc, xc, a, b, orientation)
    image_rgb[cy, cx] = (0, 0, 255)  # 在原图中用蓝色表示检测出的椭圆

    # 分别用白色表示canny边缘，用红色表示检测出的椭圆，进行对比
    edges = color.gray2rgb(edges)
    edges[cy, cx] = (250, 0, 0)

    fig2, (ax1, ax2) = plt.subplots(ncols=2, nrows=1, figsize=(8, 4))

    ax1.set_title('Original picture')
    ax1.imshow(image_rgb)

    ax2.set_title('Edge (white) and result (red)')
    ax2.imshow(edges)

    plt.show()


if __name__ == '__main__':
    # src = cv.imread('../img/circle1.jpg')
    # cv.namedWindow('input_image', cv.WINDOW_NORMAL)  # 设置为WINDOW_NORMAL可以任意缩放
    # cv.imshow('input_image', src)
    # opencv_hough_circle(src)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    skimage_hough_circle()

