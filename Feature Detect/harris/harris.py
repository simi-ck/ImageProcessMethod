from scipy.ndimage import filters
import numpy as np
from pylab import *
from PIL import Image
np.seterr(divide='ignore', invalid='ignore')
def compute_harris_response(im, sigma=3):
    imx = np.zeros(im.shape)  # 计算导数
    filters.gaussian_filter(im, (sigma, sigma), (0, 1), imx)
    imy = np.zeros(im.shape)
    filters.gaussian_filter(im, (sigma, sigma), (1, 0), imy)
    Wxx = filters.gaussian_filter(imx * imx, sigma)
    # 计算harris矩阵分量
    Wxy = filters.gaussian_filter(imx * imy, sigma)
    Wyy = filters.gaussian_filter(imy * imy, sigma)
    Wdet = Wxx * Wyy - Wxy ** 2  # 计算矩阵的特征值和迹
    Wtr = Wxx + Wyy
    return Wdet / Wtr


def get_harris_points(harrisim, min_dist=10, threshold=0.1):
    conner_threshold = harrisim.max() * threshold
    harrisim_t = (harrisim > conner_threshold) * 1
    # 得到特征点中大于阈值的点的坐标
    coords = np.array(harrisim_t.nonzero()).T
    # 得到特征点中大于阈值的点的值
    candidate_values = [harrisim[c[0], c[1]] for c in coords]
    # 得到索引
    index = np.argsort(candidate_values)
    allowed_locations = np.zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist, min_dist:-min_dist] = 1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i, 0], coords[i, 1]] == 1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i, 0] - min_dist):(coords[i, 0] + min_dist),
            (coords[i, 1] - min_dist):(coords[i, 1] + min_dist)] = 0  # 此处保证min_dist*min_dist仅仅有一个harris特征点
    return filtered_coords


def plot_harris_points(image, filtered_coords):
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '+')
    axis('off')
    show()


if __name__ == '__main__':
    img = np.array(Image.open('../img/flowers.png').convert('L'))
    harrisim = compute_harris_response(img)
    print(harrisim)
    filter_coords = get_harris_points(harrisim)
    print(filter_coords)
    plot_harris_points(img, filter_coords)
