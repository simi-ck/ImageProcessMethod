import numpy as np


def cpu_nums(dets, thresh=0.7):
    x1 = dets[:,0]
    y1 = dets[:,1]
    x2 = dets[:,2]
    y2 = dets[:,3]
    scores = dets[:,4]

    areas = (x2 - x1 + 1)*(y2 - y1 + 1) # 检测框box的面积

    index = scores.argsort()[::-1]  #将每个box的置信度由高到低排序，并返回其在原列表中的索引
    keep = []   # 保留经nms后的box的索引

    while index.size > 0: 

        i = index[0]

        keep.append(i)

        #  求检测框之间的交集的面积
        x11 = np.maximum(x1[i], x1[index[1:]])
        y11 = np.maximum(y1[i], y1[index[1:]])
        x22 = np.minimum(x2[i], x2[index[1:]])
        y22 = np.minimum(y2[i], y2[index[1:]])

        w = np.maximum(0, x22-x11+1)
        h = np.maximum(0, y22-y11+1)
        overlaps = w*h

        ious = overlaps/(areas[i] + areas[index[1:]] - overlaps)   # 检测框的交并比

        idx = np.where(ious < thresh)[0]  # 保留IoU小于阈值的box

        index = index[idx+1]  # idx的长度 比index的长度小1， 所以+1

    return keep