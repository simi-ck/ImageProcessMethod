def cpu_soft_nms(dets, sigma = 0.5, Nt = 0.7, method = 0, weight = 0, thresh = 0.2 ):
    box_len = len(dets)   # box的个数
    for i in range(box_len):
        max_scores = dets[i, 4]

        tmpx1 = dets[i,0]
        tmpy1 = dets[i,1]
        tmpx2 = dets[i,2]
        tmpy2 = dets[i,3]
        ts = dets[i,4]
        max_pos = i

        pos = i+1
        while pos < box_len:
            if dets[max_pos, 4] < dets[pos, 4]:
                max_scores = dets[pos, 4]
                max_pos = pos

            pos += 1

        # 选取置信度最高的框
        # dets[i,0] = dets[max_pos, 0]
        # dets[i,1] = dets[max_pos, 1]
        # dets[i,2] = dets[max_pos, 2]
        # dets[i,3] = dets[max_pos, 3]
        # dets[i,4] = dets[max_pos, 4]
        dets[i,:] = dets[max_pos, :]

        dets[max_pos, 0] = tmpx1
        dets[max_pos, 1] = tmpy1
        dets[max_pos, 2] = tmpx2
        dets[max_pos, 3] = tmpy2
        dets[max_pos, 4] = ts

        # 将置信度最高的 box 赋给临时变量
        tmpx1, tmpy1, tmpx2, tmpy2, ts = dets[i,0], dets[i,1], dets[i,2], dets[i,3], dets[i,4]

        pos = i+1
        while pos < box_len:

            x1 = boxes[pos,0]
            y1 = boxes[pos,1]
            x2 = boxes[pos,2]
            y2 = boxes[pos,3]

            area = (x2 - x1 + 1)*(y2 - y1 + 1)

            iw = (min(tmpx2, x2) - max(tmpx1, x1) + 1)
            ih = (min(tmpy2, y2) - max(tmpy1, y1) + 1)
            if iw > 0 and ih >0:
                overlaps = iw * ih
                ious = overlaps/((tmpx2 - tmpx1 + 1) * (tmpy2 - tmpy1 + 1) + area - overlaps)

                if method==1:    # 线性
                    if ious > Nt:
                        weight = 1 - ious
                    else:
                        weight = 1
                elif method==2:   # gaussian
                    weight = np.exp(-(ious**2)/sigma)
                else:        # original NMS
                    if ious > Nt:
                        weight = 0
                    else:
                        weight = 1

                # 赋予该box新的置信度
                dets[pos, 4] = weight*dets[pos, 4]
                # print(dets[pos, 4])

                # 如果box得分低于阈值thresh，则通过与最后一个框交换来丢弃该框
                if dets[pos,4]< thresh:
                    dets[pos,0], dets[pos,1], dets[pos,2], dets[pos,3] = dets[box_len-1,0], dets[box_len-1,1], dets[box_len-1,2],dets[box_len-1,3]
                    dets[pos,4] = dets[pos,4]

                    box_len = box_len-1
                    pos = pos-1
            pos +=1

    keep = [i for i in range(box_len)]
    return keep