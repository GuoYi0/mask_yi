import numpy as np
import config
def _generate_basic_anchors(heights, width, scale):
    anchors = np.zeros((len(heights), 4), np.float32)
    x_ctr = (scale-1)/2
    y_ctr = (scale-1)/2
    anchors[:,0] = -width/2 + x_ctr
    anchors[:,2] = width/2 + x_ctr
    anchors[:, 1] = - heights / 2 + y_ctr
    anchors[:, 3] = heights / 2 + y_ctr
    return anchors


def generate_anchors(scales, ratios, shape, feature_stride, anchor_stride):
    """
    :param scales: 一个整数，所要生成的anchor大小，像素坐标,例如最小anchor尺寸 32
    :param ratios: 一个元组，(0.5,1.0,2)
    :param shape: （高，宽） 该层feature map的大小
    :param feature_stride: 该层feature map相对于输入图片的缩放比，就是resolution
    :param anchor_stride: 一般是1
    :return: 生成的anchor [N, （x1, y1, x2, y2）]
    """
    r = np.sqrt(ratios)
    heights = scales / r
    widths = scales * r  # anchor的高宽
    # 返回一个3行4列矩阵，每行是一个anchor，x1,y1,x2,y2.他是相对于中心位置的坐标，中心位置的坐标是
    #     x_ctr = (scale-1)/2， y_ctr = (scale-1)/2
    _anchors = _generate_basic_anchors(np.array(heights), np.array(widths), feature_stride)

    shifts_x = np.arange(0, shape[1], anchor_stride) * feature_stride
    shifts_y = np.arange(0, shape[0], anchor_stride) * feature_stride
    shift_x, shift_y = np.meshgrid(shifts_x, shifts_y)
    shifts = np.vstack((shift_x.ravel(), shift_y.ravel(),
                        shift_x.ravel(), shift_y.ravel())).transpose()
    A = _anchors.shape[0] # anchor个数
    K = shifts.shape[0] # feature_map的像素个数

    num_coords = 4  # 这里的4，表示每个坐标由四个数来表达
    all_anchors = (_anchors.reshape((1, A, num_coords)) +
                   shifts.reshape((1, K, num_coords)).transpose((1, 0, 2)))
    all_anchors = all_anchors.reshape((K * A, num_coords))

    return all_anchors


def generate_pyramid_anchors(batch_size,resolution, image_shape, smallest_anchor_scale):
    """
    :param resolution:  一个int列表，其长度等于feature_map的层级个数，由底层到高层，依次是缩放比。对于resnet50而言，是
    [8, 16， 32， 64， 128]
    :param image_shape: 二元组，(gao ,kuan)输入图片的大小
    :param smallest_anchor_scale: 最底层feature map上面每个像素对应的anchor大小
    :return: 所有的anchor，由底层到高层拼接起来，形状是[批数， 个数，(x1, y1, x2, y2)],相对于输入图片的像素坐标
    """
    anchors = []
    anchor_scale = smallest_anchor_scale
    for i in range(len(resolution)):
        reso = resolution[i]
        anchor = generate_anchors(scales=anchor_scale, ratios=config.anchor_ratios,
                                  shape=(image_shape[0] / reso, image_shape[1] / reso), feature_stride=reso,
                                  anchor_stride=1)
        anchors.append(anchor)
        anchor_scale *= 2

    a = np.concatenate(anchors, axis=0)
    final_anchors = np.broadcast_to(a, (batch_size,)+a.shape)
    return final_anchors


def norm_boxes(boxes, shape):
    """
    把box从像素坐标转换为normalize坐标,我们shif，是因为在像素坐标系下，区间形式是[，)即包左不包右
    :param boxes:
    :param shape:
    :return:
    """
    if boxes.shape[0] > 0 :
        height = shape[0]
        width = shape[1]
        scale = np.array([width-1,height-1,width-1,height-1])
        shift = np.array([0, 0, 1, 1])
        boxes = np.divide((boxes-shift), scale).astype(np.float32)
    else:
        boxes = boxes.astype(np.float32)
    return boxes




if __name__ == "__main__":
    h = np.array([2, 3, 4])
    w = np.array([2, 3, 4])
    an = generate_anchors(32, (1,), (2, 2), 4, 1)
    print(an)