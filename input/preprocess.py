import os
import sys

import cv2
import numpy as np
import numpy.random as npr
from pycocotools.coco import COCO
from lib.gen_box import generate_pyramid_anchors
sys.path.append(os.curdir)
import config
from config import resolution
from config import RPN_BBOX_STD_DEV
from config import pixel_average
from config import random_resize
from config import train_annFile
from config import posi_anchor_thresh
from config import neg_anchor_thresh
from config import smallest_anchor_size
from config import input_shape
from config import MINI_MASK_SHAPE
from config import allowed_border
from config import batch_size
from config import RPN_TRAIN_ANCHORS_PER_IMAGE
from config import USE_MINI_MASK
coco = COCO(train_annFile)

from lib import bbox_overlaps
from lib import draw_boxes

def resize(image, bboxs, segmentations):
    bboxs = bboxs.astype(np.int32)
    if USE_MINI_MASK:
        mini_segs = []
        for i in range(len(segmentations)):
            x1, y1, x2, y2 = bboxs[i]
            mini_seg = segmentations[i][y1:y2+1, x1:x2+1]
            mini_seg = cv2.resize(mini_seg, config.MINI_MASK_SHAPE)
            mini_segs.append(mini_seg)
        segmentations = mini_segs
    else:
        segmentations = [cv2.resize(seg, input_shape) for seg in segmentations]

    resize_ratio_h = image.shape[0] / input_shape[0]
    resize_ratio_w = image.shape[1] / input_shape[1]
    image = cv2.resize(image, input_shape)
    if len(bboxs) > 0:
        bboxs[:, [0, 2]] = bboxs[:, [0, 2]] / resize_ratio_w
        bboxs[:, [1, 3]] = bboxs[:, [1, 3]] / resize_ratio_h
    return image, bboxs, segmentations

def data_augmentation(image, raw_size, bboxs, categories, segmentations):
    """
    :param image: 仅仅去均值以后的图片
    :param raw_size: (高，宽)输入图片的大小
    :param bboxs: numpy数组 [N, (x1, y1, x2, y2)]
    :param segmentations: 一个列表， 每个元素是binary mask (numpy 2D array)
    :param categories： int型一维数组
    :return: crop_and_resize_image, bboxs, segmentations
    """
    rdm_ratio = random_resize[npr.randint(0, len(random_resize))]  # 0.5， 1， 1.5， 2

    new_size = raw_size * rdm_ratio
    image = cv2.resize(image, (int(new_size[1]), int(new_size[0])))

    bboxs = bboxs * rdm_ratio
    segmentations = [cv2.resize(seg.astype(np.float32),(int(new_size[1]), int(new_size[0]))) for seg in segmentations]

    # crop some area form image
    image, bboxs, segmentations, categories = crop_area(image, bboxs, segmentations, categories)


    # image_draw = draw_boxes(image.copy(), bboxs)
    # cv2.imwrite('./demo.jpg', image_draw)

    return image, bboxs, categories, segmentations

# image, bboxs, segmentations, categories
def crop_area(im, bboxs, masks, tags, crop_background=True, max_tries=50):
    '''
    make random crop from the input image
    :param im: [高，宽，通道数] 已经去均值了
    :param bboxs：numpy数组，[N, (x1, y1, x2, y2)]
    :param masks: 一个长度为N的列表，列表的每个元素是一个二维numpy数组，mask，float型
    :param tags: 标签数组
    :param crop_background: 是否裁剪背景
    :param max_tries:
    :return:
    '''

    h, w = im.shape[0:2]
    pad_h = h // 10
    pad_w = w // 10
    bboxs = np.round(bboxs).astype(np.int32)
    h_array = np.zeros((h + pad_h*2,), dtype=np.int32)
    w_array = np.zeros((w + pad_w*2,), dtype=np.int32)
    # 填充的目的是为了增加保持完整图片的可能性
    num = len(masks)
    for i in range(num):
        index_x = np.argwhere(masks[i] > 0.0001)[:,0]
        index_y = np.argwhere(masks[i] > 0.0001)[:,1]
        min_x = min(np.min(index_x), bboxs[i, 0])
        max_x = max(np.max(index_x), bboxs[i, 2])
        min_y = min(np.min(index_y), bboxs[i, 1])
        max_y = max(np.max(index_y), bboxs[i, 3])
        w_array[min_x + pad_w:max_x + pad_w+1] = 1
        h_array[min_y + pad_h:max_y + pad_h+1] = 1

    h_axis = np.where(h_array == 0)[0]
    w_axis = np.where(w_array == 0)[0]
    # if the there is nowhere to crop
    if len(h_axis) == 0 or len(w_axis) == 0:
        return im, bboxs, masks, tags
    for i in range(max_tries):
        xx = np.random.choice(w_axis, size=2)
        yy = np.random.choice(h_axis, size=2)

        xmin = np.min(xx) - pad_w
        xmax = np.max(xx) - pad_w
        xmin = np.clip(xmin, 0, w - 1)
        xmax = np.clip(xmax, 0, w - 1)
        ymin = np.min(yy) - pad_h
        ymax = np.max(yy) - pad_h
        ymin = np.clip(ymin, 0, h - 1)
        ymax = np.clip(ymax, 0, h - 1)

        if xmax - xmin < 0.3 * w or ymax - ymin < 0.3 * h:
            continue
        if len(bboxs) != 0:
            # 找出在裁剪区域以内的框框
            poly_axis_in_area = (bboxs[:, 0] >= xmin) & (bboxs[:, 2] <= xmax) \
                                & (bboxs[:, 1] >= ymin) & (bboxs[:, 3] <= ymax)

            selected_polys = np.where(poly_axis_in_area)[0]
        else:
            selected_polys = []

        if len(selected_polys) == 0:
            if crop_background:  # 裁剪到背景了，原图片返回
                return im, bboxs, masks, tags
            else:
                continue

        im = im[ymin:ymax + 1, xmin:xmax + 1, :]  # 截图
        masks = [masks[i] for i in selected_polys]  # 选择出mask
        masks = [mask[ymin:ymax + 1, xmin:xmax + 1] for mask in masks]  # 截取mask
        tags = tags[selected_polys] # 截取标签
        bboxs = bboxs[selected_polys]
        bboxs[:, 0] -= xmin
        bboxs[:, 1] -= ymin
        bboxs[:, 2] -= xmin
        bboxs[:, 3] -= ymin
        return im, bboxs, masks, tags

    return im, bboxs, masks, tags


def cls_target(img_shape, bboxes, gt_class_ids):
    """

    :param img_shape:
    :param bboxes:
    :param gt_class_ids:
    :return:
    """
    # 返回值是[批数，anchor数，(x1, y1, x2, y2)]，相对输入图片的像素坐标
    anchors = generate_pyramid_anchors(batch_size, resolution, input_shape, smallest_anchor_size)
    all_anchors = anchors[0]  # 只需要取第一批, [num, (x1, y1, x2, y2)]

    # 在图片里面
    inside = (
            (all_anchors[:, 0] >= -allowed_border) &
            (all_anchors[:, 1] >= -allowed_border) &
            (all_anchors[:, 2] < img_shape[1] + allowed_border) &
            (all_anchors[:, 3] < img_shape[0] + allowed_border)
    )

    num_anchors = all_anchors.shape[0]

    rpn_labels = np.empty(shape=(num_anchors,), dtype=np.int32)
    rpn_labels.fill(-1)
    anchor_deltas = np.empty(shape=(num_anchors, 4), dtype=np.float32)

    # 有的bounding box可能框住了多个实例，标签就是-1
    crowd_ix = np.where(gt_class_ids < 0)[0]
    if crowd_ix.shape[0] > 0:
        non_crowd_ix = np.where(gt_class_ids > 0)[0]
        crowd_boxes = bboxes[crowd_ix]

        gt_boxes = bboxes[non_crowd_ix]
        # 计算anchor与crowd的iou,如果与crowd的iou过大，那这个anchor不进行训练
        crowd_overlaps = bbox_overlaps(np.ascontiguousarray(all_anchors, dtype=np.float),
                                       np.ascontiguousarray(crowd_boxes, dtype=np.float))
        crowd_iou_max = np.amax(crowd_overlaps, axis=1) # 长度是所有anchor的个数
        no_crowd_bool = (crowd_iou_max < 0.001)
    else:
        no_crowd_bool = np.ones(shape=(num_anchors,), dtype=bool)
        gt_boxes = bboxes

    if gt_boxes.shape[0] > 0:
        overlaps = bbox_overlaps(
            np.ascontiguousarray(all_anchors, dtype=np.float),
            np.ascontiguousarray(gt_boxes, dtype=np.float))

        argmax_overlaps = overlaps.argmax(axis=1)  # 长度为num_anchors


        max_overlaps = overlaps[np.arange(num_anchors,), argmax_overlaps]

        # 将iou小于0.3并且没有与crowd相交的，设置为0，表示负例
        rpn_labels[(max_overlaps < neg_anchor_thresh) & no_crowd_bool & inside] = 0

        rpn_labels[(max_overlaps>=posi_anchor_thresh) & inside] = 1

        # 对于某个GT而言，即使所有anchor与他的iou都小于0.3，也需要把与之iou最大的那个设置为正例
        gt_iou_argmax = np.argmax(overlaps, axis=0)
        rpn_labels[gt_iou_argmax] = 1


        pos_ids = np.where(rpn_labels == 1)[0]

        # 不能让正例超过一半
        extra = len(pos_ids) - RPN_TRAIN_ANCHORS_PER_IMAGE//2
        if extra > 0:
            rpn_labels[np.random.choice(pos_ids, extra, replace=False)] = -1
            pos_ids = np.where(rpn_labels == 1)[0]
        pos_anchor = all_anchors[pos_ids]
        for i, a in zip(pos_ids, pos_anchor):
            gt = gt_boxes[argmax_overlaps[i]]
            gt_h = gt[3] - gt[1]
            gt_w = gt[2] - gt[0]
            gt_ctr_x = gt[0] + 0.5 * gt_w
            gt_ctr_y = gt[1] + 0.5 * gt_h

            an_h = a[3] - a[1]
            an_w = a[2] - a[0]
            an_ctr_x = a[0] + 0.5 * an_w
            an_ctr_y = a[1] + 0.5 * an_h

            anchor_deltas[i] = [(gt_ctr_x-an_ctr_x)/an_w, (gt_ctr_y-an_ctr_y)/an_h,
                           np.log(gt_h/an_h), np.log(gt_w/an_w)]
            anchor_deltas[i] /= RPN_BBOX_STD_DEV

        neg_ids = np.where(rpn_labels == 0)[0]
        extra = len(neg_ids) - (RPN_TRAIN_ANCHORS_PER_IMAGE - len(pos_ids))
        if extra > 0:
            rpn_labels[np.random.choice(neg_ids, extra, replace=False)] = -1
            # neg_ids = np.where(rpn_labels == 0)[0]
    else:
        rpn_labels[np.random.choice(num_anchors, RPN_TRAIN_ANCHORS_PER_IMAGE,replace=False)] = 0

    return rpn_labels, anchor_deltas


def mask_target(img_shape, segmentations, bbox):
    """
    :param img_shape: 图片高宽
    :param segmentations: 一个长度为gt个数的列表，列表的每个元素是一个多边形，相对于原始图片的像素坐标
    :param bbox: 长度为gt个数的数组，每一行表示一个框框(x1, y1, x2, y2)，像素坐标
    :return: numpy数组，(gt个数，gao，宽)
    """
    # 不使用MINI mask
    h, w = img_shape[:2]
    length = len(segmentations)
    segmentation_mask = np.zeros((length, h, w), dtype=np.float32)
    min_mask = np.zeros(shape=(length,)+MINI_MASK_SHAPE, dtype=np.int32)
    for i in range(length):
        seg = np.array(segmentations[i], dtype=np.int32)
        cv2.fillPoly(segmentation_mask[i], [seg], 1)
        x1, y1, x2, y2 = bbox[i]
        mini_seg = segmentation_mask[i,y1:y2, x1:x2]
        min_mask[i] = cv2.resize(mini_seg, dsize=MINI_MASK_SHAPE)

    if USE_MINI_MASK:
        return min_mask
    else:
        return segmentation_mask

