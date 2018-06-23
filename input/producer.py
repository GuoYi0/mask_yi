import os
import cv2
import numpy as np
import tensorflow as tf
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils
from config import train_annFile
from config import batch_size
from config import trainImage_path
from config import pixel_average
from input.preprocess import data_augmentation,cls_target,mask_target, resize
from lib.gen_box import norm_boxes
from visualize import apply_box_mask
from config import input_shape
import config
import shutil







# coco = COCO(train_annFile)
def producer():
    # iscrowd: 0 segmentation is polygon
    # iscrowd: 1 segmentation is RLE



    # 以列表的形式返回imageID
    imgIds = coco.getImgIds()

    imgIds_list = tf.constant(imgIds)
    # class_ids = sorted(coco.getCatIds())
    # print(class_ids," 000000")

    dataset = tf.data.Dataset.from_tensor_slices((imgIds_list,))
    dataset = dataset.map(
        lambda imgId: tuple(tf.py_func(
            #                          img_name   image,      gt_box,    gt_class,  mask, anchor_labels, anchor_deltas
            sample_handler, [imgId], [tf.string, tf.float32, tf.float32, tf.int32, tf.bool, tf.int32,    tf.float32])),
        num_parallel_calls=1).repeat().batch(batch_size)
    return dataset




def sample_handler(imgId=532481):
    """
    :return: image，input_shape, 进行裁剪以后的图片，float
              gt_box，numpy数组，[N, (x1, y1, x2, y2)] normalized坐标, N表示gt个数，没有对标签进行筛选
              segmentation_mask，numpu数组，(N，gao，宽)
              anchor_labels numpy数组， [num_anchor] 所有anchor的标签，1表示正例，0表示负例，-1不予训练
              anchor_deltas numpy数组。[num_anchor, (dx, dy, log(h), log(w))], 除以了RPN_BBOX_STD_DEV
    """
    # 下面这个函数是以列表形式返回的，所以要取[0]
    img_info = coco.loadImgs(int(imgId))[0]
    img_name = img_info['file_name']
    height, width = img_info['height'], img_info['width']

    img_path = os.path.join(trainImage_path, img_name)
    image = cv2.imread(img_path)
    # if image is None:
    #     print("No image to read in path {}".format(img_path))
    #     raise
    image = np.array(image, dtype=np.float32)
    image -= pixel_average  # 去均值

    annIds = coco.getAnnIds(imgIds=imgId)
    # anns是一个长度为N的列表，代表N个box，每个box是一个字典，
    # 包括'segmentation','area', 'iscrowd', 'image_id', 'bbox', 'category_id'
    anns = coco.loadAnns(annIds)
    # 定义三个列表，其长度是实例个数，
    # instance_masks，mask，一个高宽就是图片宽高的的bool型mask，binary mask (numpy 2D array)
    # class_ids是一个int型列表
    # bboxes的每个元素依然是列表，元素列表包含四个元素，x1, y1, x2, y2
    segmentations = []
    class_ids = []
    bboxes = []
    for idx, ann in enumerate(anns):
        # 获取类别ID
        class_id = ann['category_id']
        if class_id: # 不是背景
            # 根据标注取出mask
            m = coco.annToMask(ann)
            if m.max() < 1:
                continue
            if ann["iscrowd"]:
                class_id *= -1
                if m.shape[0] != height or m.shape[1] != width:
                    m = np.ones([height, width], dtype=bool)

            bbox = np.array(ann['bbox'])
            # left top x, left top y, width, height -> xmin, ymin, xmax, ymax
            bbox_n = [bbox[0],
                      bbox[1],
                      bbox[0] + bbox[2],
                      bbox[1] + bbox[3]]
            bboxes.append(bbox_n)
            segmentations.append(m)
            class_ids.append(class_id)

    raw_size = np.array((height, width), dtype=np.float32)

    bboxes = np.array(bboxes,dtype=np.float32)
    class_ids = np.array(class_ids,dtype=np.int32)
    if config.CROP_AUGMENTATION:
        image, bboxes, class_ids, segmentations = data_augmentation(image, raw_size, bboxes, class_ids, segmentations)

    image, bboxes, segmentations = resize(image, bboxes, segmentations)

    # 所有anchor的标签，以及回归值
    anchor_labels, anchor_deltas = cls_target(image.shape, bboxes, class_ids)

    # segmentation_mask = np.round(mask_target(image.shape, segmentations, bboxs)).astype(np.bool)

    segmentations = [np.where(seg > 0.5, 1, 0).astype(np.bool) for seg in segmentations]
    segmentations = np.array(segmentations,dtype=np.bool)

    gt_box = norm_boxes(bboxes, image.shape)

    return img_name, image, gt_box, class_ids, segmentations, anchor_labels, anchor_deltas










def _ummode(boxes, ids, mask, image_shape):
    """
    归一化坐标，转换为像素坐标；mask也转化为图片大小的mask
    :param boxes: [num，4]，归一化坐标
    :param ids: [num]
    :param mask: [num, 28, 28]
    :param image_shape: 二元组 (高，宽)
    :return: boxes, [num, 4], int, 像素坐标
              mask， [num, 高，宽]，其高宽与输入的image_shape一致
    """
    num = len(ids)
    if num > 0:
        mask = mask.astype(np.float32)
        assert boxes.shape[0] == ids.shape[0] == mask.shape[0], "{}, {}, {}".format(boxes.shape, ids.shape, mask.shape)

        height, width = image_shape[0:2]
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        x1 = np.clip(np.round(x1 * width),0,width-1).astype(np.int32)
        y1 = np.clip(np.round(y1 * height), 0,height-1).astype(np.int32)
        x2 = np.clip(np.round(x2 * width),0,width-1).astype(np.int32)
        y2 = np.clip(np.round(y2 * height), 0,height-1).astype(np.int32)
        box_width = x2 - x1 + 1
        box_height = y2 - y1 + 1
        boxes = np.stack([x1, y1, x2, y2], axis=1)
        full_mask = []
        for i in range(num):
            temp_mask = cv2.resize(mask[i], (box_width[i], box_height[i]))
            temp_mask = np.where(temp_mask >= config.MASK_THRESH, 1, 0).astype(np.bool)
            full_pic = np.zeros(image_shape, dtype=np.bool)
            full_pic[y1[i]:(y2[i]+1), x1[i]:(x2[i]+1)] = temp_mask
            full_mask.append(full_pic)

        mask = np.stack(full_mask, axis=0)
    else:
        boxes = boxes.astype(np.int32)

    return boxes, mask


if __name__ == '__main__':
    # data = producer()
    # print(data)
    originimage = "../image_test/originImage"
    ann_image = '../image_test/ann_image'
    if os.path.exists(originimage):
        shutil.rmtree(originimage)
    os.makedirs(originimage)

    if os.path.exists(ann_image):
        shutil.rmtree(ann_image)
    os.makedirs(ann_image)


    next_batch = producer().make_one_shot_iterator().get_next()
    with tf.Session() as sess:
        for _ in range(50):
            img_name2,image2, gt_box2, gt_class2, segmentation_mask2, anchor_labels2, anchor_deltas_in2 = sess.run(next_batch)
            image2 = np.squeeze(image2,0)
            image2 += config.pixel_average

            img_name2 = img_name2[0].decode()
            cv2.imwrite(os.path.join(originimage,img_name2),image2)
            gt_box2 = np.squeeze(gt_box2, 0)
            gt_class2 = np.squeeze(gt_class2, 0)
            segmentation_mask2 = np.squeeze(segmentation_mask2, 0)
            anchor_labels2 = np.squeeze(anchor_labels2, 0)
            anchor_deltas_in2 = np.squeeze(anchor_deltas_in2, 0)
            # boxes, ids, mask, image_shape
            boxes2, masks = _ummode(gt_box2, gt_class2, segmentation_mask2, image_shape=input_shape)
            image3 = apply_box_mask(image=image2.copy(), box=boxes2, mask=masks, ids=gt_class2, num_class=config.NUM_CLASSES)
            cv2.imwrite(os.path.join(ann_image,img_name2), image3)



