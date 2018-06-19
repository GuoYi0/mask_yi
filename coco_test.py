
# -*- coding:utf-8 -*-
import time

import cv2
import numpy as np
import tensorflow as tf
import os
from input.producer import producer
import config
from model import MASK_RCNN
from visualize import apply_box_mask

def ummode(boxes, ids, mask, image_shape):
    """
    归一化坐标，转换为像素坐标；mask也转化为图片大小的mask
    :param boxes: [批数，num，4]，归一化坐标
    :param ids: [num]
    :param mask: [num, 28, 28]
    :param image_shape: 二元组 (高，宽)
    :return: boxes, [num, 4], int, 像素坐标
              mask， [num, 高，宽]，其高宽与输入的image_shape一致
    """
    boxes = np.squeeze(boxes, axis=0) #[num, 4]
    num = len(ids)
    height, width = image_shape
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    x1 = round(x1 * width)
    y1 = round(y1 * height)
    x2 = round(x2 * width)
    y2 = round(y2 * height)
    box_width = x2 - x1 + 1
    box_height = y2 - y1 + 1
    boxes = np.stack([x1, y1, x2, y2], axis=1)
    full_mask = []
    for i in range(num):
        temp_mask = cv2.resize(mask[i], (box_width[i], box_height[i]))
        temp_mask = np.where(temp_mask >= config.MASK_THRESH, 1, 0).astype(np.bool)
        full_pic = np.zeros(image_shape, dtype=np.bool)
        full_pic[y1:(y2+1), x1:(x2+1)] = temp_mask
        full_mask.append(full_pic)

    full_mask = np.stack(full_mask, axis=0)

    return boxes, full_mask





def main(_):
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    if not tf.gfile.Exists(config.result_pic):
        tf.gfile.MakeDirs(config.result_pic)
    else:
        tf.gfile.DeleteRecursively(config.result_pic)
    # 输入图片，经过去均值和缩放以后的

    with tf.get_default_graph().as_default():
        # 输入图片 [批数，高，宽，通道数]
        input_images = tf.placeholder(dtype=tf.float32, shape=(1, None, None, 3), name='input_images')
        # anchor坐标，归一化坐标 [批数，个数，(x1, y1, x2, y2)]
        anchors = tf.placeholder(dtype=tf.float32, shape=(1, None, 4), name='anchors')
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        mask_rcnn = MASK_RCNN()
        # 这里返回的，boxes [批数，个数，4]； ids [num]； probs [num]；mask [num, 28, 28]
        boxes, ids, probs, mask = mask_rcnn.build_model('inference',input_images)

        variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
        saver = tf.train.Saver(variable_averages.variables_to_restore())

        with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

            ckpt = tf.train.get_checkpoint_state(config.checkpoint_path)
            if ckpt and ckpt.model_checkpoint_path:
                print('Restoring from {}...'.format(ckpt.model_checkpoint_path), end=' ')
                try:
                    saver.restore(sess, ckpt.model_checkpoint_path)
                except:
                    raise 'Check your pretrained {:s}'.format(ckpt.model_checkpoint_path)
                print('restore done')
            else:
                raise 'Check your pretrained {}'.format(config.checkpoint_path)

            for _ in range(config.NUM_TEST_IMAGE):
                start_time = time.time()
                timer = {'net': 0, 'draw': 0}

                data = [None, None, None ,None, 'pic.jpg']
                # data[0]是缩放和去均值化以后的图片
                # data[1]是对应的anchor
                # data[2]是缩放以前，最原始的输入图片的大小
                # data[3]是最原始的输入图片，没有做任何处理的图片
                # data[4]是图片名字
                boxes_run, ids_run, probs_run, mask_run = sess.run([boxes, ids, probs, mask],
                                                                   feed_dict={input_images: data[0], anchors: data[1]})

                start_time2 = time.time()
                timer['net'] = start_time2 - start_time

                boxes, masks = ummode(boxes_run, ids_run, mask_run, image_shape=data[2])
                image= apply_box_mask(image=data[3].copy(), box=boxes, mask=masks, ids=ids_run, num_class=config.NUM_CLASSES)
                cv2.imwrite(os.path.join(config.result_pic, data[4]), image)
                timer['draw'] = time.time() - start_time2
                print('pic {} took {:.0f}ms for detection, {:.0f}ms for drawing, total time is {:.0f}ms'.format(
                    data[4],timer['net'], timer['draw'], time.time()-start_time))

    print("test done!")

if __name__ == '__main__':
    tf.app.run(main)
