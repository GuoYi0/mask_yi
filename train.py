import cv2
import tensorflow as tf
# import tensorlayer as tl
import numpy as np
import os
import math
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from input.producer import producer
from net import backbone
from resnet.resnet50 import Model

from config import batch_size

# from config import batch_size

if __name__ == '__main__':
    next_batch = producer().make_one_shot_iterator().get_next()

    imgPH = tf.placeholder(tf.float32, shape=(batch_size, None, None, 3))

    # net_in = tl.layers.InputLayer(imgPH, name='input_layer')
    # ret = backbone(net_in).outputs

    resnet = Model([[3, 256], [4, 512], [6, 1024], [3, 2048]])

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(1):
            image = sess.run(next_batch)
            print(image[0].shape)

            # fp, resolution = resnet(image[0], True)
            # print(fp[-1])

            # fm = sess.run([ret],feed_dict={imgPH: image[0]})
            # print(fm[0].shape)


