
import tensorflow as tf
import numpy as np
#
# input = np.random.random_integers(0,24,24).reshape([2,3,4])
# k = 3
# output = tf.nn.top_k(input[:, :, 1], k, sorted=True)
# xi = output.indices
# with tf.Session() as sess:
#     print(input)
#     print("===================")
#     print(input[:, :, 1])
#     print("================")
#     print(sess.run(xi))
#     print("================")
#     print(sess.run(tf.gather(input[:, :, 1], xi)))

# value = np.arange(24).reshape([6,4])
# print(value)
# value2 = tf.gather(value, [0,1],axis=1)
# with tf.Session() as sess:
#
#     print(sess.run(value2))

# input = np.random.random_integers(0,24,24).reshape([2,3,4])
# out = tf.expand_dims(input, -1)
# with tf.Session() as sess:
#     print(input.shape)
#     print(sess.run(out).shape)
#
# a, b = [],[]
# print(a)
# print(b)
#!/usr/bin/python

#
# import matplotlib.pyplot as plt
# # import tensorflow as tf
# # 读取图像数据
# img = tf.gfile.FastGFile('tu2.jpg', 'rb').read()
#
# with tf.Session() as sess:
#     img_data = tf.image.decode_jpeg(img,channels=3)
#     # tf.image.draw_bounding_boxes要求图像矩阵中的数字为实数
#     # 利用tf.image.convert_image_dtype将图像矩阵转化为实数
#
#     batched = tf.expand_dims(tf.image.convert_image_dtype(img_data, tf.float32), 0)
#     batched = tf.image.resize_images(batched, (512, 512))
#     # 边界框坐标是相对于宽度和宽度在[0.0，1.0]内的浮点数，即这里给出的都是图像的相对位置[0.1, 0.2, 0.8, 0.8]即（0.1*wide, 0.2*high）到（0.8*wide, 0.8*high）
#     boxes = tf.constant([[[0.01, 0.2, 0.5, 0.5]]])
#     # 在图像上绘制边界框
#     result = tf.image.draw_bounding_boxes(batched, boxes)
#     boxes = tf.constant([[0.01, 0.2, 0.5, 0.5]])
#     result2 = tf.image.crop_and_resize(batched, boxes, tf.range(0, 1), (256, 256))
#
#     plt.subplot(121), plt.imshow(batched[0].eval()), plt.title('original')
#     plt.subplot(122), plt.imshow(result2[0].eval()), plt.title('result')
#
#     plt.show()

# a = tf.constant([[1, 2], [3, 4]])
# a1, a2 = tf.split(a, 2, 1)
# with tf.Session() as sess:
#     print(sess.run(a))
#     print(sess.run(a1))
#     print(sess.run(a2))

# a = np.array([[1, 2, 3, 4, 5], [3, 4, 8, 2 ,4]])
# c = (a % 2 == 0)
# ix = tf.where(c)
#
# b = np.array([[[1, 2, 3, 4, 5], [3, 4, 8, 2 ,4]],[[1, 2, 3, 4, 5], [3, 4, 8, 2 ,4]]])
# with tf.Session() as sess:
#     print(sess.run(ix).shape)
#     print(sess.run(ix))
#     print(sess.run(ix)[:,0])
    # print(sess.run(tf.gather_nd(b, ix)))

# import tensorflow as tf
#
# temp = tf.constant([[1,2,3],[4,5,6], [7, 8, 9]])
# temp2 = tf.argmax(temp,axis=1)
#
# with tf.Session() as sess:
#
#     print(sess.run(temp))
#     print(sess.run(temp2))

# a = np.arange(12).reshape((3, 4))
# b = np.array([1,2,3])
# index = tf.where(tf.not_equal(b, 3))
# index2 = tf.where(tf.not_equal(b, 3))[:,0]
# gather = tf.gather(a, index)
# gather2 = tf.gather(a, index2)
# with tf.Session() as sess:
#     print(a)
#     print("==================")
#     print(sess.run(gather))
#     print("==================")
#     print(sess.run(gather2))
#     print("==================")
#     print(sess.run(index))
#     print("==================")
#     print(sess.run(index2))
#     print("==================")
#     print(sess.run(tf.gather_nd(a,index)))

# import tensorflow as tf

# temp = tf.constant([2,4,6,7])
# temp2 = tf.constant([2,7,8,9])
# section = tf.sets.set_intersection(tf.expand_dims(temp, 0), tf.expand_dims(temp2, 0))
# sparse = tf.sparse_tensor_to_dense(section)
#
# with tf.Session() as sess:
#
#     print(sess.run(temp))
#     print(sess.run(temp2))
#     print(sess.run(section))
#     print(sess.run(sparse))

# temp = tf.constant([2,4,6,2])
#
#
# with tf.Session() as sess:
#
#     print(sess.run(tf.unique(temp))[0])
# def gt():
#     return [34, 123]
#
# a, b = gt()
# print(a, b)
import cv2
# image = np.ones((3,4,3))
# image2 = cv2.resize(image, (10, 11))
# print(image.shape)
# print(image2.shape)
import h5py

# # HDF5的写入：
# imgData = np.zeros((2, 4))
# f = h5py.File('HDF5_FILE.h5', 'w')  # 创建一个h5文件，文件指针是f
# f['data'] = imgData  # 将数据写入文件的主键data下面
# f['labels'] = np.array([1, 2, 3, 4, 5])  # 将数据写入文件的主键labels下面
# f.close()  # 关闭文件

# # HDF5的读取：
# f = h5py.File('E:\mask\pre_trained_weights\mask_rcnn_coco.h5', 'r')  # 打开h5文件
# for key in f.keys():
#     print("++++++++++++++",key)
#     for subkey in f[key]:
#         for ssubkey in f[key][subkey]:
#             print("==============",ssubkey)
#             print()
#     print()
import os
# import numpy as np
# import tensorflow as tf
# from .preprocess import parse_function


# from pycocotools.coco import COCO
# from config import train_annFile
# from config import batch_size
#
#
# coco = COCO(train_annFile)
#
# imgIds = coco.getImgIds()
# img_info = coco.loadImgs(int(imgIds[0]))[0]
# annids = coco.getAnnIds(imgIds=imgIds[0])
# anns = coco.loadAnns(annids)
# print(len(anns))
# print(anns[0])

# a = (1,2)
# b = (3,)
# print(a+b)

# a = np.array([1,2])
# # b = np.repeat(np.expand_dims(a,0), 2, 0)
#
# # b = np.broadcast_to(a, (3,)+a.shape)
# # print(b)
# #
# # b = np.arange(12).reshape((6, 2))
# # print(b)
# # print(b-a)
#
# x,y=a
# print(x)
#
# segmentation_mask = np.array([[0.6,0.3],[0.5,0.7]])
# segmentation_mask = np.where(segmentation_mask > 0.5,1,0)
# print(type(segmentation_mask[0,0]))
# print(" ")
# b = np.round(segmentation_mask).astype(np.bool)
# print(b)
import cv2
# mask = np.zeros((6, 5), dtype=np.bool)
# gg = mask.astype(np.float32)
# gg[0:3, 3:4] = 2.0
# print(gg)
# index = np.argwhere(gg>0)
# print("=========")
# print(index)
# print(min(1,2,3))

# def return_zero():
#     return 1
#
# def add():
#     return 2
#
# a = tf.constant(2)
#
# b = tf.cond(a>0, return_zero, add)
#
# with tf.Session() as sess:
#     print(sess.run(b))

# a = np.array([[3,2,4,1]])
# b = tf.nn.top_k(a, 2, sorted=True).indices
# b=b[0]
# print(a)
# print("======")
# with tf.Session() as sess:
#     print(sess.run(b))



a = tf.constant([3,8])
b = tf.where(a>10)[:,0]
gg = tf.gather(a, b)
positive_indices = tf.random_shuffle(b)[:0]

with tf.Session() as sess:
    print(sess.run(positive_indices))

