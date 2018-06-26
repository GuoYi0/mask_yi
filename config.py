
# -*- coding:utf-8 -*-
import tensorflow as tf
import numpy as np
import os

# common
dataDir = '.'
train_type = 'train2017'
trainImage_path = '{}/cocodata2/images'.format(dataDir)
train_annFile = '{}/cocodata2/annotations/instances_{}.json'.format(dataDir,train_type)
checkpoint_path = 'checkpoints'
summary_path = "summary"
result_pic = 'result_pic' # 输出图片的地址
COCO_WEIGHTS_PATH = os.path.join('pre_trained_weights', "mask_rcnn_coco.h5")  # 预训练模型所在地址
MASK_THRESH = 0.5 # mask的阈值，在测试时候对于输出的mask，大于该阈值的判为True，否则为False
restore = False # 是否接着训练
NUM_TEST_IMAGE = 100 # 测试图片的张数
max_steps = 100000
save_checkpoint_steps = 100 # 每100步保存一下ckeckpoints
save_summary_steps = 100 # 每100步保存一下summary
learning_rate = 0.001
DTYPE = tf.float32
moving_average_decay = 0.9
# train
input_shape = (1024, 1024) # （height，width）
smallest_anchor_size = 32  # 最底层的feature map所要框住的目标大小,有五层，最高层的则是512
random_resize = [0.5, 1, 1.5, 2]
pixel_average = [102.9801, 115.9465, 122.7717]  # BGR顺序
feat_strides = np.array([128, 64, 32, 16, 8])
resolution = np.array([8, 16, 32, 64, 128])
positive_ratio = 0.25
posi_anchor_thresh = 0.7  # anchor 大于为正值
neg_anchor_thresh = 0.3  # anchor 小于为负值
batch_anchor_num = 128  # 每轮正值和负值总数
batch_size = 1
anchor_ratios = (0.5, 1.0, 2.0)
anchor_scales = [64, 128, 256]
anchor_per_location = 3
allowed_border = 4  # 只有在图片里面的anchor才进行训练，但由于图片可能有填充， 允许超出边界的像素个数
# 非极大值抑制以后，保留的roi的个数 (training and inference)
POST_NMS_ROIS_TRAINING = 2000  # 选出分数最高的这些anchor来生成的proposal，非极大值抑制之前
MAX_PROPOSAL_TO_DETECT = 1000  # 要检测的正例proposal，一般小于上面的数POST_NMS_ROIS_TRAINING
POST_NMS_ROIS_INFERENCE = 1000

RPN_BBOX_STD_DEV = np.array([0.1, 0.1, 0.2, 0.2], dtype=np.float32)
RPN_NMS_THRESHOLD = 0.7  # 非极大值抑制的iou阈值
USE_MINI_MASK = True # 使用迷你mask，以节省内存。mini mask的高宽与gt bounding box的高宽对应
MINI_MASK_SHAPE = (56, 56)  # (height, width) of the mini-mask
#
IMAGE_SHAPE = input_shape + (3,)  # 输入的图片的裁剪大小,通道是3

# 喂给网络的roi个数，原文里面是512，并保证正负样本比例1:3。但是有时候没有足够的正样本数来保证，故暂且取200
# 可以把proposal的nms的阈值调高，使得这个数可以适当调高
TRAIN_ROIS_PER_IMAGE = 300

# 每张图片训练多少个anchors
RPN_TRAIN_ANCHORS_PER_IMAGE = 256
# 正例样本在总样本中的占比，正例有很多不同类别，而这里占比这么小，可能会有问题
ROI_POSITIVE_RATIO = 0.3333
# 根据论文图4右边，mask的大小为 28*28
MASK_SHAPE = (28, 28)  # 必须是下面的两倍
MASK_POOL_SIZE = (14, 14) # 必须是上面的一半
POOL_SIZE = (7, 7)  # ROI Pooling层的大小，一般是7*7
NUM_CLASSES = 90 + 1  # 图片分为多少个类别。80类正例加一类背景。一般来说，我建议先分为object/non-object，
# 然后对于object再分为80类，而不是全部混在一起进行分类

DETECTION_MIN_CONFIDENCE = 0.7  # 属于某一类别的置信度阈值
DETECTION_MAX_INSTANCE = 100  # 每一张图片里面，最多检测出的instance个数
DETECTION_NMS_THRESHHOLD = 0.3  # 同类别的检测的非极大值抑制阈值
BACKBONE_STRIDES = []
CROP_AUGMENTATION = False  # 是否通过裁剪来进行数据增强

IMAGE_MIN_DIM = 800
IMAGE_MIN_SCALE = 0
IMAGE_MAX_DIM = 1024
IMAGE_RESIZE_MODE = "square"
MAX_GT_INSTANCES = 100
dataset_dir = os.path.join(dataDir, "cocodata2")
subset = "train"
year = "2017"

if __name__ == '__main__':
    print(os.getcwd())