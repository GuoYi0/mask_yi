import sys
from resnet.resnet50 import conv2d, Model, dense, batch_norm, conv2d_transpose
import tensorflow as tf
import config
from lib.gen_box import generate_pyramid_anchors
resnet50  = [[3, 256], [4, 512], [6, 1024],  [3, 2048]]
resnet101 = [[3, 256], [4, 512], [23, 1024], [3, 2048]]
resnet152 = [[3, 256], [8, 512], [36, 1024], [3, 2048]]


def rpn_graph(feature_map, anchor_per_location,anchor_stride, name=None):
    """
    根据特征图建立RPN网络的计算图,对应网络的输出
    :param feature_map:  特征图，形状为[批，高，宽，通道数]
    :param anchor_per_location:  int，每个像素点产生多少个anchor
    :param anchor_stride: 一般取1，表示特征图上，每个点都产生anchor
    :return: 一个列表，有三个元素，依次是anchor的logits，probs， bbox回归
    """
    batch_size, height, width, channal = feature_map.shape
    # num_anchor = height*width*anchor_per_location // anchor_stride  # 这张特征图一共可以产生num_anchor个anchor
    shared = conv2d(feature_map, out_channal=512, kernel_size=3, strides=anchor_stride, name='rpn_conv_shared'+name)
    shared = tf.nn.relu(shared)
    # out_channal=2 * anchor_per_location,区分是或者不是物体
    x = conv2d(inputs=shared,out_channal=2 * anchor_per_location,kernel_size=1,strides=1, name='rpn_class_raw'+name)
    # 把形状调整为[批数，anchor数，2]， 2 表示object/non-object
    # rpn_binary_logits = tf.reshape(x, shape=[batch_size, num_anchor, 2])
    rpn_binary_logits = tf.reshape(x, [batch_size, -1, 2])


    rpn_probs = tf.nn.softmax(rpn_binary_logits)

    x = conv2d(inputs=shared, out_channal=4*anchor_per_location, kernel_size=1,strides=1, name='rpn_bbox_pred'+name)
    # 坐标回归
    rpn_bbox = tf.reshape(x, [batch_size, -1, 4])

    return [rpn_binary_logits, rpn_probs, rpn_bbox]

def apply_box_deltas(boxes, delta):
    """
    对boxes进行修正处理
    x坐标的回归 = (gt的x - anchor的x) / anchor的宽
    高度的回归 = log(gt的高/anchor的高)
    :param boxes: [..，anchor数，4]， （x1, y1, x2, y2）
    :param delta: [..，anchor数，4], (dx, dy, log(dh), log(dw))
    :return:
    """
    shape = tf.shape(boxes)
    with tf.control_dependencies([tf.Assert(tf.equal(shape[1],tf.shape(delta)[1]),
                                            data=["shape must be same", shape, tf.shape(delta)])]):
        boxes_reshaped = tf.reshape(boxes, [-1, 4])  # [num_box, 4]
        delta_reshaped = tf.reshape(delta, [-1, 4])  # [num_box, 4]
    height = boxes_reshaped[:, 3] - boxes_reshaped[:, 1]
    width = boxes_reshaped[:, 2] - boxes_reshaped[:, 0]
    center_x = (boxes_reshaped[:, 0] + boxes_reshaped[:, 2]) / 2
    center_y = (boxes_reshaped[:, 3] + boxes_reshaped[:, 1]) / 2

    # 修正后的中心点xy坐标
    pred_x = delta_reshaped[:, 0] * width + center_x
    pred_y = delta_reshaped[:, 1] * height + center_y

    # 修正后的高度和宽度
    pred_width = tf.exp(delta_reshaped[:, 3]) * width
    pred_height = tf.exp(delta_reshaped[:, 2]) * height

    x1 = pred_x - pred_width/2
    y1 = pred_y - pred_height/2

    x2 = pred_x + pred_width/2
    y2 = pred_y + pred_height/2
    # 拼接起来返回
    result = tf.stack([x1, y1, x2, y2], axis=1)
    return tf.reshape(result,shape)

def nms(boxes, scores, max_count, thresh):
    """
    :param boxes: [批数，个数，4]
    :param scores: [批数， 个数]
    :param max_count:  最多输出的盒子个数
    :param thresh:  iou阈值
    :return: [批数，个数，4]
    """
    asserts = tf.Assert(tf.equal(tf.shape(boxes)[1],tf.shape(scores)[1]), [tf.shape(boxes), tf.shape(scores)])
    with tf.control_dependencies([asserts]):
        boxes = tf.identity(boxes)
    out = []
    batch_size = boxes.shape[0]
    for i in range(batch_size):

        selected_indices = tf.image.non_max_suppression(boxes[i], scores[i], max_count,thresh)

        out.append(tf.gather(boxes[i], selected_indices))
    return tf.stack(out, axis=0)

def overlaps_graph(boxes1, boxes2):
    """
    :param boxes1: [M, (x1, y1, x2, y2)]
    :param boxes2: [N, (x1, y1, x2, y2)]
    :return: [M, N]的iou矩阵
    """

    b1 = tf.reshape(tf.tile(tf.expand_dims(boxes1, 1),
                            [1, 1, tf.shape(boxes2)[0]]), [-1, 4])
    b2 = tf.tile(boxes2, [tf.shape(boxes1)[0], 1])
    # 计算交集
    b1_x1, b1_y1, b1_x2, b1_y2 = tf.split(b1, 4, axis=1)
    b2_x1, b2_y1, b2_x2, b2_y2 = tf.split(b2, 4, axis=1)
    y1 = tf.maximum(b1_y1, b2_y1)
    x1 = tf.maximum(b1_x1, b2_x1)
    y2 = tf.minimum(b1_y2, b2_y2)
    x2 = tf.minimum(b1_x2, b2_x2)
    intersection = tf.maximum(x2 - x1, 0) * tf.maximum(y2 - y1, 0)
    # 计算并集
    b1_area = (b1_y2 - b1_y1) * (b1_x2 - b1_x1)
    b2_area = (b2_y2 - b2_y1) * (b2_x2 - b2_x1)
    union = b1_area + b2_area - intersection
    # 计算iou
    iou = intersection / union
    overlaps = tf.reshape(iou, [tf.shape(boxes1)[0], tf.shape(boxes2)[0]])
    return overlaps


def detection_targets(proposals, gt_class_ids, gt_boxes, gt_masks):
    """
    一批一批地处理，该函数只能处理一批
    :param proposals: 一张图片对应的许多proposal，形状是[N, (x1, y1, x2, y2)] ，归一化坐标
    :param gt_class_ids: 形状是 [MAX_GT_INSTANCES]，即，其长度等于每张图片的实例个数，一般不等于N
    :param gt_boxes: [MAX_GT_INSTANCES, (x1, y1, x2, y2)]
    :param gt_masks: 形状是[ MAX_GT_INSTANCES，高, 宽]， bool值，每个instance都要有一个mask，与gt_boxes一一对应
            如果不使用MINI_MASK，那这里的高宽就是图片的高宽，否则是MINI_MASK的高宽
    :return proposals, [T, (x1, y1, x2, y2)];
            roi_gt_class_ids, [T];
            gt_deltas, [T, 4]; 只有正例的gt_deltas有效，其他的用零填充
            masks,[T, 高，宽]；只有正例的mask有效，其他的用零填充
    其中T=config.TRAIN_ROIS_PER_IMAGE，即每张图片的训练proposal个数，
    """

    # 保证至少有一个proposal
    asserts = [tf.Assert(tf.greater(tf.shape(proposals)[0], 0), [proposals])]
    with tf.control_dependencies(asserts):
        proposals = tf.identity(proposals)

    non_crowd_ix = tf.where(gt_class_ids > 0)[:, 0]

    # 把只有一个实例的框框、对应的id、mask选出来
    gt_boxes = tf.gather(gt_boxes, non_crowd_ix)
    gt_class_ids = tf.gather(gt_class_ids ,non_crowd_ix)
    gt_masks = tf.gather(gt_masks, non_crowd_ix)

    # 计算proposals和gt_boxes之间的iou。返回的shape是[proposals.shape[0], gt_boxes.shape[0]]
    overlaps = overlaps_graph(proposals, gt_boxes)
    roi_iou_max = tf.reduce_max(overlaps, axis=1)
    # 这里正例的角标，是相对于所有的全部proposal个数N的角标
    positive_indices = tf.where(roi_iou_max >= config.posi_anchor_thresh)[:, 0]

    # 在coco数据里面，可能一个bounding box框住了好多个实例，我们就把用他的标签置为-1，不予训练
    crowd_ix = tf.where(gt_class_ids < 0)[:, 0]
    if crowd_ix.shape[0] > 0:
        crowd_boxes = tf.gather(gt_boxes, crowd_ix)
        # 对于负例，是与GT的iou小于阈值，并且与多实例框框的阈值小于0.001
        crowd_overlaps = overlaps_graph(proposals, crowd_boxes)
        crowd_iou_max = tf.reduce_max(crowd_overlaps, axis=1)
        negative_indices = tf.where(tf.logical_and(roi_iou_max < config.neg_anchor_thresh, crowd_iou_max < 0.001))[:, 0]
    else:
        negative_indices = tf.where(roi_iou_max < config.neg_anchor_thresh)[:, 0]

    # 定义正负例的个数

    positive_count = tf.minimum(tf.cast(config.TRAIN_ROIS_PER_IMAGE * config.ROI_POSITIVE_RATIO, tf.int32),
                                tf.size(positive_indices))
    r = 1.0/config.ROI_POSITIVE_RATIO

    negative_count = tf.cast((r-1.0)*tf.cast(positive_count,tf.float32), tf.int32)
    negative_count = tf.minimum(negative_count, tf.size(negative_indices))

    # 随意选择正负样本
    positive_indices = tf.random_shuffle(positive_indices)[:positive_count]
    negative_indices = tf.random_shuffle(negative_indices)[:negative_count]

    # 取出正负例的proposal，只选择这些去训练
    positive_rois = tf.gather(proposals, positive_indices)
    negative_rois = tf.gather(proposals, negative_indices)

    # 取出 proposal对应的gt_boxes和相应的class_ids,mask
    positive_overlaps = tf.gather(overlaps, positive_indices)
    gt_indices = tf.argmax(positive_overlaps, axis=1)  # 其长度等于正例个数
    roi_gt_boxes = tf.gather(gt_boxes, gt_indices)
    roi_class_ids = tf.gather(gt_class_ids, gt_indices)
    roi_mask = tf.gather(gt_masks, gt_indices, axis=0)

    gt_deltas = box_deltas(positive_rois, roi_gt_boxes)
    gt_deltas /= config.RPN_BBOX_STD_DEV

    # 把mask调整为[正例个数, 高，宽， 1]
    roi_mask = tf.expand_dims(roi_mask, -1)

    # boxes的长度是所需要的正例的训练个数
    boxes = positive_rois
    if config.USE_MINI_MASK:
        # MINI mask的高宽，对应GT bounding box的高宽。
        # 下面代码的作用，是将roi相对于原始图片的坐标，转换为相对于gt的坐标
        x1, y1, x2, y2 = tf.split(positive_rois, num_or_size_splits=4, axis=1)
        gt_x1, gt_y1, gt_x2, gt_y2 = tf.split(roi_gt_boxes, num_or_size_splits=4, axis=1)
        gt_h = gt_y2 - gt_y1
        gt_w = gt_x2 - gt_x1
        y1 = (y1 - gt_y1) / gt_h
        x1 = (x1 - gt_x1) / gt_w
        y2 = (y2 - gt_y1) / gt_h
        x2 = (x2 - gt_x1) / gt_w
        boxes = tf.concat([y1, x1, y2, x2], 1)
    # 在proposal的地方裁剪mask，然后resize成[28, 28]的大小
    # 蛋疼的是，在tensorflow中，图片左上角为原点，水平向右是x轴，竖直向下是y轴
    # 对于这个api，boxes必须是规范化坐标，且为[y1, x1, y2, x2]，就是说，第一个参数衡量竖直向下的偏移量
    # 所以就有了下面这两句话
    else:
        x1, y1, x2, y2 = tf.split(boxes, num_or_size_splits=4, axis=1)
        boxes = tf.concat([y1,x1,y2,x2], axis=1)

    # 把proposal地方的mask截取出来
    masks = tf.image.crop_and_resize(
        image=tf.cast(roi_mask, tf.float32), boxes=boxes,
        box_ind=tf.range(0, tf.shape(roi_mask)[0]), crop_size=config.MASK_SHAPE)
    # 把mask的shape还原为[正例个数, 高，宽]
    masks = tf.squeeze(masks, axis=3)
    # 在resize过程中，可能导致非0和1的float数出现，这里二值化为0和1
    masks = tf.round(masks)

    # # 把正例roi和负例roi拼接起来，并对多余的进行零填充
    proposals = tf.concat([positive_rois, negative_rois], axis=0)
    b = tf.shape(negative_rois)[0]  # 负的
    roi_gt_class_ids = tf.pad(roi_class_ids, [(0, b)])  # 负例的pad为0
    gt_deltas = tf.pad(gt_deltas, [(0, b), (0, 0)])
    masks = tf.pad(masks, [(0, b), (0, 0), (0, 0)])
    ###################################以下代码是为了多批数训练而写的，每次训练一批时不用填充##################
    # a = tf.maximum(config.TRAIN_ROIS_PER_IMAGE - tf.shape(proposals)[0], 0)  # 不够凑足一个TRAIN_ROIS_PER_IMAGE的
    # proposals = tf.pad(proposals, [(0, a), (0, 0)])
    # roi_gt_class_ids = tf.pad(roi_gt_class_ids, [(0, a)], constant_values=-1) # 超出的pad为-1
    # gt_deltas = tf.pad(gt_deltas, [(0, a), (0, 0)])
    # masks = tf.pad(masks, [(0, a), (0, 0), (0, 0)])

    return proposals, roi_gt_class_ids, gt_deltas, masks

def box_deltas(box, gt_box):
    """
    计算box与gt_box之间的框框偏差
    y的回归 = （GT的y-anchor的y）/anchor的高
    高的回归 = log(GT的高 / anchor的高)
    :param box: [N, (x1, y1, x2, y2)]
    :param gt_box: [N, (x1, y1, x2, y2)]
    :return: [dx, dy, dh, dw]
    """
    asserts = [tf.Assert(tf.equal(tf.shape(box)[0], tf.shape(gt_box)[0]), ["The length of box and gt_box must be same"])]
    with tf.control_dependencies(asserts):
        box = tf.identity(box)

    box = tf.cast(box, tf.float32)
    gt_box = tf.cast(gt_box, tf.float32)

    width = box[:, 2] - box[:, 0]
    height = box[:, 3] - box[:, 1]
    center_x = (box[:, 0] + box[:, 2])/2
    center_y = (box[:, 1] + box[:, 3])/2

    gt_width = gt_box[:, 2] - gt_box[:, 0]
    gt_height = gt_box[:, 3] - gt_box[:, 1]
    gt_center_x = (gt_box[:, 0] + gt_box[:, 2])/2
    gt_center_y = (gt_box[:, 1] + gt_box[:, 3])/2

    dy = (gt_center_y - center_y) / height
    dx = (gt_center_x - center_x) / width
    dh = tf.log(gt_height / height)
    dw = tf.log(gt_width / width)

    result = tf.stack([dx, dy, dh, dw], axis=1)
    return result




def proposalLayer(inputs, max_proposal,nms_thresh, name=None):
    """
    该函数根据特征输出，来制定proposal，这些proposal已经进行了非极大值抑制
    :param inputs: 包含三个元素的列表， 依次是rpn_binary_class, rpn_bbox, anchors
    :param max_proposal:
    :param nms_thresh:
    :return:经过非极大值抑制后的proposal，shape是[批数，proposal个数，(x1, y1, x2, y2)],正则化坐标
    """
    # rpn_binary_class的形状是[批数，anchor数，2]，只取正例的分数
    scores = inputs[0][:, :, 1]
    # rpn_bbox的形状是[批数，anchor数，4]
    deltas = inputs[1]
    deltas = deltas * tf.reshape(config.RPN_BBOX_STD_DEV, [1, 1, 4])
    anchors = inputs[2]  # 取出anchors

    # 取anchor数和6000的较小值，只保留这些proposal
    pre_nms_limit = tf.minimum(config.MAX_PROPOSAL_TO_DETECT, tf.shape(anchors)[1])
    # 按照最后一个维度，取出分数最高的前k个的索引,以列表形式返回,这里，ix是一个二维列表
    ix = tf.nn.top_k(scores, pre_nms_limit, sorted=True).indices  # (批数，选中的编号)

    # scores,deltas和anchors的rank都是3， 需要定义一个函数，来对anchor进行选取
    # 虽然这种写法很蛋疼，但暂时也找不到更好的写法，待研究
    def unmatch_dim_gather(values, indice):
        batch_size = values.shape[0]  # 批数
        asserts = tf.Assert(tf.equal(batch_size, 1), [tf.shape(values)])
        with tf.control_dependencies([asserts]):
            values = tf.identity(values)
        outputs = []
        for i in range(batch_size):
            out = tf.gather(values[i], indice[i])
            # if not isinstance(out, (tuple, list)):
            #     out = [out]
            outputs.append(out)
        outputs = tf.convert_to_tensor(outputs)
        return outputs

    scores = tf.expand_dims(scores, dim=-1)

    scores = unmatch_dim_gather(scores, ix)   # (批数，anchor数，1)
    scores = tf.squeeze(scores, -1)
    deltas = unmatch_dim_gather(deltas, ix) # (批数，anchor数，4)
    pre_nms_anchors = unmatch_dim_gather(anchors, ix)  # (批数，anchor数，4)



    # 修正坐标
    boxes = apply_box_deltas(pre_nms_anchors, deltas)

    # 我们使用正则化坐标，所有框框的坐标值都在[0, 1]之间
    boxes = tf.clip_by_value(boxes, 0, 1)
    boxes = tf.squeeze(boxes,0)
    width = boxes[:,2] - boxes[:,0]
    height = boxes[:,3] - boxes[:, 1]
    index = tf.where((width > 0.00001) & (height > 0.00001))[:,0]
    boxes = tf.gather(boxes, index)
    boxes = tf.expand_dims(boxes, 0)
    scores = tf.gather(scores, index, axis=-1)

    boxes = nms(boxes, scores, max_proposal, nms_thresh)
    return boxes

def fpn_classifier_graph(rois, mrcnn_feature_maps, input_image_shape, pool_size, num_classes, name=None):
    """
    构建FPN的分类与回归
    :param rois: Proposals， [batch, num_rois, (x1, y1, x2, y2)]
    :param mrcnn_feature_maps: 一个列表，[p2, p3, p4, p5] 代表四个层级的特征图
    其相对于输入图片的缩放倍数依次是8, 16， 32， 64
    :param input_image_shape: 原始输入图片的shape，[高，宽，通道数]。一个批次的所有图片，必须有相同的shape
    :param pool_size: ROI Pooling后的大小，一般是7*7
    :param num_classes: 分类数，他决定了最终的通道数，因为我们用global avarage pool
    :return:
    """

    # [num_boxes, height, width, channels], ROI Pooling后的结果
    x = pyramidROIAlign(pool_size, rois, input_image_shape, mrcnn_feature_maps)

    num_boxes = tf.shape(x)[0]

    # 这里其实就是全连接,并且批数由num_boxes代替了

    x = conv2d(inputs=x,out_channal=1024, kernel_size=pool_size[0],strides=pool_size[0], use_bias=True, name=name+"_class_conv1")

    # 这里是全连接，就不来batch_norm了
    x = tf.nn.relu(x)
    x = conv2d(inputs=x, out_channal=1024, kernel_size=1, strides=1, use_bias=True, name=name+"_class_conv2")
    # 这时候，x的shape是[num_boxex, 1, 1, 1024], 调用下面这句话以后，变成了[num_box, 1024]
    shared = tf.squeeze(tf.nn.relu(x),axis=[1, 2])
    # 下面分为两个head，一个用于回归，一个用于分类
    mrcnn_class_logits = dense(inputs=shared,out_dimension=num_classes,use_biase=True,name=name+"_class_logits")
    mrcnn_class_probs = tf.nn.softmax(mrcnn_class_logits,name=name+"_class_probs")

    mrcnn_bbox = dense(inputs=shared, out_dimension=4*num_classes, use_biase=True,name=name+"_bbox_fc")
    mrcnn_bbox = tf.reshape(mrcnn_bbox, shape=[num_boxes, num_classes, 4], name=name+"_bbox")
    # mrcnn_class_logits, mrcnn_class_probs的shape都是[num_boxex, num_classes]
    # mrcnn_bbox 的shape是[num_boxex, num_classes, (dx, dy, log(h), log(w))]
    return mrcnn_class_logits, mrcnn_class_probs, mrcnn_bbox




def pyramidROIAlign(pool_size, rois, image_shape, feature_maps):
    """
    在不同层次的feature map上执行 ROI Pooling
    :param pool_size:  特征图进行feature pool以后的网格，[高，宽]，通常是[7，7]
    :param rois: [batch, num_boxex, (x1, y1, x2, y2)],归一化坐标，如果不够，就用零填充
    :param image_shape: 原始输入图片的shape，[高，宽，通道数]。一个批次的所有图片，必须有相同的shape
    :param feature_maps: 一个列表，列表每个元素表示来自不同特征图层级的，shape都是[batch, height, width, channels]
    :return: [num_boxes, height, width, channels], ROI Pooling后的结果
    """
    x1, y1, x2, y2 = tf.split(rois, 4, axis=2)  # shape [batch, num, 4]
    h = y2 - y1  # shape [batch, num, 1]
    w = x2 - x1

    image_area = tf.cast(image_shape[0]*image_shape[1], tf.float32)
    # 下面这几行，根据FPN的式(1)而来。我们的proposal明明有指定的来源，即每个proposal来自哪个层级，是清晰的，
    # 奈何这里又要根据公式(1)指定层级呢？
    # TODO 这里有待研究，论文写得蛋疼
    roi_level = tf.log(h*w*image_area/(224.0*224.0))/tf.log(2.0)/2  # shape [batch, num, 1]
    roi_level = tf.minimum(5, tf.maximum(2, 4+ tf.cast(tf.round(roi_level), tf.int32)))
    roi_level = tf.squeeze(roi_level, 2)  # shape [batch, num]

    # 对每一个层级进行遍历
    pooled, box_to_level = [], []
    for i, level in enumerate(range(2, 6)):
        # tf.where()的返回值中，有几个真值就返回几行，每一行代表这个真值的坐标值
        # 这里，roi_level的rank是2，故，ix的每一行有两个元素，代表了该真值所在坐标位置,坐标中，
        # 第一个表示来自哪一批，第二个表示来自该批的第几个
        ix = tf.where(tf.equal(roi_level, level))
        level_boxes = tf.gather_nd(rois, ix)
        batch_indices = tf.cast(ix[:, 0], tf.int32)  # 反映批数信息，为下面切片做准备
        box_to_level.append(ix)  # 记录该层级所拥有的proposal坐标

        # 用approximate joint training的法则，把proposal当作常数处理，对梯度不贡献
        level_boxes = tf.stop_gradient(level_boxes)
        batch_indices = tf.stop_gradient(batch_indices)

        with tf.control_dependencies([tf.assert_rank(
                level_boxes, 2, data=["may be some wrong in how to use api tf.gather_nd"])]):
            x1, y1, x2, y2 = tf.split(level_boxes, 4, 1)
            temp = tf.concat([y1, x1, y2, x2], axis=1)
            # temp = tf.stack()

            # roi是有零填充的，零填充以后，所截取的feature map，还怎么放大到7*7？？？
            # tf.image.crop_and_resize返回的shape是[所截取图片张数，高，宽，通道数]
            pooled.append(tf.image.crop_and_resize(
                feature_maps[i], temp, batch_indices, pool_size, method='bilinear'))
    # 合并成tensor，shape是[num_pic, height, width, channels]
    pooled = tf.concat(pooled, axis=0)

    # 下面这几条蛋疼的代码，是为了将同一批次的组合起来，然后合并成[批数，num_pic， 高，宽，通道数]
    # 有时间再看吧，实在懒了，就令batch_size = 1 万事大吉
    # 在第三列添加一个标记，记录box的id
    box_to_level = tf.concat(box_to_level, axis=0) # shape 是[图片张数， 2]
    box_range = tf.expand_dims(tf.range(tf.shape(box_to_level)[0]), 1)  # [图片张数，1]
    box_to_level = tf.concat([tf.cast(box_to_level, tf.int32), box_range], axis=1)  # [num_pic, 3]
    # 排序，以批次号为主，批次号相同的，再看box index
    sorting_tensor = box_to_level[:, 0] * 1000 + box_to_level[:, 1]  # [num_pic]
    ix = tf.nn.top_k(sorting_tensor, k=tf.shape(box_to_level)[0], sorted=True).indices[::-1]
    pooled = tf.gather(pooled, ix)

    return pooled


def build_fpn_mask_graph(rois, feature_maps, image_shape, pool_size, num_class, train_bn=True, name=None):
    """
    构建mask
    :param rois:  Proposals， [batch, num_rois, (x1, y1, x2, y2)]
    :param feature_maps:  一个列表，[p2, p3, p4, p5] 代表四个层级的特征图
    :param image_shape: 原始输入图片的shape，[高，宽，通道数]。一个批次的所有图片，必须有相同的shape
    :param pool_size: ROI Pooling后的大小，在论文中，对于mask是14*14
    :param num_class: 分类数，他决定了最终的通道数
    :param train_bn:
    :return: [num_boxes, 28, 28, num_classes]
    """
    # [num_boxes, height, width, channels], ROI Pooling后的结果
    x = pyramidROIAlign(pool_size, rois,image_shape,feature_maps)
    for i in range(4):
        x = conv2d(inputs=x, out_channal=256, kernel_size=3, strides=1, use_bias=False, name=name+"_conv"+str(i+1))
        x = batch_norm(x, train_bn, name=name+"_bn"+str(i+1))
        x = tf.nn.relu(x)
    x = conv2d_transpose(inputs=x, out_channal=256, kernel_size=2, strides=2, name=name+"_deconv")
    x = tf.nn.relu(x)
    x = conv2d(x, num_class, 1, 1, use_bias=True, name=name)
    return x


def smooth_l1_loss(x):
    x = tf.abs(x)
    less_than_one = tf.cast(tf.less(x, 1.0), tf.float32)
    return less_than_one * x*x/2 + (1-less_than_one)*(x - 0.5)


def rpn_binary_loss_graph(rpn_binary_gt, rpn_binary_logits):
    """
    :param rpn_binary_gt: [批数, 个数]， 1表示正例，0表示负例，-1则不予考虑
    :param rpn_binary_logits: [批数，anchors数，2]，最后一个维度里面，分别是负例、正例
    :return: 交叉熵
    """
    rpn_binary_gt = tf.reshape(rpn_binary_gt,shape=[-1])
    rpn_binary_gt = tf.cast(rpn_binary_gt, tf.int32)
    rpn_binary_logits = tf.reshape(rpn_binary_logits, shape=[-1, 2])
    with tf.control_dependencies([tf.Assert(
            tf.equal(tf.shape(rpn_binary_logits)[0], tf.shape(rpn_binary_gt)[0]),
            data=["the length must be same", tf.shape(rpn_binary_logits)[0],tf.shape(rpn_binary_gt)[0]])]):
        # 取出有效的索引
        ix = tf.where(tf.not_equal(rpn_binary_gt, -1))[:,0]

    rpn_binary_gt = tf.gather(rpn_binary_gt, ix)
    rpn_binary_logits = tf.gather(rpn_binary_logits, ix)
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=rpn_binary_gt, logits=rpn_binary_logits)
    return tf.where(tf.size(ix)>0,tf.reduce_mean(loss),tf.constant(0.0))

def rpn_bbox_loss_graph(rpn_bbox_gt, rpn_bbox_pred, rpn_binary_gt):
    """

    :param rpn_bbox_gt: [批数, anchor个数, (dx, dy, log(h), log(w))]
    :param rpn_bbox_pred:  [批数，anchors数，4]
    :param rpn_binary_gt: [批数, 个数, 1]， 1表示正例，0表示负例，-1则不予考虑，我们只考察正例的回归损失
    :return:
    """
    rpn_binary_gt = tf.cast(tf.reshape(rpn_binary_gt,shape=[-1]), tf.int32)
    rpn_bbox_gt = tf.reshape(rpn_bbox_gt, [-1, 4])
    rpn_bbox_pred = tf.reshape(rpn_bbox_pred, [-1, 4])

    with tf.control_dependencies([tf.Assert(
            tf.equal(tf.shape(rpn_bbox_gt)[0], tf.shape(rpn_bbox_pred)[0]),
            data=["in {}, line{},the length must be same".format(__file__, sys._getframe().f_lineno),
                  tf.shape(rpn_bbox_gt),tf.shape(rpn_bbox_pred)])]):
        positive_ix = tf.where(tf.equal(rpn_binary_gt, 1))[:, 0]

    rpn_bbox_gt = tf.gather(rpn_bbox_gt, positive_ix)
    rpn_bbox_pred = tf.gather(rpn_bbox_pred, positive_ix)
    loss = tf.reduce_sum(smooth_l1_loss(rpn_bbox_gt - rpn_bbox_pred))

    return tf.where(tf.size(positive_ix) > 0,loss/tf.cast(tf.size(positive_ix), tf.float32),tf.constant(0.0))


def proposal_bbox_loss_graph(target_bbox,mrcnn_bbox,target_class_ids):
    """
    注意，这里的mrcnn_bbox的shape，每个类别都有自己的回归目标
    :param target_bbox: [batch, N, 4]
    :param mrcnn_bbox: [num_boxex, num_classes, (dx, dy, log(h), log(w))]
    :param target_class_ids: [batch, N]
    :return:
    """

    target_bbox = tf.reshape(target_bbox, [-1, 4])  # [num_box, 4]
    target_class_ids = tf.reshape(target_class_ids, [-1])  # [num_box]
    target_class_ids = tf.cast(target_class_ids, tf.int32)
    with tf.control_dependencies(
            [tf.Assert(tf.logical_and(tf.equal(tf.shape(target_bbox)[0], tf.shape(mrcnn_bbox)[0]),
                                      tf.equal(tf.shape(mrcnn_bbox)[0], tf.shape(target_class_ids)[0])),
                       data=["the shape must be same"]  )]):
        positive_ix = tf.cast(tf.where(target_class_ids > 0)[:, 0], tf.int32)

    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int32)  # 正例的具体id
    indice = tf.stack([positive_ix, positive_class_ids], axis=1)  # [在num_box中的索引编号，具体的id号]
    mrcnn_bbox = tf.gather_nd(mrcnn_bbox, indice)  # [N, 4]

    target_bbox = tf.gather(target_bbox, positive_ix)  # [N, 4]

    loss = tf.reduce_sum(smooth_l1_loss(target_bbox - mrcnn_bbox))

    return tf.where(tf.size(positive_ix) > 0,loss / tf.cast(tf.size(positive_ix), tf.float32),tf.constant(0.0))


def proposal_class_loss_graph(target_ids, logits, num_class):
    """是有零填充的，"""
    target_ids = tf.reshape(target_ids, shape=[-1])
    target_ids = tf.cast(target_ids, tf.int32)
    logits = tf.reshape(logits, shape = [-1, num_class])
    with tf.control_dependencies([tf.Assert(tf.equal(tf.shape(target_ids)[0], tf.shape(logits)[0]),
                                            data=["The length must be same",tf.shape(target_ids)[0], tf.shape(logits)[0]])]):
        valid_ix = tf.where(tf.not_equal(target_ids, -1))[:, 0]
    target_ids = tf.gather(target_ids, valid_ix)
    logits = tf.gather(logits, valid_ix) # shape(90, 81)
    # TODO problem
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_ids, logits=logits)
    return tf.where(tf.size(valid_ix)>0,tf.reduce_mean(loss), tf.constant(0.0)), target_ids


def mask_loss_graph(target_mask, mrcnn_mask_logits, target_class_ids, num_class):
    """
    :param target_mask: [batch, N, 高，宽]
    :param mrcnn_mask_logits:  [num_boxes, 28, 28, num_classes]
    :param target_class_ids: [batch, N]
    :param num_class: 类别数
    :return:
    """
    target_class_ids = tf.reshape(target_class_ids, [-1])  # [num_boxes]
    target_shape = tf.shape(target_mask)
    target_mask = tf.reshape(target_mask, (-1, target_shape[2], target_shape[3]))  # [num_boxex, 高，宽]

    mrcnn_mask_logits = tf.transpose(mrcnn_mask_logits, [0, 3, 1, 2])  # [num_boxes, num_classes, 28, 28]

    positive_ix = tf.cast(tf.where(target_class_ids > 0)[:, 0], tf.int32)  # 正例在num_box中的索引号
    y_true = tf.gather(target_mask, positive_ix)  # [num_boxex, 28, 28]
    positive_class_ids = tf.cast(tf.gather(target_class_ids, positive_ix), tf.int32)  # 正例的具体id
    indice = tf.stack([positive_ix, positive_class_ids], axis=1)  # [在num_box中的索引号，类别号]
    y_pred = tf.gather_nd(mrcnn_mask_logits, indice)  # [num_boxex, 28，28]
    with tf.control_dependencies([tf.Assert(tf.equal(tf.shape(y_true)[0],tf.shape(y_pred)[0]),
                                            data=["the shape must be same",tf.shape(y_true),tf.shape(y_pred)])]):
        loss = tf.reduce_sum(tf.nn.sigmoid_cross_entropy_with_logits(labels=y_true, logits=y_pred))
    print(positive_ix,"===========")
    return tf.where(tf.size(positive_ix) > 0,loss/tf.cast(tf.size(positive_ix), tf.float32,name="cast_yi"),tf.constant(0.0))

def detectionLayer(proposal, probs, bbox, image_shape):
    """
    一次检测一张图片
    :param proposal: 经过非极大值抑制后, [批数，个数，4]
    :param probs: [num_boxex, num_classes]
    :param bbox: [num_boxex, num_classes, (dx, dy, log(h), log(w))]
    :param image_shape: [高，宽，通道数]
    :return: 盒子，ids， 概率
    """
    with tf.control_dependencies([tf.Assert(tf.shape(proposal)[0] == 1, data=["A single picture for evaluation each time"])]):
        proposal = tf.squeeze(proposal, axis=[0,])  # [num_boxes, 4]

    class_ids = tf.argmax(probs, axis=1, output_type=tf.int32)  # [num_boxes]
    indices = tf.stack([tf.range(probs.shape[0]), class_ids], axis=1)  # [序号，类别号]
    class_probs = tf.gather_nd(probs, indices)  # [num_box]
    deltas = tf.gather_nd(bbox, indices)  # [num_box, 4]
    refined_rois = apply_box_deltas(proposal, deltas*config.RPN_BBOX_STD_DEV)  # [num_boxes, 4]
    refined_rois = tf.clip_by_value(refined_rois, 0, 1)

    keep = tf.where(class_ids > 0)[:, 0]  # 取出前景
    if config.DETECTION_MIN_CONFIDENCE:
        # 如果使用最小confidence，就取交集
        conf_keep = tf.where(class_probs >= config.DETECTION_MIN_CONFIDENCE)[:, 0]
        keep = tf.sets.set_intersection(tf.expand_dims(keep, 0),
                                        tf.expand_dims(conf_keep, 0))
        keep = tf.sparse_tensor_to_dense(keep)[0]

    ###########################下面的，全部重新组合排序########################
    pre_nms_class_ids = tf.gather(class_ids, keep)  # 取出id  [num]
    pre_nms_scores = tf.gather(class_probs, keep)  # 取出相应的概率值  [num]
    pre_nms_rois = tf.gather(refined_rois,   keep)  # 取出相应的框框 [num, 4]
    unique_pre_nms_class_ids = tf.unique(pre_nms_class_ids)[0]  # 能找到的分类数  [unique]

    def nms_keep_map(class_id):
        """
        只有属于同一类别的，才进行非极大值抑制
        :param class_id: 给定的某一类别号
        :return:
        """
        ixs = tf.where(tf.equal(pre_nms_class_ids, class_id))[:, 0]
        class_keep = tf.image.non_max_suppression(
            boxes=tf.gather(pre_nms_rois, ixs),
            scores=tf.gather(pre_nms_scores, ixs),
            max_output_size=config.DETECTION_MAX_INSTANCE,
            iou_threshold=config.DETECTION_NMS_THRESHHOLD)
        # class_keep取出的是相对于ixs的序号，tf.gather(ixs, class_keep)就是ixs的数值本身
        # ixs的数值本身，就是pre_nms_class_ids的序号
        class_keep = tf.gather(ixs, class_keep)
        # 用-1填充，使得都有相同的shape
        gap = config.DETECTION_MAX_INSTANCE - tf.shape(class_keep)[0]
        class_keep = tf.pad(class_keep, [(0, gap)], mode='CONSTANT', constant_values=-1)
        # 设置shape，使得map_fn()能够立即知道她的shape
        class_keep.set_shape([config.DETECTION_MAX_INSTANCE])
        return class_keep

    # nms_keep的shape是，[unique_pre_nms_class_ids的长度，config.DETECTION_MAX_INSTANCE]
    nms_keep = tf.map_fn(nms_keep_map, unique_pre_nms_class_ids, dtype=tf.int32)

    nms_keep = tf.reshape(nms_keep, [-1])
    # keep就是pre_nms_class_ids的序号
    keep = tf.gather(nms_keep, tf.where(nms_keep > -1)[:, 0])

    #
    class_probs2 = tf.gather(pre_nms_scores, keep)
    num_keep = tf.minimum(tf.shape(keep)[0], config.DETECTION_MAX_INSTANCE)
    top_ids = tf.nn.top_k(class_probs2, num_keep, sorted=True).indices
    keep = tf.gather(keep, top_ids)  # 至此，keep是分数最大的，不超过实例个数的保留值了

    return tf.gather(pre_nms_rois, keep), tf.gather(pre_nms_class_ids, keep), tf.gather(pre_nms_scores, keep)

def filter_mask(mask, ids):
    """
    :param mask: float型，[num, 28, 28, num_classes]
    :param ids: int，[num]
    :return: [num，28， 28]，与ids一一对应
    """
    num = tf.size(ids)
    mask = tf.transpose(mask, [0, 3, 1, 2])
    indice = tf.stack([tf.range(num), ids], axis=1)
    mask = tf.gather_nd(mask, indice)
    return mask

class MASK_RCNN(object):
    def __init__(self):
        self.anchors = None

    def get_anchors(self, batch_size,resolution, image_shape, smallest_anchor_scale):
        """
        :param resolution:  一个int列表，其长度等于feature_map的层级个数，由底层到高层，依次是缩放比。对于resnet50而言，是
        [8, 16， 32， 64， 128]
        :param image_shape: 二元组，(gao ,kuan)输入图片的大小
        :param smallest_anchor_scale: 最底层feature map上面每个像素对应的anchor大小
        :param batch_size: 生成多少批
        :return: 所有的anchor，由底层到高层拼接起来，形状是[批数， 个数，(x1, y1, x2, y2)],normalized坐标
        """
        # 返回相对像素坐标
        anchors = generate_pyramid_anchors(batch_size, resolution, image_shape, smallest_anchor_scale)
        anchors = tf.convert_to_tensor(anchors)
        anchors = tf.cast(anchors, tf.float32)
        height = image_shape[0]
        width = image_shape[1]
        scale = tf.constant([width - 1, height - 1, width - 1, height - 1],dtype=tf.float32)
        shift = tf.constant([0.0, 0.0, 1.0, 1.0])
        anchors =  tf.divide((anchors - shift), scale)

        self.anchors = anchors

    # image, gt_box, gt_class, mask, anchor_labels, anchor_deltas
    def build_model(self, mode, input_image, gt_boxes=None, class_ids=None,
                    input_gt_mask=None, anchor_labels=None,anchor_deltas=None):
        """
        feature map有五个层级，p2, p3, p4, p5, p6，其相对于输入图片的缩放倍数依次是8, 16， 32， 64， 128
        在特征图上每个像素点的位置都要产生3个不同ratio的anchor.假设第s层有N个像素点，则在s层产生的anchor数是3N,
        其shape为[3N, （x1, y1, x2, y2）]。有五个层级，则调用tf.cancat函数在第0维拼接起来，形成的shape是
        [num_anchor, （x1, y1, x2, y2）]。最后，按照批数拼接起来，最终的shape是[batch, num_anchor, （x1, y1, x2, y2）]
        我们采用正则化坐标，故所有的坐标值的范围都必须在区间[0,1]里面

        mode:  必须是'training','validation','inference'三者之一。mode是'training' 或者 'validation'时，所有参数都不能是None，
        mode是'inference'时，只需要提供input_image即可

        :param mode:  必须是'training','validation','inference'三者之一
        :param input_image: [1, 高, 宽, 3]  # 简单一点，每次一张图片
        :param gt_boxes: shape=[1, gt个数， 4]
        :param class_ids: shape=[1, gt个数]
        :param input_gt_mask: [1，gt个数，高，宽]
        :param anchor_labels: [批数，anchor个数]，其中1表示正例，0表示负例，-1表示不予考虑
        :param anchor_deltas: anchor与gt之间的回归差异，[批数，anchor个数，(dx, dy, log(h), log(w))]
        :return:
        """
        mode_validation = mode in ['training','validation','inference']
        with tf.control_dependencies([tf.Assert(mode_validation, data=["invalid mode"])]):
            batch_size = input_image.shape[0]
            resnet = Model(resnetlist=resnet50, version=1)
        training = True if mode == 'training' else False

        # layer是一个列表，包含c2, c3, c4, c5，其相对于输入图片的缩放比例依次是8,16,32,64
        # resolution是输入图片相对于特征图的分辨率倍数，是一个列表，依次是[8,16,32,64]
        layer, resolution = resnet(inputs=input_image, training=training)
        P5 = conv2d(inputs=layer[3], out_channal=256, kernel_size=1, strides=1, name='fpn_c5p5')

        P4 = tf.add_n([conv2d_transpose(inputs=P5, out_channal=256, kernel_size=1, strides=2, name="fpn_trans4"),
                     conv2d(inputs=layer[2], out_channal=256, kernel_size=1, strides=1, name="fpn_c4p4")], name="fpn_p4add")

        P3 = tf.add_n([conv2d_transpose(inputs=P4, out_channal=256, kernel_size=1, strides=2, name="fpn_trans3"),
                     conv2d(inputs=layer[1], out_channal=256, kernel_size=1, strides=1, name="fpn_c3p3")],name="fpn_p3add")

        P2 = tf.add_n([conv2d_transpose(inputs=P3, out_channal=256, kernel_size=1, strides=2, name="fpn_trans2"),
                     conv2d(inputs=layer[0], out_channal=256, kernel_size=1, strides=1, name="fpn_c2p2")], name="fpn_p2add")

        # 根据FPN，最终来一个卷积，得到最后的特征图。没有非线性函数
        p2 = conv2d(P2, 256, 3, 1, name="fpn_p2")
        p3 = conv2d(P3, 256, 3, 1, name="fpn_p3")
        p4 = conv2d(P4, 256, 3, 1, name="fpn_p4")
        p5 = conv2d(P5, 256, 3, 1, name="fpn_p5")

        # feature map 6 用来做RPN，不用来做proposal的相关分类
        p6 = tf.layers.max_pooling2d(inputs=p5, pool_size=1, strides=2, name='feature_map6')
        resolution6 = resolution[-1]*2
        resolution.append(resolution6)

        rpn_feature_maps = [p2, p3, p4, p5, p6]
        mrcnn_feature_maps = [p2, p3, p4, p5]

        # 定义一个列表，其长度为输出特征图的层级数，用于装入每级特征图的rpn输出。
        # rpn输出包含[rpn_binary_logits, rpn_probs, rpn_bbox]，
        # 其shape依次是[批数，每个层级的anchors数，2],[批数，anchors数，2],[批数，anchors数，4]
        layer_output = []

        for i, p in enumerate(rpn_feature_maps):
            layer_output.append(rpn_graph(p, config.anchor_per_location, anchor_stride=1, name=str(i)))

        # 把各层的输出连接起来，[[a1, b1, c1], [a2, b2, c2]] => [[a1, a2], [b1, b2], [c1, c2]]
        output_name = ['rpn_binary_logits', 'rpn_binary_probs', 'rpn_bbox']
        outputs = list(zip(*layer_output))
        # 从底层到高层连接，连接以后，anchor数翻了5倍
        outputs = [tf.concat(list(o),axis=1,name=n) for o, n in zip(outputs, output_name)]
        # [批数，anchors数，2], [批数，anchors数，2], [批数，anchors数，4]
        rpn_binary_logits, rpn_binary_probs, rpn_bbox_pred = outputs

        # 保留的proposal个数
        num_proposal = config.POST_NMS_ROIS_TRAINING if mode == 'training' else config.POST_NMS_ROIS_INFERENCE

        if self.anchors is None:
            self.get_anchors(config.batch_size, resolution, config.input_shape, config.smallest_anchor_size)

        # 生成经过非极大值抑制后的proposal, 形状是 [批数，个数，4]
        proposal = proposalLayer(inputs=[rpn_binary_probs, rpn_bbox_pred, self.anchors],
                                 max_proposal=num_proposal,nms_thresh=config.RPN_NMS_THRESHOLD, name="ROI")

        if mode == 'inference':
            mrcnn_class_logits, mrcnn_class_probs, mrcnn_bbox = fpn_classifier_graph(
                proposal, mrcnn_feature_maps, config.IMAGE_SHAPE, config.POOL_SIZE, config.NUM_CLASSES,name="mrcnn")
            # 经过最终处理以后的盒子，[x1, y1, x2, y2], 对应的类别，概率
            boxes, ids, probs = detectionLayer(proposal, mrcnn_class_probs, mrcnn_bbox, config.IMAGE_SHAPE)
            mask = build_fpn_mask_graph(
                tf.expand_dims(boxes, 0), mrcnn_feature_maps,
                config.IMAGE_SHAPE, config.MASK_POOL_SIZE, config.NUM_CLASSES, train_bn=training, name="mrcnn_mask")
            mask = filter_mask(mask, ids)
            mask = tf.nn.sigmoid(mask)
            return [boxes, ids, probs, mask]

        else:
            # 调用detection_targets函数，返回proposal,以及相应的类别、回归、masks
            # 因为批数不好处理，故只能蛋疼地分成一批一批地处理
            rois_list, target_class_ids_list, target_bbox_list, target_mask_list = [], [], [], []
            for i in range(batch_size):
                # roi_gt_class_ids[M], 反映proposal的分类
                # gt_deltas[M, (dx, dy, log(h), log(w))]
                # 反映proposal相对于gt的回归
                # masks[M, 高，宽]
                # [N, (x1, y1, x2, y2)]; [N];  [N, 4]; [N, 高，宽]
                rois, target_class_ids, target_bbox, target_mask = detection_targets(
                    proposal[i], gt_class_ids=class_ids[i],gt_boxes=gt_boxes[i],gt_masks=input_gt_mask[i])
                rois_list.append(rois)
                target_bbox_list.append(target_bbox)
                target_class_ids_list.append(target_class_ids)
                target_mask_list.append(target_mask)
            rois = tf.convert_to_tensor(rois_list)
            target_bbox = tf.convert_to_tensor(target_bbox_list)
            target_class_ids = tf.convert_to_tensor(target_class_ids_list)
            target_mask = tf.convert_to_tensor(target_mask_list)  # [batch, N, 高，宽]


            # mrcnn_class_logits, mrcnn_class_probs的shape都是[num_boxex, num_classes]
            # mrcnn_bbox 的shape是[num_boxex, num_classes, (dx, dy, log(h), log(w))]
            mrcnn_class_logits, mrcnn_class_probs, mrcnn_bbox = fpn_classifier_graph(
                rois, mrcnn_feature_maps, config.IMAGE_SHAPE,config.POOL_SIZE, config.NUM_CLASSES, name="mrcnn")

            # [num_boxes, 28, 28, num_classes]
            mrcnn_mask_logits = build_fpn_mask_graph(
                rois, mrcnn_feature_maps, config.IMAGE_SHAPE,
                config.MASK_POOL_SIZE, config.NUM_CLASSES, train_bn=training, name="mrcnn_mask")

            # rpn loss
            rpn_binary_loss = rpn_binary_loss_graph(anchor_labels, rpn_binary_logits)
            rpn_bbox_loss = rpn_bbox_loss_graph(anchor_deltas, rpn_bbox_pred, anchor_labels)
            # TODO some problem in this loss
            proposal_class_loss, targets_id = proposal_class_loss_graph(target_class_ids, mrcnn_class_logits, config.NUM_CLASSES)
            proposal_bbox_loss = proposal_bbox_loss_graph(target_bbox,mrcnn_bbox,target_class_ids)
            mask_loss = mask_loss_graph(target_mask, mrcnn_mask_logits, target_class_ids, config.NUM_CLASSES)
            rpn_loss = rpn_binary_loss + rpn_bbox_loss  # rpn的损失
            proposal_loss = proposal_class_loss + proposal_bbox_loss  # proposal的损失
            total_loss = rpn_loss + proposal_loss + mask_loss
            # 返回rpn的损失，proposal的损失，mask的损失，和总损失
            return [rpn_loss, proposal_loss, mask_loss, total_loss]
