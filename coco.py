# -*- coding: UTF-8 -*-
import time
import shutil
import numpy as np
import tensorflow as tf
import os
import urllib.request
import config
from model import MASK_RCNN
from input.producer import producer
from config import input_shape
import argparse
import sys
from tensorflow.python import debug as tf_debug
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
COCO_MODEL_URL = "https://github.com/matterport/Mask_RCNN/releases/download/v2.0/mask_rcnn_coco.h5"

weights_path = config.COCO_WEIGHTS_PATH

# tf.float32, tf.float32, tf.int32, tf.bool, tf.int32, tf.float32
# image,      gt_box,    gt_class,  mask, anchor_labels, anchor_deltas
def main(_):

    ##############下面，除了图片，所有坐标都是归一化坐标##################
    # 输入图片 [批数，高，宽，通道数], 已经去均值
    input_images = tf.placeholder(dtype=tf.float32, shape=(1, input_shape[0], input_shape[1], 3), name='input_images')
    # ground truth， [批数，MAX_GT_INSTANCES，4]
    gt_boxes = tf.placeholder(dtype=tf.float32, shape=(1, None, 4), name='gt_boxes')
    # 类别编号 [批数，MAX_GT_INSTANCES]
    class_ids = tf.placeholder(dtype=tf.int32, shape=(1, None), name='class_ids')
    # MASK，[批数，MAX_GT_INSTANCES， 高，宽]，每个GT都要有一个标签，一个mask
    input_gt_mask = tf.placeholder(dtype=tf.bool, shape=(1, None, None, None), name='input_gt_mask')
    # 真实的anchor标签，[批数，anchor个数]，其中1表示正例，0表示负例，-1表示不予考虑
    rpn_binary_gt = tf.placeholder(dtype=tf.int32, shape=(1, None), name='rpn_binary_gt')
    # anchor与gt之间的回归差异，[批数，anchor个数，(dx, dy, log(h), log(w))]
    anchor_deltas = tf.placeholder(dtype=tf.float32, shape=(1, None, 4), name='anchor_deltas')

    if not tf.gfile.Exists(config.checkpoint_path):
        tf.gfile.MakeDirs(config.checkpoint_path)
    else:
        if not config.restore:
            tf.gfile.DeleteRecursively(config.checkpoint_path)
            tf.gfile.MakeDirs(config.checkpoint_path)

    global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
    learning_rate = tf.train.exponential_decay(config.learning_rate, global_step, decay_steps=10000, decay_rate=0.94,
                                               staircase=True)
    tf.summary.scalar('lr', learning_rate)
    opt = tf.train.AdamOptimizer(learning_rate)

    mask_rcnn = MASK_RCNN()
    #  mode, input_image, gt_boxes=None, class_ids=None,
    #                 input_gt_mask=None, anchor_labels=None,anchor_deltas=None
    rpn_loss, proposal_loss, mask_loss, model_loss = mask_rcnn.build_model(
        'training', input_images, gt_boxes, class_ids, input_gt_mask, rpn_binary_gt, anchor_deltas)
    total_loss = model_loss + tf.add_n(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
    # total_loss = model_loss +
    batch_norm_updates_op = tf.group(*tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    with_clip = False
    if with_clip:
        tvars = tf.trainable_variables()
        grads, norm = tf.clip_by_global_norm(tf.gradients(total_loss, tvars), 10.0)
        gradient_op = opt.apply_gradients(list(zip(grads, tvars)), global_step=global_step)
    else:
        gradient_op = opt.minimize(loss=total_loss, global_step=global_step)
    summary_op = tf.summary.merge_all()

    # 定义滑动平均对象
    variable_averages = tf.train.ExponentialMovingAverage(config.moving_average_decay, global_step)
    # 将该滑动平均对象作用于所有的可训练变量。tf.trainable_variables()以列表的形式返回所有可训练变量
    variables_averages_op = variable_averages.apply(tf.trainable_variables())
    # 下面这两句话等价于 train_op = tf.group(variables_averages_op, apply_gradient_op, batch_norm_updates_op)
    with tf.control_dependencies([variables_averages_op, gradient_op, batch_norm_updates_op]):
        train_op = tf.no_op(name='train_op')

    saver = tf.train.Saver(tf.global_variables())
    summary_writer = tf.summary.FileWriter(config.summary_path, tf.get_default_graph())

    init = tf.global_variables_initializer()

    next_batch = producer().make_one_shot_iterator().get_next()


    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        # 如果是从原来的模型中接着训练，就不需要sess.run(tf.global_variables_initializer())
        if config.restore:
            print('continue training from previous checkpoint')
            ckpt = tf.train.latest_checkpoint(config.checkpoint_path)
            saver.restore(sess, ckpt)
        # elif config.COCO_WEIGHTS_PATH is not None:
        #     try:
        #         print("trying to assign pre-trained model...")
        #         load_trained_weights(weights_path, sess, ignore_missing=True)
        #         print("assign pre-trained model done!")
        #     except:
        #         raise 'loading pre-trained model failed,please check your pretrained ' \
        #               'model {:s}'.format(config.COCO_WEIGHTS_PATH)
        else:
            sess.run(init)
        if FLAGS.debug and FLAGS.tensorboard_debug_address:
            raise ValueError("The --debug and --tensorboard_debug_adress flags are mutually exclusive")
        if FLAGS.debug:
            sess = tf_debug.LocalCLIDebugWrapperSession(sess, ui_type=FLAGS.ui_type)
        elif FLAGS.tensorboard_debug_address:
            sess = tf_debug.TensorBoardDebugWrapperSession(sess, FLAGS.tensorboard_debug_address)

        start = time.time()
        for step in range(config.max_steps):
            # img_name   image,      gt_box,    gt_class,  mask, anchor_labels, anchor_deltas
            # image,gt_box, gt_class, segmentation_mask, anchor_labels, anchor_deltas
            image_name,image,  gt_box, gt_class, segmentation_mask, anchor_labels, anchor_deltas_in = sess.run(next_batch)

            ml, tl, _, r_loss, p_loss, m_loss = sess.run(
                [model_loss, total_loss, train_op,  rpn_loss, proposal_loss, mask_loss],
                                 feed_dict={input_images: image,
                                            gt_boxes: gt_box,
                                            class_ids: gt_class,
                                            input_gt_mask: segmentation_mask,
                                            rpn_binary_gt: anchor_labels,
                                            anchor_deltas: anchor_deltas_in})
            if np.isnan(tl):
                print('Loss diverged, stop training')
                break

            if step % 10 == 0:
                avg_time_per_step = (time.time() - start) / 10
                start = time.time()
                print('Step {}, model loss {:.4f}, total loss {:.4f}, {:.2f} seconds/step'.format(step, ml, tl,
                                                                                                      avg_time_per_step))

            if step % config.save_checkpoint_steps == 0:
                filename = os.path.join(config.checkpoint_path, "model.ckpt")
                saver.save(sess, filename, global_step=global_step)

            if step % config.save_summary_steps == 0:
                _, tl, summary_str = sess.run([train_op, total_loss, summary_op],
                                              feed_dict={input_images: image,
                                                         gt_boxes: gt_box,
                                                         class_ids: gt_class,
                                                         input_gt_mask: segmentation_mask,
                                                         rpn_binary_gt: anchor_labels,
                                                         anchor_deltas: anchor_deltas_in})
                summary_writer.add_summary(summary_str, global_step=step)


def load_trained_weights(file_path, sess, ignore_missing=False):
    import h5py
    if h5py is None:
        raise ImportError('load_weights require h5py')
    data_dict = h5py.File(file_path, mode='r')
    for key in data_dict.keys():
        for subkey in data_dict[key]:
            with tf.variable_scope(subkey, reuse=True):
                for g in data_dict[key][subkey]:
                    try:
                        var = tf.get_variable(g.split(":")[0])
                        sess.run(var.assign(data_dict[key][subkey][g][...]))
                    except ValueError:
                        print("ignore "+ key)
                        if not ignore_missing: # 有缺失项，但是又不去忽略，就报错
                            raise



def download_trained_weights(coco_model_path, verbose=1):
    if verbose > 0:
        print("Downloading pretrained model to " + coco_model_path + " ...")
    with urllib.request.urlopen(COCO_MODEL_URL) as resp, open(coco_model_path, 'wb') as out:
        shutil.copyfileobj(resp, out)
    if verbose > 0:
        print("... done downloading pretrained model!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ui_type',
        type=str,
        default='curses',
        help="Command-line user interface type (curses | readline)")
    parser.add_argument(
        "--debug",
        type=bool,
        nargs="?",
        const=True,
        default=False,
        help="Use debugger to track down bad values during training. "
             "Mutually exclusive with the --tensorboard_debug_address flag.")
    parser.add_argument(
        "--tensorboard_debug_address",
        type=str,
        default=None,
        help="Connect to the TensorBoard Debugger Plugin backend specified by "
             "the gRPC address (e.g., localhost:1234). Mutually exclusive with the "
             "--debug flag.")

    if not os.path.exists(weights_path):
        print("no pretrained model locally")
        download_trained_weights(weights_path)
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main, argv=[sys.argv[0]] + unparsed)
