import os
import urllib.request
import shutil
import zipfile
from pycocotools.coco import COCO
import numpy as np
from lib.gen_box import generate_pyramid_anchors
import config
from config import batch_size
from config import resolution
from config import smallest_anchor_size
from config import input_shape
import cv2
from input.preprocess import cls_target2
import logging
import time
import threading
import multiprocessing
try:
    import queue
except ImportError:
    import Queue as queue


class CocoDataset(object):
    def __init__(self, dataset_dir, subset, year,class_ids=None,return_coco=False, auto_download=False):
        self.num_classes = 0  # 类别数，包括背景
        self.class_ids = None  # 类别id
        self.class_names = None  # 类的名字
        self.num_images = None  # 图片张数
        self._image_ids = None  # 这个class里面的图片的编号，从0到张数，并非coco的内置的图片id

        # 生成一个很多表项的字典，从coco数据集的类别ID映射到该class定义的类别编号
        self.class_from_source_map = None
        # 从coco数据集的图片ID映射到该class定义的图片编号
        self.image_from_source_map = None

        # 一个字典，键是“coco”，名是一个列表，列表包含了所有的类别id，从1到90
        self.source_class_ids = {"coco": list(range(self.num_classes))}
        self._image_ids = []
        # 字典列表， 键有"source","id","path","height","width","annotations"
        self.image_info = []
        # 字典列表，source永远是"coco"，类别id，这个id对应的类别的英文名
        self.class_info = [{"source":"", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)
        self.coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        self._load_coco(dataset_dir, subset, year,class_ids,return_coco)


    def add_image(self, source, image_id, path, **kwargs):
        image_info = {"id": image_id, "source": source, "path": path}
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def add_class(self, source, class_id, class_name):
        assert "." not in source, "Source name cannot contain a dot"
        # Does the class exist already?
        for info in self.class_info:
            if info['source'] == source and info["id"] == class_id:
                # source.class_id combination already available, skip
                return
        # Add the class
        self.class_info.append({
            "source": source,
            "id": class_id,
            "name": class_name,
        })

    def _load_coco(self, dataset_dir, subset, year="2017", class_ids=None,return_coco=False):
        """
        加载coco数据集, 填充 class_info和image_info
        :param dataset_dir: 数据集所在路径名
        :param subset: "train" or "val"
        :param year: 暂且只有"2017"
        :param class_ids: 只返回给定的类别的图像，否则全部返回
        :param return_coco: 是否返回coco
        :param auto_download: 是否下载
        :return:
        """
        assert subset in ("train", "val"), "subset must be either 'train' or 'val' !"

        image_dir = config.trainImage_path

        if not class_ids:
            # 获取全部的类别，train2017的类别标注是从1到90
            class_ids = sorted(self.coco.getCatIds())

        if class_ids:
            # 根据类别ids来获取图片ids
            image_ids = []
            for class_id in class_ids:
                image_ids.extend(list(self.coco.getImgIds(catIds=[class_id])))
            image_ids = list(set(image_ids))  # 去重
        else:  # 获取全部的图片ids
            image_ids = list(self.coco.imgs.keys())

        # 把类别添加进去
        for i in class_ids:
            self.add_class("coco", i, self.coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                "coco",
                image_id=i,
                path=os.path.join(image_dir, self.coco.imgs[i]['file_name']),
                height=self.coco.imgs[i]["height"],
                width=self.coco.imgs[i]["width"],
                annotations = self.coco.loadAnns(self.coco.getAnnIds([i], class_ids, iscrowd=None))
            )
        if return_coco:
            return self.coco

    @property
    def image_ids(self):
        return self._image_ids

    def prepare(self):
        def clean_name(name):
            """Returns a shorter version of object names for cleaner display."""
            return ",".join(name.split(",")[:1])

        # Build (or rebuild) everything else from the info dicts.
        self.num_classes = len(self.class_info)  # 类别数，包括背景
        self.class_ids = np.arange(self.num_classes)  # 类别id
        self.class_names = [clean_name(c["name"]) for c in self.class_info]  # 类的名字
        self.num_images = len(self.image_info)  # 图片张数
        self._image_ids = np.arange(self.num_images)  # 图片的id，从0到张数

        # 生成一个很多表项的字典，从数据集内在的id映射到class定义的类
        self.class_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.class_info, self.class_ids)}
        self.image_from_source_map = {"{}.{}".format(info['source'], info['id']): id
                                      for info, id in zip(self.image_info, self.image_ids)}

        # 一个字典，键是“coco”，名是一个列表，列表包含了所有的类别id，从1到90
        self.source_class_ids = {"coco": list(range(self.num_classes))}

    def load_image(self, image_id):
        # BGR
        image = cv2.imread(self.image_info[image_id]["path"])
        if image is None:
            print("reading image in path {} failed".format(self.image_info[image_id]["path"]))
            exit(0)
        if image.ndim != 3:
            image = cv2.cvtColor(image,cv2.COLOR_GRAY2BGR)
        if image.shape[-1] == 4:
            image = image[..., :3]
        return image



    def load_mask(self, image_id):
        """
        给定图片编号，获取mask，和
        :param image_id: 图片编号
        :return: mask，bool型np数组，[图片高，图片宽，个数]；class_ids，int32型np数组，[个数]
        """
        image_info = self.image_info[image_id]
        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]
        for ann in annotations:
            # 通过coco的类别id，得到该class的编号id
            class_id = self.class_from_source_map["coco.{}".format(ann["category_id"])]
            if class_id:
                m = self.coco.annToMask(ann)
                if m.max() < 1:
                    continue
                if ann["iscrowd"]:
                    class_id *= -1
                    if m.shape[0] != image_info['height'] or m.shape[1] != image_info['width']:
                        m = np.ones([image_info['height'], image_info['width']], dtype=bool)
                instance_masks.append(m)
                class_ids.append(class_id)

        if class_ids:
            mask = np.stack(instance_masks, axis=2).astype(np.bool)
            class_ids = np.array(class_ids, dtype=np.int32)
            return mask, class_ids
        else:
            mask = np.empty(shape=[0, 0, 0],dtype=np.bool)
            class_ids = np.empty(shape = [0], dtype=np.int32)
            return mask, class_ids


    def auto_download(self, dataDir, dataType, dataYear):
        """Download the COCO dataset/annotations if requested.
        dataDir: The root directory of the COCO dataset.
        dataType: What to load (train, val, minival, valminusminival)
        dataYear: What dataset year to load (2014, 2017) as a string, not an integer
        Note:
            For 2014, use "train", "val", "minival", or "valminusminival"
            For 2017, only "train" and "val" annotations are available
        """

        # Setup paths and file names
        if dataType == "minival" or dataType == "valminusminival":
            imgDir = "{}/{}{}".format(dataDir, "val", dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, "val", dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format("val", dataYear)
        else:
            imgDir = "{}/{}{}".format(dataDir, dataType, dataYear)
            imgZipFile = "{}/{}{}.zip".format(dataDir, dataType, dataYear)
            imgURL = "http://images.cocodataset.org/zips/{}{}.zip".format(dataType, dataYear)
        # print("Image paths:"); print(imgDir); print(imgZipFile); print(imgURL)

        # Create main folder if it doesn't exist yet
        if not os.path.exists(dataDir):
            os.makedirs(dataDir)

        # Download images if not available locally
        if not os.path.exists(imgDir):
            os.makedirs(imgDir)
            print("Downloading images to " + imgZipFile + " ...")
            with urllib.request.urlopen(imgURL) as resp, open(imgZipFile, 'wb') as out:
                shutil.copyfileobj(resp, out)
            print("... done downloading.")
            print("Unzipping " + imgZipFile)
            with zipfile.ZipFile(imgZipFile, "r") as zip_ref:
                zip_ref.extractall(dataDir)
            print("... done unzipping")
        print("Will use images in " + imgDir)

        # Setup annotations data paths
        annDir = "{}/annotations".format(dataDir)
        if dataType == "minival":
            annZipFile = "{}/instances_minival2014.json.zip".format(dataDir)
            annFile = "{}/instances_minival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/o43o90bna78omob/instances_minival2014.json.zip?dl=0"
            unZipDir = annDir
        elif dataType == "valminusminival":
            annZipFile = "{}/instances_valminusminival2014.json.zip".format(dataDir)
            annFile = "{}/instances_valminusminival2014.json".format(annDir)
            annURL = "https://dl.dropboxusercontent.com/s/s3tw5zcg7395368/instances_valminusminival2014.json.zip?dl=0"
            unZipDir = annDir
        else:
            annZipFile = "{}/annotations_trainval{}.zip".format(dataDir, dataYear)
            annFile = "{}/instances_{}{}.json".format(annDir, dataType, dataYear)
            annURL = "http://images.cocodataset.org/annotations/annotations_trainval{}.zip".format(dataYear)
            unZipDir = dataDir
        # print("Annotations paths:"); print(annDir); print(annFile); print(annZipFile); print(annURL)

        # Download annotations if not available locally
        if not os.path.exists(annDir):
            os.makedirs(annDir)
        if not os.path.exists(annFile):
            if not os.path.exists(annZipFile):
                print("Downloading zipped annotations to " + annZipFile + " ...")
                with urllib.request.urlopen(annURL) as resp, open(annZipFile, 'wb') as out:
                    shutil.copyfileobj(resp, out)
                print("... done downloading.")
            print("Unzipping " + annZipFile)
            with zipfile.ZipFile(annZipFile, "r") as zip_ref:
                zip_ref.extractall(unZipDir)
            print("... done unzipping")
        print("Will use annotations in " + annFile)




def load_image_gt(dataset, image_id, augmentation=None,use_mini_mask=False):
    """Load and return ground truth data for an image (image, mask, bounding boxes).

    augment: (Depricated. Use augmentation instead). If true, apply random
        image augmentation. Currently, only horizontal flipping is offered.
    augmentation: Optional. An imgaug (https://github.com/aleju/imgaug) augmentation.
        For example, passing imgaug.augmenters.Fliplr(0.5) flips images
        right/left 50% of the time.
    use_mini_mask: If False, returns full-size masks that are the same height
        and width as the original image. These can be big, for example
        1024x1024x100 (for 100 instances). Mini masks are smaller, typically,
        224x224 and are generated by extracting the bounding box of the
        object and resizing it to MINI_MASK_SHAPE.

    Returns:
    image: [height, width, 3]
    shape: the original shape of the image before resizing and cropping.
    class_ids: [instance_count] Integer class IDs
    bbox: [instance_count, (y1, x1, y2, x2)]
    mask: [height, width, instance_count]. The height and width are those
        of the image unless use_mini_mask is True, in which case they are
        defined in MINI_MASK_SHAPE.
    """
    # Load image and mask
    image = dataset.load_image(image_id)
    mask, class_ids = dataset.load_mask(image_id)
    original_shape = image.shape
    image, window, scale, padding, crop = resize_image(
        image,
        min_dim=config.IMAGE_MIN_DIM,
        min_scale=config.IMAGE_MIN_SCALE,
        max_dim=config.IMAGE_MAX_DIM,
        mode=config.IMAGE_RESIZE_MODE)
    mask = resize_mask(mask, scale, padding, crop)

    # Augmentation
    # This requires the imgaug lib (https://github.com/aleju/imgaug)
    if augmentation:
        import imgaug

        # Augmentors that are safe to apply to masks
        # Some, such as Affine, have settings that make them unsafe, so always
        # test your augmentation on masks
        MASK_AUGMENTERS = ["Sequential", "SomeOf", "OneOf", "Sometimes",
                           "Fliplr", "Flipud", "CropAndPad",
                           "Affine", "PiecewiseAffine"]

        def hook(images, augmenter, parents, default):
            """Determines which augmenters to apply to masks."""
            return (augmenter.__class__.__name__ in MASK_AUGMENTERS)

        # Store shapes before augmentation to compare
        image_shape = image.shape
        mask_shape = mask.shape
        assert image_shape == mask_shape, "the shape of mask and image must be same"
        # Make augmenters deterministic to apply similarly to images and masks
        det = augmentation.to_deterministic()
        image = det.augment_image(image)
        # Change mask to np.uint8 because imgaug doesn't support np.bool
        mask = det.augment_image(mask.astype(np.uint8),
                                 hooks=imgaug.HooksImages(activator=hook))
        # Verify that shapes didn't change
        assert image.shape == image_shape, "Augmentation shouldn't change image size"
        assert mask.shape == mask_shape, "Augmentation shouldn't change mask size"
        # Change mask back to bool
        mask = mask.astype(np.bool)

    # Note that some boxes might be all zeros if the corresponding mask got cropped out.
    # and here is to filter them out
    _idx = np.sum(mask, axis=(0, 1)) > 0
    mask = mask[:, :, _idx]
    class_ids = class_ids[_idx]
    # Bounding boxes. Note that some boxes might be all zeros
    # if the corresponding mask got cropped out.
    # bbox: [num_instances, (x1, y1, x2, y2)]
    bbox = extract_bboxes(mask) # bbox array [num_instances, (x1, y1, x2, y2)].

    # Active classes
    # Different datasets have different classes, so track the
    # classes supported in the dataset of this image.
    active_class_ids = np.zeros([dataset.num_classes], dtype=np.int32)
    source_class_ids = dataset.source_class_ids[dataset.image_info[image_id]["source"]]
    active_class_ids[source_class_ids] = 1

    # Resize masks to smaller size to reduce memory usage
    if use_mini_mask:
        mask = minimize_mask(bbox, mask, config.MINI_MASK_SHAPE)

    # Image meta data
    image_meta = compose_image_meta(image_id, original_shape, image.shape,
                                    window, scale, active_class_ids)

    return image, image_meta, class_ids, bbox, mask


def compose_image_meta(image_id, original_image_shape, image_shape,
                       window, scale, active_class_ids):
    """Takes attributes of an image and puts them in one 1D array.

    image_id: An int ID of the image. Useful for debugging.
    original_image_shape: [H, W, C] before resizing or padding.
    image_shape: [H, W, C] after resizing and padding
    window: (y1, x1, y2, x2) in pixels. The area of the image where the real
            image is (excluding the padding)
    scale: The scaling factor applied to the original image (float32)
    active_class_ids: List of class_ids available in the dataset from which
        the image came. Useful if training on images from multiple datasets
        where not all classes are present in all datasets.
    """
    meta = np.array(
        [image_id] +                  # size=1
        list(original_image_shape) +  # size=3
        list(image_shape) +           # size=3
        list(window) +                # size=4 (x1, y1, x2, y2) in image cooredinates
        [scale] +                     # size=1
        list(active_class_ids)        # size=num_classes
    )
    return meta





def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    mini_mask = np.zeros(mini_shape + (mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = mask[:, :, i].astype(bool)
        x1, y1, x2, y2 = bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = cv2.resize(m, (mini_shape[1], mini_shape[0]))
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask



def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (x1, y1, x2, y2)].
    """
    boxes = np.zeros([mask.shape[-1], 4], dtype=np.int32)
    for i in range(mask.shape[-1]):
        m = mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([x1, y1, x2, y2])
    return boxes.astype(np.int32)


def data_generator(dataset, random_rois=0,shuffle=True, augmentation=None):
    """
    A generator that returns images and corresponding target class ids,
    bounding box deltas, and masks.
    :param dataset:
    :param shuffle:
    :param augmentation:
    :return:
    """
    b = 0  # batch item index
    image_index = -1
    image_ids = np.copy(dataset.image_ids)  # 这个class里面的图片的编号，从0到张数，并非coco的内置的图片id
    error_count = 0
    # 返回值是[批数，anchor数，(x1, y1, x2, y2)]，相对输入图片的像素坐标
    anchors = generate_pyramid_anchors(batch_size, resolution, input_shape, smallest_anchor_size)
    anchors = anchors[0]  # 只需要取第一批, [num, (x1, y1, x2, y2)]


    # 无限循环提供数据
    while True:
        try:
            # Increment index to pick next image. Shuffle if at the start of an epoch.
            image_index = (image_index + 1) % len(image_ids)
            if shuffle and image_index == 0:
                np.random.shuffle(image_ids)

            # Get GT bounding boxes and masks for image.
            image_id = image_ids[image_index]
            image, image_meta, gt_class_ids, gt_boxes, gt_masks = \
                load_image_gt(dataset, image_id,
                              augmentation=augmentation,
                              use_mini_mask=config.USE_MINI_MASK)

            # Skip images that have no instances. This can happen in cases
            # where we train on a subset of classes and the image doesn't
            # have any of the classes we care about.
            if not np.any(gt_class_ids > 0):
                continue

            # RPN Targets  img_shape, bboxes, gt_class_ids
            rpn_match, rpn_bbox = cls_target2(image.shape, anchors, gt_boxes, gt_class_ids)

            # Mask R-CNN Targets
            # if random_rois:
            #     rpn_rois = generate_random_rois(
            #         image.shape, random_rois, gt_class_ids, gt_boxes)
            #     if detection_targets:
            #         rois, mrcnn_class_ids, mrcnn_bbox, mrcnn_mask =\
            #             build_detection_targets(
            #                 rpn_rois, gt_class_ids, gt_boxes, gt_masks, config)

            # Init batch arrays
            if b == 0:
                batch_image_meta = np.zeros(
                    (batch_size,) + image_meta.shape, dtype=image_meta.dtype)
                batch_rpn_match = np.zeros(
                    [batch_size, anchors.shape[0], 1], dtype=rpn_match.dtype)
                batch_rpn_bbox = np.zeros(
                    [batch_size, config.RPN_TRAIN_ANCHORS_PER_IMAGE, 4], dtype=rpn_bbox.dtype)
                batch_images = np.zeros(
                    (batch_size,) + image.shape, dtype=np.float32)
                batch_gt_class_ids = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES), dtype=np.int32)
                batch_gt_boxes = np.zeros(
                    (batch_size, config.MAX_GT_INSTANCES, 4), dtype=np.int32)
                batch_gt_masks = np.zeros(
                    (batch_size, gt_masks.shape[0], gt_masks.shape[1],
                     config.MAX_GT_INSTANCES), dtype=gt_masks.dtype)
                # if random_rois:
                #     batch_rpn_rois = np.zeros(
                #         (batch_size, rpn_rois.shape[0], 4), dtype=rpn_rois.dtype)
                #     if detection_targets:
                #         batch_rois = np.zeros(
                #             (batch_size,) + rois.shape, dtype=rois.dtype)
                #         batch_mrcnn_class_ids = np.zeros(
                #             (batch_size,) + mrcnn_class_ids.shape, dtype=mrcnn_class_ids.dtype)
                #         batch_mrcnn_bbox = np.zeros(
                #             (batch_size,) + mrcnn_bbox.shape, dtype=mrcnn_bbox.dtype)
                #         batch_mrcnn_mask = np.zeros(
                #             (batch_size,) + mrcnn_mask.shape, dtype=mrcnn_mask.dtype)

            # If more instances than fits in the array, sub-sample from them.
            if gt_boxes.shape[0] > config.MAX_GT_INSTANCES:
                ids = np.random.choice(
                    np.arange(gt_boxes.shape[0]), config.MAX_GT_INSTANCES, replace=False)
                gt_class_ids = gt_class_ids[ids]
                gt_boxes = gt_boxes[ids]
                gt_masks = gt_masks[:, :, ids]

            # Add to batch
            batch_image_meta[b] = image_meta
            batch_rpn_match[b] = rpn_match[:, np.newaxis]
            batch_rpn_bbox[b] = rpn_bbox
            batch_images[b] = mold_image(image.astype(np.float32))
            batch_gt_class_ids[b, :gt_class_ids.shape[0]] = gt_class_ids
            batch_gt_boxes[b, :gt_boxes.shape[0]] = gt_boxes
            batch_gt_masks[b, :, :, :gt_masks.shape[-1]] = gt_masks
            # if random_rois:
            #     batch_rpn_rois[b] = rpn_rois
            #     if detection_targets:
            #         batch_rois[b] = rois
            #         batch_mrcnn_class_ids[b] = mrcnn_class_ids
            #         batch_mrcnn_bbox[b] = mrcnn_bbox
            #         batch_mrcnn_mask[b] = mrcnn_mask
            b += 1

            # Batch full?
            if b >= batch_size:
                inputs = [batch_images, batch_image_meta, batch_rpn_match, batch_rpn_bbox,
                          batch_gt_class_ids, batch_gt_boxes, batch_gt_masks]
                outputs = []

                # if random_rois:
                #     inputs.extend([batch_rpn_rois])
                #     if detection_targets:
                #         inputs.extend([batch_rois])
                #         # Keras requires that output and targets have the same number of dimensions
                #         batch_mrcnn_class_ids = np.expand_dims(
                #             batch_mrcnn_class_ids, -1)
                #         outputs.extend(
                #             [batch_mrcnn_class_ids, batch_mrcnn_bbox, batch_mrcnn_mask])

                yield inputs, outputs

                # start a new batch
                b = 0
        except (GeneratorExit, KeyboardInterrupt):
            raise
        except:
            # Log it and skip the image
            logging.exception("Error processing image {}".format(
                dataset.image_info[image_id]))
            error_count += 1
            if error_count > 5:
                raise


def mold_image(images):
    """Expects an RGB image (or array of images) and subtraces
    the mean pixel and converts it to float. Expects image
    colors in RGB order.
    """
    return images.astype(np.float32) - config.pixel_average


def resize_mask(mask, scale, padding, crop=None):
    """Resizes a mask using the given scale and padding.
    Typically, you get the scale and padding from resize_image() to
    ensure both, the image and the mask, are resized consistently.

    scale: mask scaling factor
    padding: Padding to add to the mask in the formgit
            [(top, bottom), (left, right), (0, 0)]
    """
    mask = mask.astype(np.float32)
    h, w = mask[:2]
    mask = cv2.resize(mask,(np.round(w * scale).astype(np.int32), np.round(h * scale).astype(np.int32)))
    if crop is not None:
        x, y, w, h = crop
        mask = mask[y:y + h, x:x + w]
    else:
        mask = np.pad(mask, padding, mode='constant', constant_values=0)
    mask = round(mask)
    mask = mask.astype(np.bool)
    return mask




def resize_image(image, min_dim=None, max_dim=None, min_scale=None, mode="square"):
    """Resizes an image keeping the aspect ratio unchanged.

    min_dim: if provided, resizes the image such that it's smaller
        dimension == min_dim
    max_dim: if provided, ensures that the image longest side doesn't
        exceed this value.
    min_scale: if provided, ensure that the image is scaled up by at least
        this percent even if min_dim doesn't require it.
    mode: Resizing mode.
        none: No resizing. Return the image unchanged.
        square: Resize and pad with zeros to get a square image
            of size [max_dim, max_dim].此时max_dim必须提供
        pad64: Pads width and height with zeros to make them multiples of 64.
               If min_dim or min_scale are provided, it scales the image up
               before padding. max_dim is ignored in this mode.
               The multiple of 64 is needed to ensure smooth scaling of feature
               maps up and down the 6 levels of the FPN pyramid (2**6=64).
        crop: Picks random crops from the image. First, scales the image based
              on min_dim and min_scale, then picks a random crop of
              size min_dim x min_dim. Can be used in training only.
              max_dim is not used in this mode.

    Returns:
    image: the resized image
    window: (x1, y1, x2, y2). If max_dim is provided, padding might
        be inserted in the returned image. If so, this window is the
        coordinates of the image part of the full image (excluding
        the padding). The x2, y2 pixels are not included.
    scale: The scale factor used to resize the image
    padding: Padding added to the image [(top, bottom), (left, right), (0, 0)]
    """
    # Keep track of image dtype and return results in the same dtype
    image_dtype = image.dtype
    # Default window (y1, x1, y2, x2) and default scale == 1.
    h, w = image.shape[:2]
    window = (0, 0, h, w)
    scale = 1
    padding = [(0, 0), (0, 0), (0, 0)]
    crop = None

    if mode == "none":
        return image, window, scale, padding, crop

    # Scale?
    if min_dim:
        # Scale up but not down scale=缩放后的，除以原图 scale>=1
        scale = max(1, min_dim / min(h, w))
    if min_scale and scale < min_scale:
        scale = min_scale

    # Does it exceed max dim?
    if max_dim and mode == "square":
        image_max = max(h, w)
        if round(image_max * scale) > max_dim:
            scale = max_dim / image_max

    # Resize image using bilinear interpolation
    if scale != 1:
        image = cv2.resize(image, (round(w * scale), round(h * scale)))

    # Need padding or cropping?
    if mode == "square":
        # Get new height and width
        h, w = image.shape[:2]
        top_pad = (max_dim - h) // 2
        bottom_pad = max_dim - h - top_pad
        left_pad = (max_dim - w) // 2
        right_pad = max_dim - w - left_pad
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (left_pad, top_pad, w + left_pad, h + top_pad)
    elif mode == "pad64":
        h, w = image.shape[:2]
        # Both sides must be divisible by 64
        assert min_dim % 64 == 0, "Minimum dimension must be a multiple of 64"
        # Height
        if h % 64 > 0:
            max_h = h - (h % 64) + 64
            top_pad = (max_h - h) // 2
            bottom_pad = max_h - h - top_pad
        else:
            top_pad = bottom_pad = 0
        # Width
        if w % 64 > 0:
            max_w = w - (w % 64) + 64
            left_pad = (max_w - w) // 2
            right_pad = max_w - w - left_pad
        else:
            left_pad = right_pad = 0
        padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
        image = np.pad(image, padding, mode='constant', constant_values=0)
        window = (top_pad, left_pad, h + top_pad, w + left_pad)
    elif mode == "crop":
        # Pick a random crop
        h, w = image.shape[:2]
        y = np.random.randint(0, (h - min_dim))
        x = np.random.randint(0, (w - min_dim))
        crop = (x, y, min_dim, min_dim)
        image = image[y:y + min_dim, x:x + min_dim]
        window = (0, 0, min_dim, min_dim)
    else:
        raise Exception("Mode {} not supported".format(mode))
    return image.astype(image_dtype), window, scale, padding, crop




def get_batch(num_workers, **kwargs):
    try:
        # GeneratorEnqueuer的第一个参数是一个无限产生数据的生成器函数；第二个使用多进程，默认0.05ms产生一个数据
        # 生成器yeild的数据为images, image_fns, score_maps, geo_maps, training_masks
        enqueuer = GeneratorEnqueuer(data_generator(**kwargs), use_multiprocessing=True)
        print('Generator use 10 batches for buffering, this may take a while, you can tune this yourself.')
        # 启动线程，它从生成器中添加数据到队列。max_queue_size：队列大小； workers：工作线程数
        enqueuer.start(max_queue_size=5, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    # Creates a generator to extract data from the queue.
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()

class GeneratorEnqueuer(object):
    """Builds a queue out of a data generator.

    Used in `fit_generator`, `evaluate_generator`, `predict_generator`.

    # Arguments
        generator: a generator function which endlessly yields data
        use_multiprocessing: use multiprocessing if True, otherwise threading
        wait_time: time to sleep in-between calls to `put()`
        random_seed: Initial seed for workers,
            will be incremented by one for each workers.
    """

    def __init__(self, generator,
                 use_multiprocessing=False,
                 wait_time=0.05,
                 random_seed=None):
        self.wait_time = wait_time
        self._generator = generator
        self._use_multiprocessing = use_multiprocessing
        self._threads = []
        self._stop_event = None
        self.queue = None
        self.random_seed = random_seed

    def start(self, workers=1, max_queue_size=10):
        """Kicks off threads which add data from the generator into the queue.

        # Arguments
            workers: number of worker threads
            max_queue_size: queue size
                (when full, threads could block on `put()`)
        """

        def data_generator_task():
            while not self._stop_event.is_set():
                try:
                    if self._use_multiprocessing or self.queue.qsize() < max_queue_size:
                        generator_output = next(self._generator)
                        self.queue.put(generator_output)
                    else:
                        time.sleep(self.wait_time)
                except Exception:
                    self._stop_event.set()
                    raise

        try:
            if self._use_multiprocessing:
                self.queue = multiprocessing.Queue(maxsize=max_queue_size)
                self._stop_event = multiprocessing.Event()
            else:
                self.queue = queue.Queue()
                self._stop_event = threading.Event()

            for _ in range(workers):
                if self._use_multiprocessing:
                    # Reset random seed else all children processes
                    # share the same seed
                    np.random.seed(self.random_seed)
                    thread = multiprocessing.Process(target=data_generator_task)
                    thread.daemon = True
                    if self.random_seed is not None:
                        self.random_seed += 1
                else:
                    thread = threading.Thread(target=data_generator_task)
                self._threads.append(thread)
                thread.start()
        except:
            self.stop()
            raise

    def is_running(self):
        return self._stop_event is not None and not self._stop_event.is_set()

    def stop(self, timeout=None):
        """Stops running threads and wait for them to exit, if necessary.

        Should be called by the same thread which called `start()`.

        # Arguments
            timeout: maximum time to wait on `thread.join()`.
        """
        if self.is_running():
            self._stop_event.set()

        for thread in self._threads:
            if thread.is_alive():
                if self._use_multiprocessing:
                    thread.terminate()
                else:
                    thread.join(timeout)

        if self._use_multiprocessing:
            if self.queue is not None:
                self.queue.close()

        self._threads = []
        self._stop_event = None
        self.queue = None

    def get(self):
        """Creates a generator to extract data from the queue.

        Skip the data if it is `None`.

        # Returns
            A generator
        """
        while self.is_running():
            if not self.queue.empty():
                inputs = self.queue.get()
                if inputs is not None:
                    yield inputs
            else:
                time.sleep(self.wait_time)