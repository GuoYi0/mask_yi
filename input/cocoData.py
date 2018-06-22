import os
import urllib.request
import shutil
import zipfile
from pycocotools.coco import COCO
import numpy as np



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
        # 字典列表， source是"coco", 图片id，图片所存放的地址, 图片的高，宽，标注
        self.image_info = []
        # 字典列表，source永远是"coco"，类别id，这个id对应的类别的英文名
        self.class_info = [{"source":"", "id": 0, "name": "BG"}]
        self.source_class_ids = {}
        if auto_download is True:
            self.auto_download(dataset_dir, subset, year)
        self.coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        self.load_coco(dataset_dir, subset, year,class_ids,return_coco, auto_download)

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

    def load_coco(self, dataset_dir, subset, year="2017", class_ids=None,return_coco=False, auto_download=False):
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

        image_dir = "{}/{}{}".format(dataset_dir, subset, year)

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





    def load_mask(self, image_id):
        """
        给定图片编号，获取mask
        :param image_id: 图片编号
        :return:
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
            mask = np.empty(shape=[0, 0, 0])
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