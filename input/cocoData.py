import os
import urllib.request
import shutil
import zipfile
from pycocotools.coco import COCO
class CocoDataset(object):
    def __init__(self):
        self._image_ids = []
        # 字典列表， source是"coco", 图片id，图片所存放的地址,图片的高，宽，标注
        self.image_info = []
        # 字典列表，source永远是"coco"，类别id，这个id对应的类别的英文名
        self.class_info = [{"source":"", "id": 0, "name": "BG"}]
        self.source_class_ids = {}

    def add_image(self, source, image_id, path, **kwargs):
        image_info = {"id": image_id, "source": source, "path": path}
        image_info.update(kwargs)
        self.image_info.append(image_info)

    def get_imgInfo_byID(self, imgID):
        for img_info in self.image_info:
            if img_info["id"] == imgID:
                return img_info
        return None


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
        :param dataset_dir: 数据集所在路径名
        :param subset: "train" or "val"
        :param year: 暂且只有"2017"
        :param class_ids: 只返回给定的类别的图像，否则全部返回
        :param return_coco: 是否返回coco
        :param auto_download: 是否下载
        :return:
        """
        assert subset in ("train", "val"), "subset must be either 'train' or 'val' !"
        if auto_download:
            self.auto_download(dataset_dir, subset, year)

        image_dir = "{}/{}{}".format(dataset_dir, subset, year)
        coco = COCO("{}/annotations/instances_{}{}.json".format(dataset_dir, subset, year))
        if not class_ids:
            # 获取全部的类别，val2017的类别标注是从1到90
            class_ids = sorted(coco.getCatIds())

        if class_ids:
            # 根据类别ids来获取图片ids
            image_ids = []
            for class_id in class_ids:
                image_ids.extend(list(coco.getImgIds(catIds=[class_id])))
            image_ids = list(set(image_ids))  # 去重
        else:  # 获取全部的图片ids
            image_ids = list(coco.imgs.keys())

        # 把类别添加进去
        for i in class_ids:
            self.add_class("coco", i, coco.loadCats(i)[0]["name"])

        for i in image_ids:
            self.add_image(
                "coco",
                image_id=i,
                path=os.path.join(image_dir, coco.imgs[i]['file_name']),
                height=coco.imgs[i]["height"],
                width=coco.imgs[i]["width"],
                annotations = coco.loadAnns(coco.getAnnIds([i], class_ids, iscrowd=None))
            )

        if return_coco:
            return coco

    def load_mask(self, image_id):
        image_info = self.get_imgInfo_byID(image_id)
        instance_masks = []
        class_ids = []
        annotations = image_info["annotations"]


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