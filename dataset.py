from pathlib import Path
import logging
logging.getLogger("root").setLevel(logging.ERROR)
from pascal import annotation_from_xml

import torch
import torch.utils.data as D
from pycocotools.coco import COCO
from torchvision.io import read_image
import torchvision.transforms.v2 as v2
from torchvision import tv_tensors
from matplotlib.patches import Rectangle
from torchvision.ops import box_convert

img_res = 128
image_mean = torch.tensor([0.457])
image_std = torch.tensor([0.256])

def yolo_to_coords(bbox):
    bbox = bbox.clone()
    bbox = box_convert(bbox, 'cxcywh', 'xyxy')

    return bbox

def inv_normalize_image(img):
    return (img * image_std + image_mean).clip(0, 1)

def mask_appropriate_image_mask(bboxes, mask, large_image_mask, medium_image_mask, small_image_mask):
    area = bboxes[:, 2] * bboxes[:, 3]

    large_area = area > 0.08
    medium_area = (area <= 0.08) & (area > 0.025)
    small_area = area <= 0.025

    if torch.any(large_area):
        large_image_mask[large_area] = mask_with_bbox(large_image_mask[large_area], bboxes[large_area], mask[large_area])
    if torch.any(medium_area):
        medium_image_mask[medium_area] = mask_with_bbox(medium_image_mask[medium_area], bboxes[medium_area], mask[medium_area])
    if torch.any(small_area):
        small_image_mask[small_area] = mask_with_bbox(small_image_mask[small_area], bboxes[small_area], mask[small_area])


def mask_with_bbox(image_mask, bboxes, mask):
    bboxes = (bboxes * img_res).int()

    batch_size, num_channels, height, width = image_mask.shape

    coords = torch.meshgrid(torch.arange(height), torch.arange(width), indexing='ij')
    coords = torch.stack(coords, dim=-1).to(image_mask.device)
    coords = coords.unsqueeze(0).unsqueeze(0).expand(batch_size, num_channels, -1, -1, -1)

    bboxes = bboxes.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(batch_size, num_channels, height, width, -1)

    x = bboxes[..., 0]
    y = bboxes[..., 1]
    w = bboxes[..., 2]
    h = bboxes[..., 3]

    x1 = x - w / 2
    y1 = y - h / 2
    x2 = x + w / 2
    y2 = y + h / 2

    bbox_mask = ((coords[..., 1] >= x1)
            & (coords[..., 1] <= x2)
            & (coords[..., 0] >= y1)
            & (coords[..., 0] <= y2))

    mask = mask.unsqueeze(1).unsqueeze(1).unsqueeze(1).expand(-1, num_channels, height, width)

    bbox_mask = bbox_mask.masked_fill(mask, 0)

    image_mask = image_mask.masked_fill(bbox_mask, 1)
    return image_mask

def bbox_to_plt(bbox, color='r'):
    bbox = [int(val * img_res) for val in bbox]

    x, y, w, h = bbox
    x -= w / 2
    y -= h / 2

    rect = Rectangle((x, y), w, h, linewidth=5, edgecolor=color, facecolor='none')
    return rect


class COCODataset(D.Dataset):
    def __init__(self, subset="train", use_augmentation=True, min_area=1500., max_area=60000.):
        super().__init__()

        annFile = f"coco/annotations/instances_{subset}2017.json"
        self.coco = COCO(annFile)

        categories = self.coco.getCatIds(supNms="vehicles")

        self.img_idxs = set()

        for cat in categories:
            img_idxs = self.coco.getImgIds(catIds=cat)

            for img_idx in img_idxs:
                ann_idxs = self.coco.getAnnIds(imgIds=img_idx, catIds=cat, areaRng=[min_area, max_area])
                if len(ann_idxs) > 0:
                    self.img_idxs.add(img_idx)

        self.img_idxs = list(self.img_idxs)

        self.categories = categories
        self.min_area = min_area
        self.max_area = max_area
        self.subset = subset

        self.images_dir = f"coco/{subset}2017"

        if use_augmentation:
            self.transform = v2.Compose([
                v2.Grayscale(),
                v2.Resize((img_res, img_res)),
                v2.RandomErasing(scale=(0.02,0.2)),
                v2.RandomHorizontalFlip(),
                v2.GaussianBlur(kernel_size=(5, 5), sigma=(0.1, 0.2)),
                # v2.RandomPerspective(distortion_scale=0.2),
                v2.RandomAffine(degrees=(-15, 15), translate=(0.01, 0.1), scale=(0.9, 1.05)),
                v2.RandomPerspective(distortion_scale=0.2),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(labels_getter=None),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(image_mean, image_std),
            ])
        else:
            self.transform = v2.Compose([
                v2.Grayscale(),
                v2.Resize((img_res, img_res)),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(image_mean, image_std),
            ])

    @staticmethod
    def _sort_boxes(boxes):
        boxes = sorted(boxes, key=lambda x: (x[2] * x[3]))
        return boxes

    def __getitem__(self, idx):
        img_idx = self.img_idxs[idx]
        img_file_name = self.coco.loadImgs(img_idx)[0]["file_name"]
        img = read_image(f"{self.images_dir}/{img_file_name}")

        _, height, width = img.shape

        anns_idxs = self.coco.getAnnIds(imgIds=img_idx, catIds=self.categories, areaRng=[self.min_area, self.max_area])
        anns = self.coco.loadAnns(anns_idxs)

        bboxes = []
        for ann in anns:
            bbox = ann["bbox"]
            bbox = [bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2, bbox[2], bbox[3]]

            bboxes.append(bbox)

        bboxes = self._sort_boxes(bboxes)

        bboxes = tv_tensors.BoundingBoxes(bboxes, format="CXCYWH", canvas_size=img.shape[-2:])

        img, bboxes = self.transform(img, bboxes)

        bboxes /= img_res

        labels = torch.ones((len(bboxes),), dtype=torch.long)

        return img, bboxes, labels


    def __len__(self):
        return len(self.img_idxs)


class PascalDataset(D.Dataset):
    def __init__(self, subset="train", use_augmentation=True):
        super().__init__()

        attr_type_spec = {"truncated": bool, "difficult": bool}
        label_map = {"aeroplane": 0, "bicycle": 1, "boat": 2, "bus": 3, "car": 4, "motorbike": 5, "train": 6}

        ds_path = Path("pascal")

        img_src = ds_path / "JPEGImages"
        ann_src = ds_path / "Annotations"

        img_list_src = ds_path / "ImageSets" / "Main"

        img_list_file = (img_list_src / f"{subset}").with_suffix(".txt")
        with open(img_list_file, "r") as file:
            img_list = file.readlines()
            img_list = [Path(im.strip()).with_suffix(".jpg") for im in img_list]

        self.img_filenames = []
        self.yolo_anns = {}

        for img in img_list:
            ann_file = (ann_src / img.name).with_suffix(".xml")
            ann = annotation_from_xml(ann_file, attr_type_spec)
            yolo_ann = ann.to_yolo(label_map)
            if len(yolo_ann) == 0:
                continue

            yolo_list = []
            for object_str in yolo_ann.split("\n"):
                object_list = object_str.split()
                yolo_list.append(
                    [float(object_list[1]), float(object_list[2]), float(object_list[3]),
                     float(object_list[4])])

            self.img_filenames.append(img)
            self.yolo_anns[img] = yolo_list

            self.img_src = img_src

        self.resize_augmentation = v2.Compose([
            v2.Grayscale(),
            v2.Resize((img_res, img_res)),
        ])

        if use_augmentation:
            self.transform = v2.Compose([
                v2.RandomHorizontalFlip(),
                v2.RandomAffine(degrees=(-30, 30), translate=(0.1, 0.2), scale=(0.8, 1.)),
                v2.ClampBoundingBoxes(),
                v2.SanitizeBoundingBoxes(labels_getter=None),
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(image_mean, image_std),
            ])
        else:
            self.transform = v2.Compose([
                v2.ToDtype(torch.float32, scale=True),
                v2.Normalize(image_mean, image_std),
            ])

    @staticmethod
    def _sort_boxes(boxes):
        boxes = sorted(boxes, key=lambda x: (x[1], x[0]))
        return boxes

    def __getitem__(self, idx):
        img_filename = self.img_filenames[idx]
        img = read_image(self.img_src / img_filename)

        bboxes = self.yolo_anns[img_filename]

        bboxes = self._sort_boxes(bboxes)

        img = self.resize_augmentation(img)

        bboxes = torch.tensor(bboxes)
        bboxes *= img_res

        bboxes = tv_tensors.BoundingBoxes(bboxes, format="CXCYWH", canvas_size=img.shape[-2:])

        img, bboxes = self.transform(img, bboxes)

        bboxes /= img_res

        labels = torch.ones((len(bboxes),), dtype=torch.long)

        return img, bboxes, labels


    def __len__(self):
        return len(self.img_filenames)


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    dataset = COCODataset('val', True)
    # dataset = PascalDataset('val', False)

    image, bboxes, labels = dataset[31]

    image = image.unsqueeze(0)

    image = image.squeeze(0).permute(1, 2, 0)
    image = inv_normalize_image(image)
    plt.imshow(image, cmap='gray')
    plt.show()
    assert False

    height, width, _ = image.shape

    large_image_mask = torch.ones((1, 1, height, width)) * -1
    medium_image_mask = torch.ones((1, 1, height, width)) * -1
    small_image_mask = torch.ones((1, 1, height, width)) * -1

    for bbox in bboxes:
        plt.imshow(image, cmap='gray')

        print(bbox[2] * bbox[3])

        mask_appropriate_image_mask(bbox.unsqueeze(0), torch.zeros((1,), dtype=torch.bool), large_image_mask, medium_image_mask, small_image_mask)

        rect = bbox_to_plt(bbox)
        plt.gca().add_patch(rect)

        plt.show()
        plt.imshow(large_image_mask.squeeze(0).permute(1, 2, 0), cmap="gray")
        plt.show()
        plt.imshow(medium_image_mask.squeeze(0).permute(1, 2, 0), cmap="gray")
        plt.show()
        plt.imshow(small_image_mask.squeeze(0).permute(1, 2, 0), cmap="gray")
        plt.show()