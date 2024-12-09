import torch.nn.functional as F
from torchvision.ops import complete_box_iou_loss
from dataset import yolo_to_coords


def normal_loss(input_bbox, input_label, target_bbox, target_label, mask):
    label_loss = F.cross_entropy(input_label, target_label, ignore_index=2)

    input_bbox = yolo_to_coords(input_bbox)
    target_bbox = yolo_to_coords(target_bbox)
    bbox_loss = complete_box_iou_loss(input_bbox, target_bbox)
    bbox_loss = bbox_loss.masked_fill(mask, 0)
    bbox_loss = bbox_loss.mean()

    loss = 0.01 * label_loss + bbox_loss

    return loss, {'label_loss': label_loss.item(), 'bbox_loss': bbox_loss.item()}
