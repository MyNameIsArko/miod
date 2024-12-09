import random

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm.auto import tqdm

import time
from dataset import yolo_to_coords
from dataloader import get_dataloader
from ultralytics import YOLO


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    batch_size = 1

    model = YOLO('runs/detect/train4/weights/best.pt')
    val_dataloader = get_dataloader("val", False, batch_size)

    map_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    amount_iou = 0

    time_amount = 0
    avg_pred_time = 0

    for image, bboxes, labels in tqdm(val_dataloader):
        bboxes = bboxes[:, :-1]
        start_time = time.perf_counter()
        results = model(image, imgsz=128)
        end_time = time.perf_counter()

        avg_pred_time += end_time - start_time
        time_amount += 1

        pred_bboxes = []

        for result in results:
            pred_bbox = result.boxes.xyxyn
            pred_bboxes.append(pred_bbox)

        pred_bboxes = torch.stack(pred_bboxes)

        bboxes = bboxes[0].to("cuda")
        bboxes = yolo_to_coords(bboxes)

        pred_bboxes = pred_bboxes[0]

        iou_list = []
        used_preds = []

        for bbox in bboxes:
            bbox = bbox.unsqueeze(0)
            bbox_iou = box_iou(pred_bboxes, bbox)
            if len(bbox_iou) > 0:
                bbox_iou = bbox_iou[0]

            best_iou = 0
            best_pred = -1

            for i, b_iou in enumerate(bbox_iou):
                if i not in used_preds and b_iou > best_iou:
                    best_iou = b_iou
                    best_pred = i
            
            if best_pred != -1:
                iou_list.append(best_iou)
                used_preds.append(best_pred)

        for iou_l in iou_list:
            for j in range(10):
                map_list[j] += 1 if iou_l >= 0.5 + j * 0.05 else 0
            amount_iou += 1

        amount_iou += abs(len(bboxes) - len(pred_bboxes))

    avg_pred_time /= time_amount
    print("TIME:", avg_pred_time)

    map_list = [o / amount_iou for o in map_list]
    print("mAP 0.5:", map_list[0])
    print("mAP 0.5:0.95:0.05:", sum(map_list) / 10)

if __name__ == '__main__':
    main()
