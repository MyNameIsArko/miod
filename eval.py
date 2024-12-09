import random
import time

import numpy as np
import torch
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from dataloader import get_dataloader, get_mask
from dataset import yolo_to_coords, mask_appropriate_image_mask
from model import BBoxModel


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    batch_size = 1

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using {device_str}")
    model = BBoxModel().to(device)
    val_dataloader = get_dataloader("val", False, batch_size, False)

    checkpoint = torch.load('ckpt/checkpoint_140.pt')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    map_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    amount_iou = 0

    time_amount = 0
    avg_pred_time = 0

    for image, bbox, label in tqdm(val_dataloader, desc="Eval"):
        image = image.to(device)
        bbox = bbox.to(device)
        label = label.to(device)

        batch_size, _, height, width = image.shape

        large_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board
        medium_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board
        small_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board

        pred_bboxes = []

        tmp_pred_time = 0

        for i in range(bbox.shape[1] - 1):
            with torch.no_grad():
                start_time = time.perf_counter()
                pred_bbox, pred_label = model(image, large_image_mask, medium_image_mask, small_image_mask)

                i_bbox = bbox[:, i]
                i_label = label[:, i]

                mask = get_mask(i_label)

                mask_appropriate_image_mask(i_bbox, mask, large_image_mask, medium_image_mask, small_image_mask)

                end_time = time.perf_counter()
                tmp_pred_time += end_time - start_time


            if torch.argmax(pred_label[0]) == 1:
                pred_bboxes.append(pred_bbox[0])
            else:
                avg_pred_time += tmp_pred_time / (i + 1)
                time_amount += 1
                break

        if len(pred_bboxes) == 0:
            amount_iou += len(bbox[0, :-1])
            continue

        pred_bboxes = torch.stack(pred_bboxes)

        pred_bboxes = yolo_to_coords(pred_bboxes)

        bboxes = yolo_to_coords(bbox[0, :-1])

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
