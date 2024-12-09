import random
import wandb

import numpy as np
import torch
import torch.optim as O
from torch.utils.tensorboard import SummaryWriter
from torchvision.ops import box_iou
from tqdm.auto import tqdm

from dataloader import get_dataloader, get_mask
from dataset import yolo_to_coords, mask_appropriate_image_mask
from loss import normal_loss
from model import BBoxModel
from utils import save_checkpoint, get_linear_interpolation


def main():
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    batch_size = 576
    epochs = 1000
    # get_tf_decay = get_linear_interpolation(epochs // 10, 1.0, 0.0)

    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    print(f"Using {device_str}")
    model = BBoxModel().to(device)
    train_dataloader = get_dataloader("train", True, batch_size, True)
    val_dataloader = get_dataloader("val", True, batch_size, False)
    optim = O.AdamW(model.parameters())

    # checkpoint = torch.load('ckpt/checkpoint_tf_935.pt')
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optim.load_state_dict(checkpoint['optimizer_state_dict'])

    best_scores = []

    # wandb.init(project="inzynierka", sync_tensorboard=True, settings=wandb.Settings(code_dir="."))

    writer = SummaryWriter()

    writer_i = 0

    for epoch in range(epochs):
        model.train()
        amount = 0
        train_loss = 0
        train_acc = 0

        # teacher_forcing = get_tf_decay(epoch)
        # writer.add_scalar("teacher_forcing", teacher_forcing, epoch)

        for image, bbox, label in tqdm(train_dataloader, desc="Train"):
            image = image.to(device)
            bbox = bbox.to(device)
            label = label.to(device)

            batch_size, _, height, width = image.shape

            large_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board
            medium_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board
            small_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board

            amount += bbox.shape[1]

            bbox_loss = 0
            label_loss = 0

            for i in range(bbox.shape[1]):
                i_bbox = bbox[:, i]
                i_label = label[:, i]

                mask = get_mask(i_label)

                pred_bbox, pred_label = model(image, large_image_mask, medium_image_mask, small_image_mask)
                loss, info = normal_loss(pred_bbox, pred_label, i_bbox, i_label, mask)

                bbox_loss += info['bbox_loss']
                label_loss += info['label_loss']

                optim.zero_grad(set_to_none=True)
                loss.backward()
                optim.step()

                with torch.no_grad():
                    # if torch.rand(1) < teacher_forcing:
                    #     mask_appropriate_image_mask(i_bbox, mask, large_image_mask, medium_image_mask, small_image_mask)
                    # else:
                    #     pred_mask = get_mask(pred_label.argmax(dim=-1))
                    #     mask_appropriate_image_mask(pred_bbox, pred_mask, large_image_mask, medium_image_mask, small_image_mask)

                    mask_appropriate_image_mask(i_bbox, mask, large_image_mask, medium_image_mask, small_image_mask)

                    train_loss += loss.item()
                    pred_bbox = yolo_to_coords(pred_bbox)
                    i_bbox = yolo_to_coords(i_bbox)
                    iou = box_iou(pred_bbox, i_bbox)
                    iou = iou[torch.arange(batch_size), torch.arange(batch_size)]
                    iou = iou.masked_fill(mask, 0)
                    iou = iou.sum()
                    iou = iou / ((~mask).sum() + 1e-8)
                    train_acc += iou.item()

            bbox_loss /= bbox.shape[1]
            label_loss /= bbox.shape[1]

            writer.add_scalar("bbox_loss", bbox_loss, writer_i)
            writer.add_scalar("label_loss", label_loss, writer_i)
            writer_i += 1

        train_loss /= amount
        train_acc /= amount

        writer.add_scalar("train_loss", train_loss, epoch)
        writer.add_scalar("train_acc", train_acc, epoch)

        val_acc = 0
        val_loss = 0
        amount = 0

        model.eval()

        for image, bbox, label in tqdm(val_dataloader, desc="Eval"):
            image = image.to(device)
            bbox = bbox.to(device)
            label = label.to(device)

            batch_size, _, height, width = image.shape

            large_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board
            medium_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board
            small_image_mask = torch.ones((batch_size, 1, height, width), device=device) * -1  # Start with -1 board

            amount += bbox.shape[1]

            for i in range(bbox.shape[1]):
                with torch.no_grad():
                    pred_bbox, pred_label = model(image, large_image_mask, medium_image_mask, small_image_mask)

                i_bbox = bbox[:, i]
                i_label = label[:, i]

                mask = get_mask(i_label)
                loss, _ = normal_loss(pred_bbox, pred_label, i_bbox, i_label, mask)

                # if torch.rand(1) < teacher_forcing:
                #     mask_appropriate_image_mask(i_bbox, mask, large_image_mask, medium_image_mask, small_image_mask)
                # else:
                #     pred_mask = get_mask(pred_label.argmax(dim=-1))
                #     mask_appropriate_image_mask(pred_bbox, pred_mask, large_image_mask, medium_image_mask, small_image_mask)

                mask_appropriate_image_mask(i_bbox, mask, large_image_mask, medium_image_mask, small_image_mask)

                val_loss += loss.item()
                pred_bbox = yolo_to_coords(pred_bbox)
                i_bbox = yolo_to_coords(i_bbox)
                iou = box_iou(pred_bbox, i_bbox)
                iou = iou[torch.arange(batch_size), torch.arange(batch_size)]
                iou = iou.masked_fill(mask, 0)
                iou = iou.sum()
                iou = iou / ((~mask).sum() + 1e-8)
                val_acc += iou.item()

        val_loss /= amount
        val_acc /= amount

        writer.add_scalar("val_loss", val_loss, epoch)
        writer.add_scalar("val_acc", val_acc, epoch)

        save_checkpoint(model, optim, val_acc, epoch, best_scores)

        print(f"Episode {epoch} complete")


if __name__ == '__main__':
    main()
