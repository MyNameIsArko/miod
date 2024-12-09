import torch
import torch.utils.data as D

from dataset import COCODataset

def get_mask(input: torch.tensor):
    return input != 1


def collate_fn(batch):
    img = [b[0] for b in batch]
    bbox = [b[1] for b in batch]
    label = [b[2] for b in batch]

    img = torch.stack(img)

    bbox_new = []
    label_new = []

    # +1 for pad/end token
    max_length = max([len(box) + 1 for box in bbox])

    for box, lbl in zip(bbox, label):
        pad_box = torch.ones(max_length - len(box), 4)
        pad_box[:, :2] = 0  # [0, 0, 1, 1]
        tmp_box = torch.cat([box, pad_box])
        bbox_new.append(tmp_box)
        end_label = torch.zeros((1,), dtype=torch.long)
        tmp_label = torch.cat([lbl, end_label])
        if max_length - len(box) - 1 > 0:
            pad_label = torch.ones((max_length - len(box) - 1,), dtype=torch.long) * 2  # PAD_IDX is 2
            tmp_label = torch.cat([tmp_label, pad_label])
        label_new.append(tmp_label)

    bbox = torch.stack(bbox_new)
    label = torch.stack(label_new)

    return img, bbox, label


def get_dataloader(subset="train", use_augmentation=True, batch_size=64, shuffle=True):
    dataset = COCODataset(subset, use_augmentation)
    return D.DataLoader(dataset, batch_size, shuffle=shuffle, collate_fn=collate_fn,
                        pin_memory=True, num_workers=8, persistent_workers=True)


if __name__ == '__main__':
    val_loader = get_dataloader("val")
    imgs, bboxes, labes = next(iter(val_loader))