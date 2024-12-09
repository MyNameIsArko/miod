import os
import torch
import wandb

from dataset import yolo_to_coords


def save_checkpoint(model, optimizer, val_score, epoch, best_scores, checkpoint_dir='ckpt'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_{epoch}.pt')

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_score': val_score
    }, checkpoint_path)

    # wandb.save(checkpoint_path)

    best_scores.append((val_score, checkpoint_path))
    best_scores.sort(key=lambda x: x[0], reverse=True)

    # If there are more than 3 best scores, remove the oldest one
    if len(best_scores) > 3:
        _, path_to_remove = best_scores.pop()
        if os.path.exists(path_to_remove):
            os.remove(path_to_remove)

    return best_scores

def get_linear_interpolation(decay_epochs, start_value, end_value):
    def fn(current):
        if current < decay_epochs:
            return start_value + (end_value - start_value) * (current / decay_epochs)
        else:
            return end_value

    return fn