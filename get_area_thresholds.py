import numpy as np
from tqdm.auto import tqdm

from dataset import COCODataset, PascalDataset


def divide_into_buckets(areas):
    # Sort areas
    areas.sort()
    n = len(areas)

    # Calculate bucket sizes
    if n % 3 == 2:
        size = n // 3 + 1
        middle_sep = n - size + 1
    else:
        size = n // 3
        middle_sep = n - size

    buckets = [areas[:size], areas[size:middle_sep], areas[middle_sep:]]

    return buckets

def process_dataset(dataset):
    areas = []
    for i in tqdm(range(len(dataset))):
        _, bboxes, _ = dataset[i]
        area = bboxes[:, 2] * bboxes[:, 3]
        areas.extend(area.tolist())
    buckets = divide_into_buckets(areas)
    small_medium_separation = (buckets[0][-1] + buckets[1][0]) / 2
    medium_large_separation = (buckets[1][-1] + buckets[2][0]) / 2

    # Calculate average separating areas for the entire dataset
    small_medium_separation = np.array(small_medium_separation)
    medium_large_separation = np.array(medium_large_separation)
    average_small_medium_separation = np.mean(small_medium_separation)
    average_medium_large_separation = np.mean(medium_large_separation)
    return average_small_medium_separation, average_medium_large_separation

dataset = COCODataset(use_augmentation=False)
average_separating_areas = process_dataset(dataset)
print("Average separating areas for the buckets:", average_separating_areas)
