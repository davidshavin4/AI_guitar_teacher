import torch
import torchvision.datasets as datasets
import os
from torch.utils.data import WeightedRandomSampler, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn

def get_loader(root_dir, batch_size=100):
    my_transforms = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ]
    )
    dataset = datasets.ImageFolder(root=root_dir, transform=my_transforms)

    class_weights = []
    for i, folder in enumerate(dataset.classes):
        num_class = len(os.listdir(os.path.join(root_dir, folder)))
        class_weights.append(1/num_class)

    print('class_weights: ', class_weights)

    sample_weights = [0] * len(dataset)

    for idx, (data, label) in enumerate(dataset):
        #print(f"idx: {idx} label: {label}")
        class_weight = class_weights[label]
        sample_weights[idx] = class_weight

    #print('sample_weights\n', sample_weights)

    sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
    loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler)

    return loader


def main():
    dl = get_loader("datasets/simple_dataset_by_chord_post_crop")

    labels_count = dict()
    for data, labels in dl:
        for label in labels:
            if label.item() in labels_count:
                labels_count[label.item()] += 1
            else:
                labels_count[label.item()] = 0
    print(labels_count)
    for data, labels in dl:
        print(labels)

if __name__=="__main__":
    dataset = datasets.ImageFolder(
        root="datasets/simple_dataset_by_chord_post_crop",
        transform=transforms.ToTensor())
    print(dataset.classes)
    main()

