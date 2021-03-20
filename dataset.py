import numpy as np
import torch
from torchvision import datasets, transforms
import tqdm

class_list = [
    "aeroplane",
    "bicycle",
    "bird",
    "boat",
    "bottle",
    "bus",
    "car",
    "cat",
    "chair",
    "cow",
    "diningtable",
    "dog",
    "horse",
    "motorbike",
    "person",
    "pottedplant",
    "sheep",
    "sofa",
    "train",
    "tvmonitor"
]

class_map = {name: id + 10 for id, name in enumerate(class_list)}


class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, image_set):
        self.transform = transforms.Resize((448, 448))
        self.dataset = datasets.VOCDetection(
            root="./data/",
            year="2012",
            image_set=image_set,
            download=False,
            transform=transforms.ToTensor())

    def __getitem__(self, item: int):
        image, target = self.dataset[item]
        target_tensor = torch.zeros((7, 7, 30), dtype=torch.float32)
        image_height, image_width = image.shape[-2:]
        xscale = 1 / image_width
        yscale = 1 / image_height
        for object in target["annotation"]["object"]:
            xmin, xmax, ymin, ymax = [int(object["bndbox"][key]) for key in ["xmin", "xmax", "ymin", "ymax"]]
            xmin, xmax, ymin, ymax = xmin * xscale, xmax * xscale, ymin * yscale, ymax * yscale
            centerx, centery = (xmax + xmin) * 0.5, (ymax + ymin) * 0.5
            gridx, gridy = np.floor(centerx * 7).astype(np.int32), np.floor(centery * 7).astype(np.int32)
            width, height = xmax - xmin, ymax - ymin
            target_tensor[gridx, gridy, :5] = torch.from_numpy(np.array([
                centerx - gridx / 7,
                centery - gridy / 7,
                width,
                height,
                1.0], dtype=np.float32))
            target_tensor[gridx, gridy, 5:10] = target_tensor[gridy, gridx, :5]
            class_id = class_map[object["name"]]
            target_tensor[gridx, gridy, class_id] = 1
        return self.transform(image), target_tensor

    def __len__(self):
        return len(self.dataset)


def main():
    data = VOCDataset("trainval")
    num_samples = len(data)
    train_data, valid_data, extern_data = torch.utils.data.random_split(data, [num_samples//3, num_samples//3, num_samples - 2 * (num_samples // 3)])
    train_queue = torch.utils.data.DataLoader(train_data, batch_size=5, pin_memory=False)
    for images, targets in tqdm.tqdm(train_queue):
        pass


if __name__ == '__main__':
    main()
