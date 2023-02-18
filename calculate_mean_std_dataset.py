import torch
import torchvision
from torchvision import transforms
import keep_aspect_ratio

# TRAIN_DATA_PATH = "./original_dataset_rgba"
# TRAIN_DATA_PATH = "./non_iid_dataset_rgba/black_bin"
# TRAIN_DATA_PATH = "./non_iid_dataset_rgba/blue_bin"
TRAIN_DATA_PATH = "./non_iid_dataset_rgba/green_bin"

WIDTH = 384
HEIGHT = 380
AR_INPUT = WIDTH / HEIGHT

TRANSFORM_IMG = transforms.Compose([
    keep_aspect_ratio.PadToMaintainAR(aspect_ratio=AR_INPUT),
    transforms.Resize((WIDTH, HEIGHT), transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor()
])


def get_mean_and_std(dataloader):

    channels_sum, channels_squared_sum, num_batches = 0, 0, 0

    for batch_idx, (images, _) in enumerate(dataloader):

        print("Calculating statistics for batch {}".format(batch_idx))
        channels_sum += torch.mean(images, dim=[0, 2, 3])
        channels_squared_sum += torch.mean(images**2, dim=[0, 2, 3])
        num_batches += 1

    mean = channels_sum / num_batches

    # std = sqrt(E[X^2] - (E[X])^2)
    std = torch.sqrt((channels_squared_sum / num_batches) - mean ** 2)

    return mean, std


train_data = torchvision.datasets.ImageFolder(
    root=TRAIN_DATA_PATH, transform=TRANSFORM_IMG)

data_loader_train = torch.utils.data.DataLoader(dataset=train_data,
                                                batch_size=128,
                                                shuffle=True, num_workers=8)

mean, std = get_mean_and_std(data_loader_train)

print("mean: {}".format(mean))
print("std: {}".format(std))
