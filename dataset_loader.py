import os
import random

import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class BrainTumorDataset(Dataset):
    """Simple custom dataset that returns transformed images and labels."""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # Folder names and their numeric labels.
        class_map = {
            "yes": 1,
            "no": 0,
        }

        # Read all image file paths from the yes/ and no/ folders.
        for folder_name, label in class_map.items():
            folder_path = os.path.join(root_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            for file_name in sorted(os.listdir(folder_path)):
                file_path = os.path.join(folder_path, file_name)
                if os.path.isfile(file_path):
                    self.image_paths.append(file_path)
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        # Open the image and convert it to RGB format.
        image = Image.open(self.image_paths[index]).convert("RGB")
        label = self.labels[index]

        # Apply resize and tensor conversion if a transform is provided.
        if self.transform is not None:
            image = self.transform(image)

        return image, label


def count_images(data_dir):
    """Print the number of images in each class folder."""
    yes_dir = os.path.join(data_dir, "yes")
    no_dir = os.path.join(data_dir, "no")

    yes_count = len([name for name in os.listdir(yes_dir) if os.path.isfile(os.path.join(yes_dir, name))])
    no_count = len([name for name in os.listdir(no_dir) if os.path.isfile(os.path.join(no_dir, name))])

    print(f"Number of tumor images (yes): {yes_count}")
    print(f"Number of no tumor images (no): {no_count}")


def show_random_images(data_dir, num_images=5):
    """Randomly display a few images with their class names."""
    image_records = []

    for folder_name, label_name in [("yes", "Tumor"), ("no", "No Tumor")]:
        folder_path = os.path.join(data_dir, folder_name)
        for file_name in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_name)
            if os.path.isfile(file_path):
                image_records.append((file_path, label_name))

    sample_size = min(num_images, len(image_records))
    random_samples = random.sample(image_records, sample_size)

    plt.figure(figsize=(15, 5))
    for index, (file_path, label_name) in enumerate(random_samples, start=1):
        image = Image.open(file_path).convert("RGB")
        plt.subplot(1, sample_size, index)
        plt.imshow(image)
        plt.title(label_name)
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def create_dataloader(data_dir, batch_size=8):
    """Create a dataset and dataloader with basic torchvision transforms."""

    # Resize every image to 224x224 and convert it to a tensor.
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
        ]
    )

    dataset = BrainTumorDataset(root_dir=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataset, dataloader


def main():
    data_dir = os.path.join("dataset", "brain_tumor_dataset")

    # Step 1: print the number of images in each folder.
    count_images(data_dir)

    # Step 2: display 5 random images from the dataset.
    show_random_images(data_dir, num_images=5)

    # Step 3: create the transformed dataset and dataloader.
    dataset, dataloader = create_dataloader(data_dir)

    print(f"\nTotal images loaded: {len(dataset)}")

    # Display one batch shape so it is clear the loader returns images and labels.
    images, labels = next(iter(dataloader))
    print(f"Batch image tensor shape: {images.shape}")
    print(f"Batch labels: {labels}")


if __name__ == "__main__":
    main()