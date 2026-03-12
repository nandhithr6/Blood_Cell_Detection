from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from lxml import etree
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F


CLASS_TO_ID = {
    "RBC": 1,
    "WBC": 2,
    "Platelets": 3,
}

ID_TO_CLASS = {
    1: "RBC",
    2: "WBC",
    3: "Platelets",
}


def parse_voc_xml(xml_path: Path) -> Tuple[List[List[float]], List[int]]:
    """Read one Pascal VOC XML file and return boxes and numeric labels."""
    tree = etree.parse(str(xml_path))
    root = tree.getroot()

    boxes: List[List[float]] = []
    labels: List[int] = []

    for obj in root.findall("object"):
        class_name = obj.findtext("name", default="").strip()
        if class_name not in CLASS_TO_ID:
            continue

        bndbox = obj.find("bndbox")
        if bndbox is None:
            continue

        xmin = float(bndbox.findtext("xmin", default="0"))
        ymin = float(bndbox.findtext("ymin", default="0"))
        xmax = float(bndbox.findtext("xmax", default="0"))
        ymax = float(bndbox.findtext("ymax", default="0"))

        if xmax <= xmin or ymax <= ymin:
            continue

        boxes.append([xmin, ymin, xmax, ymax])
        labels.append(CLASS_TO_ID[class_name])

    return boxes, labels


class BCCDDataset(Dataset):
    """Dataset loader for the BCCD blood cell detection dataset."""

    def __init__(self, root_dir: str = "BCCD_Dataset", resize_to: Optional[Tuple[int, int]] = None):
        self.root_dir = Path(root_dir)
        if not (self.root_dir / "JPEGImages").exists() and (self.root_dir / "BCCD").exists():
            self.root_dir = self.root_dir / "BCCD"

        self.image_dir = self.root_dir / "JPEGImages"
        self.annotation_dir = self.root_dir / "Annotations"
        self.resize_to = resize_to

        if not self.image_dir.exists() or not self.annotation_dir.exists():
            raise FileNotFoundError(
                "BCCD_Dataset folder was not found with 'JPEGImages' and 'Annotations'. "
                "Place the cloned dataset at the project root as BCCD_Dataset/."
            )

        self.image_paths = sorted(
            [path for path in self.image_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"}]
        )

        self.samples = []
        for image_path in self.image_paths:
            xml_path = self.annotation_dir / f"{image_path.stem}.xml"
            if xml_path.exists():
                self.samples.append((image_path, xml_path))

        if not self.samples:
            raise FileNotFoundError("No image and XML annotation pairs were found inside BCCD_Dataset/.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, xml_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        boxes, labels = parse_voc_xml(xml_path)

        if self.resize_to is not None:
            image, boxes = self._resize_image_and_boxes(image, boxes, self.resize_to)

        filtered_boxes: List[List[float]] = []
        filtered_labels: List[int] = []
        for box, label in zip(boxes, labels):
            xmin, ymin, xmax, ymax = box
            if xmax <= xmin or ymax <= ymin:
                continue
            filtered_boxes.append(box)
            filtered_labels.append(label)

        boxes = filtered_boxes
        labels = filtered_labels

        boxes_tensor = torch.as_tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.as_tensor(labels, dtype=torch.int64)

        if boxes_tensor.numel() == 0:
            boxes_tensor = torch.zeros((0, 4), dtype=torch.float32)
            area_tensor = torch.zeros((0,), dtype=torch.float32)
        else:
            area_tensor = (boxes_tensor[:, 2] - boxes_tensor[:, 0]) * (boxes_tensor[:, 3] - boxes_tensor[:, 1])

        target = {
            "boxes": boxes_tensor,
            "labels": labels_tensor,
            "image_id": torch.tensor([index]),
            "area": area_tensor,
            "iscrowd": torch.zeros((len(labels_tensor),), dtype=torch.int64),
            "image_path": str(image_path),
        }

        image_tensor = F.to_tensor(image)
        return image_tensor, target

    @staticmethod
    def _resize_image_and_boxes(
        image: Image.Image,
        boxes: List[List[float]],
        new_size: Tuple[int, int],
    ) -> Tuple[Image.Image, List[List[float]]]:
        """Resize the image and scale the annotation boxes to match."""
        original_width, original_height = image.size
        new_width, new_height = new_size
        resized_image = image.resize(new_size)

        if not boxes:
            return resized_image, boxes

        scale_x = new_width / float(original_width)
        scale_y = new_height / float(original_height)

        resized_boxes = []
        for xmin, ymin, xmax, ymax in boxes:
            resized_boxes.append(
                [xmin * scale_x, ymin * scale_y, xmax * scale_x, ymax * scale_y]
            )

        return resized_image, resized_boxes


def create_dataloader(
    root_dir: str = "BCCD_Dataset",
    resize_to: Optional[Tuple[int, int]] = (512, 512),
    batch_size: int = 2,
    shuffle: bool = False,
) -> Tuple[BCCDDataset, DataLoader]:
    """Create a dataloader for simple experiments and evaluation."""
    dataset = BCCDDataset(root_dir=root_dir, resize_to=resize_to)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=lambda batch: tuple(zip(*batch)))
    return dataset, dataloader