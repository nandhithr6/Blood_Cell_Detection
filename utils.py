import os
import urllib.request
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights, fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F

from dataset import ID_TO_CLASS


ImageInput = Union[str, Image.Image]
DEFAULT_CHECKPOINT_PATH = "checkpoints/bccd_fasterrcnn.pth"


def get_device() -> torch.device:
    """Use GPU when available, otherwise run on CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_checkpoint_url() -> Optional[str]:
    """Read a checkpoint download URL from the environment when available."""
    return os.getenv("BCCD_CHECKPOINT_URL") or os.getenv("CHECKPOINT_URL")


def download_checkpoint(checkpoint_url: str, destination_path: str = DEFAULT_CHECKPOINT_PATH) -> str:
    """Download a checkpoint file only when it is not already present locally."""
    if os.path.exists(destination_path):
        return destination_path

    os.makedirs(os.path.dirname(destination_path), exist_ok=True)
    urllib.request.urlretrieve(checkpoint_url, destination_path)
    return destination_path


def ensure_checkpoint_available(
    checkpoint_path: Optional[str] = None,
    checkpoint_url: Optional[str] = None,
) -> Optional[str]:
    """Return a usable checkpoint path, downloading it first when configured."""
    resolved_path = resolve_checkpoint_path(checkpoint_path)
    if resolved_path and os.path.exists(resolved_path):
        return resolved_path

    checkpoint_url = checkpoint_url or get_checkpoint_url()
    if checkpoint_url:
        destination_path = checkpoint_path or DEFAULT_CHECKPOINT_PATH
        return download_checkpoint(checkpoint_url, destination_path=destination_path)

    return resolved_path


def resolve_checkpoint_path(checkpoint_path: Optional[str] = None) -> Optional[str]:
    """Use the provided checkpoint, or a default checkpoint if one exists."""
    if checkpoint_path:
        return checkpoint_path

    if torch.jit.is_scripting():
        return checkpoint_path
    if os.path.exists(DEFAULT_CHECKPOINT_PATH):
        return DEFAULT_CHECKPOINT_PATH
    return None


def load_detection_model(
    num_classes: int = 4,
    checkpoint_path: Optional[str] = None,
    device: Optional[torch.device] = None,
    verbose: bool = True,
):
    """Load Faster R-CNN and replace its head for BCCD classes."""
    if device is None:
        device = get_device()

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)

    if checkpoint_path:
        model = fasterrcnn_resnet50_fpn(weights=None, weights_backbone=None)
    else:
        model = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)

    # Replace the classifier so the detector can predict background + 3 blood cell classes.
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
        if verbose:
            print(f"Loaded fine-tuned checkpoint: {checkpoint_path}")
    else:
        if verbose:
            print(
            "Warning: no fine-tuned checkpoint was provided. The model uses a pretrained backbone, "
            "but the new blood-cell detection head is not trained yet."
            )

    model.to(device)
    model.eval()
    return model


def load_image(image_input: ImageInput) -> Image.Image:
    """Load an image path or uploaded file into a PIL image."""
    if isinstance(image_input, Image.Image):
        return image_input.convert("RGB")
    return Image.open(image_input).convert("RGB")


def preprocess_image(
    image_input: ImageInput,
    resize_to: Tuple[int, int] = (512, 512),
) -> Tuple[Image.Image, torch.Tensor]:
    """Resize the image for inference and convert it to a tensor."""
    image = load_image(image_input)
    resized_image = image.resize(resize_to)
    image_tensor = F.to_tensor(resized_image)
    return resized_image, image_tensor


def run_inference(
    model,
    image_tensor: torch.Tensor,
    device: Optional[torch.device] = None,
    score_threshold: float = 0.6,
) -> Dict[str, np.ndarray]:
    """Run detection and keep only predictions above the threshold."""
    if device is None:
        device = get_device()

    with torch.no_grad():
        outputs = model([image_tensor.to(device)])[0]

    boxes = outputs["boxes"].detach().cpu().numpy()
    scores = outputs["scores"].detach().cpu().numpy()
    labels = outputs["labels"].detach().cpu().numpy()
    keep = scores >= score_threshold

    return {
        "boxes": boxes[keep],
        "scores": scores[keep],
        "labels": labels[keep],
    }


def draw_boxes(
    image: np.ndarray,
    boxes: Sequence[Sequence[float]],
    labels: Optional[Sequence[int]] = None,
    scores: Optional[Sequence[float]] = None,
    color: Tuple[int, int, int] = (0, 255, 0),
) -> np.ndarray:
    """Draw bounding boxes and optional class labels on an image."""
    output_image = image.copy()

    for index, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(value) for value in box]
        cv2.rectangle(output_image, (x1, y1), (x2, y2), color, 2)

        text_parts = []
        if labels is not None:
            label_id = int(labels[index])
            text_parts.append(ID_TO_CLASS.get(label_id, str(label_id)))
        if scores is not None:
            text_parts.append(f"{float(scores[index]):.2f}")

        if text_parts:
            text = " | ".join(text_parts)
            cv2.putText(
                output_image,
                text,
                (x1, max(20, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
                cv2.LINE_AA,
            )

    return output_image


def show_comparison(original_image: np.ndarray, compared_image: np.ndarray) -> None:
    """Display the plain image and the annotated result side by side."""
    figure, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(original_image)
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    axes[1].imshow(compared_image)
    axes[1].set_title("Ground Truth and Predictions")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


def compute_iou(box_a: Sequence[float], box_b: Sequence[float]) -> float:
    """Compute Intersection over Union between two boxes."""
    x_left = max(box_a[0], box_b[0])
    y_top = max(box_a[1], box_b[1])
    x_right = min(box_a[2], box_b[2])
    y_bottom = min(box_a[3], box_b[3])

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a[2] - box_a[0]) * max(0.0, box_a[3] - box_a[1])
    area_b = max(0.0, box_b[2] - box_b[0]) * max(0.0, box_b[3] - box_b[1])
    union = area_a + area_b - intersection
    return 0.0 if union <= 0 else float(intersection / union)


def evaluate_detections(
    pred_boxes: Sequence[Sequence[float]],
    pred_labels: Sequence[int],
    gt_boxes: Sequence[Sequence[float]],
    gt_labels: Sequence[int],
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute simple precision, recall, and mean IoU for one image."""
    matched_gt = set()
    true_positives = 0
    iou_scores: List[float] = []

    for pred_index, pred_box in enumerate(pred_boxes):
        pred_label = int(pred_labels[pred_index])
        best_iou = 0.0
        best_gt_index = -1

        for gt_index, gt_box in enumerate(gt_boxes):
            if gt_index in matched_gt:
                continue
            if pred_label != int(gt_labels[gt_index]):
                continue

            current_iou = compute_iou(pred_box, gt_box)
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if best_iou >= iou_threshold and best_gt_index >= 0:
            true_positives += 1
            matched_gt.add(best_gt_index)
            iou_scores.append(best_iou)

    false_positives = max(0, len(pred_boxes) - true_positives)
    false_negatives = max(0, len(gt_boxes) - true_positives)

    precision = true_positives / max(1, true_positives + false_positives)
    recall = true_positives / max(1, true_positives + false_negatives)
    mean_iou = float(np.mean(iou_scores)) if iou_scores else 0.0

    return {
        "true_positives": float(true_positives),
        "false_positives": float(false_positives),
        "false_negatives": float(false_negatives),
        "precision": precision,
        "recall": recall,
        "mean_iou": mean_iou,
    }