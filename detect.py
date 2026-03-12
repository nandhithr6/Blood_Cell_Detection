import argparse
from pathlib import Path

import cv2
import numpy as np

from dataset import BCCDDataset, ID_TO_CLASS
from utils import draw_boxes, evaluate_detections, get_device, load_detection_model, run_inference, show_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Blood cell detection with Faster R-CNN")
    parser.add_argument("--dataset_dir", type=str, default="BCCD_Dataset", help="Path to the BCCD dataset root")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Optional path to fine-tuned model weights. If omitted, checkpoints/bccd_fasterrcnn.pth is used when present.",
    )
    parser.add_argument("--threshold", type=float, default=0.6, help="Confidence threshold for detections")
    parser.add_argument(
        "--image_name",
        type=str,
        default=None,
        help="Optional image file name such as BloodImage_00000.jpg",
    )
    return parser.parse_args()


def select_sample_index(dataset: BCCDDataset, image_name: str | None) -> int:
    if image_name is None:
        return 0

    for index, (image_path, _) in enumerate(dataset.samples):
        if image_path.name == image_name:
            return index

    raise FileNotFoundError(f"Image '{image_name}' was not found in {dataset.root_dir / 'JPEGImages'}.")


def main():
    args = parse_args()
    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.exists():
        raise FileNotFoundError(
            "BCCD_Dataset was not found in the project root. Place the cloned dataset folder here and run again."
        )

    dataset = BCCDDataset(root_dir=str(dataset_dir), resize_to=(512, 512))
    sample_index = select_sample_index(dataset, args.image_name)
    image_tensor, target = dataset[sample_index]

    device = get_device()
    model = load_detection_model(checkpoint_path=args.checkpoint, device=device)

    # Run inference on one BCCD image.
    predictions = run_inference(model, image_tensor, device=device, score_threshold=args.threshold)

    image_np = (image_tensor.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    gt_boxes = target["boxes"].numpy()
    gt_labels = target["labels"].numpy()

    # Draw ground-truth boxes in green and predictions in red.
    combined_image = draw_boxes(image_np, gt_boxes, labels=gt_labels, color=(0, 255, 0))
    combined_image = draw_boxes(
        combined_image,
        predictions["boxes"],
        labels=predictions["labels"],
        scores=predictions["scores"],
        color=(255, 0, 0),
    )

    metrics = evaluate_detections(
        pred_boxes=predictions["boxes"],
        pred_labels=predictions["labels"],
        gt_boxes=gt_boxes,
        gt_labels=gt_labels,
    )

    output_path = "bccd_detection_result.jpg"
    cv2.imwrite(output_path, cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    print("=" * 60)
    print("BCCD Blood Cell Detection")
    print("=" * 60)
    print(f"Image path: {target['image_path']}")
    print(f"Device used: {device}")
    print(f"Checkpoint used: {args.checkpoint or 'checkpoints/bccd_fasterrcnn.pth if available'}")
    print(f"Confidence threshold: {args.threshold}")
    print(f"Number of ground-truth boxes: {len(gt_boxes)}")
    print(f"Number of predicted boxes: {len(predictions['boxes'])}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"Mean IoU: {metrics['mean_iou']:.4f}")
    print("Saved image: bccd_detection_result.jpg")
    print()
    print("Predicted detections:")

    if len(predictions["boxes"]) == 0:
        print("No predictions passed the threshold.")
    else:
        for index, score in enumerate(predictions["scores"]):
            label_name = ID_TO_CLASS.get(int(predictions["labels"][index]), "Unknown")
            box = [round(float(value), 2) for value in predictions["boxes"][index]]
            print(f"{index + 1}. {label_name} | score = {float(score):.4f} | box = {box}")

    print()
    print("Metric notes:")
    print("1. IoU measures overlap between a predicted box and the true box.")
    print("2. Precision measures how many predicted boxes are correct.")
    print("3. Recall measures how many true boxes were successfully found.")
    print("4. The best results come from running train.py first to create a BCCD checkpoint.")

    show_comparison(image_np, combined_image)


if __name__ == "__main__":
    main()