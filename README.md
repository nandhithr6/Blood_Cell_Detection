# Blood Cell Detection with Faster R-CNN

## Project Title

Blood Cell Detection using PyTorch, Faster R-CNN, and the BCCD Dataset.

## Dataset Description

This project uses the BCCD (Blood Cell Count and Detection) dataset.

- Images are stored in `BCCD_Dataset/JPEGImages`
- Pascal VOC XML annotations are stored in `BCCD_Dataset/Annotations`
- The dataset contains three object classes:
	- RBC
	- WBC
	- Platelets

Each XML file stores bounding box coordinates in the form:

```text
[xmin, ymin, xmax, ymax]
```

This makes the dataset suitable for object detection experiments and assignment demonstrations.

## Detection Methodology

The project follows a simple object detection pipeline:

1. Load microscopy images from `JPEGImages`
2. Parse Pascal VOC XML files from `Annotations`
3. Extract bounding boxes and class labels
4. Convert images into PyTorch tensors
5. Resize images to a fixed size for faster inference
6. Load a Faster R-CNN model from torchvision
7. Replace the classifier head so the model supports background, RBC, WBC, and Platelets
8. Run inference on images
9. Draw ground-truth boxes in green and predicted boxes in red
10. Compute IoU, precision, and recall

## Model Used

- Model: `fasterrcnn_resnet50_fpn`
- Framework: PyTorch and torchvision
- Detection head: replaced to support 3 BCCD classes plus background

Important note:

- The project supports loading a fine-tuned checkpoint if you have one
- If no checkpoint is supplied, the Faster R-CNN backbone is pretrained but the replaced blood-cell classifier head is not fine-tuned yet
- The full detection pipeline still runs, but meaningful BCCD predictions require fine-tuned weights

## Evaluation Metrics

The project computes the following basic object detection metrics:

- IoU (Intersection over Union): how much the predicted box overlaps the true box
- Precision: the fraction of predicted boxes that are correct
- Recall: the fraction of real objects that were successfully detected

These metrics are printed in `detect.py` for the selected image.

## Role of AI in Medical Diagnosis

AI can help medical experts by speeding up image analysis, locating suspicious regions, and supporting routine screening tasks. In microscopy applications, object detection can help identify and count cells more efficiently.

However, AI should be treated as a support tool. Medical systems require domain-specific training, evaluation, expert validation, and careful deployment before they can be used in real clinical environments.

## Project Structure

```text
blood-cell-detection/
|
|-- BCCD_Dataset/
|-- dataset.py
|-- detect.py
|-- utils.py
|-- app.py
|-- requirements.txt
|-- README.md
```

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Make sure the dataset is placed like this:

```text
BCCD_Dataset/
|-- BCCD/
|   |-- JPEGImages/
|   |-- Annotations/
|   |-- ImageSets/
```

The code also accepts a direct `BCCD_Dataset/JPEGImages` style layout if you organize the dataset manually.

## Run the Detection Script

```bash
python detect.py
```

Optional arguments:

```bash
python detect.py --dataset_dir BCCD_Dataset --threshold 0.6 --checkpoint bccd_fasterrcnn.pth
python detect.py --image_name BloodImage_00000.jpg
```

## Run the Streamlit App

```bash
python -m streamlit run app.py
```

## Push to GitHub Quickly

This project is prepared to keep the repository small enough for GitHub:

- The local virtual environment is ignored
- The raw BCCD dataset is ignored
- The trained checkpoint in `checkpoints/bccd_fasterrcnn.pth` is kept so the deployed app can run predictions

Initialize and push:

```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin <your-github-repo-url>
git push -u origin main
```

## Deploy to Render

This repository includes `render.yaml`, so Render can create the service automatically.

Use these steps:

1. Push this project to GitHub
2. In Render, choose New + > Blueprint
3. Select your GitHub repository
4. Render will detect `render.yaml`
5. Deploy the web service

The service starts with:

```bash
python -m streamlit run app.py --server.address 0.0.0.0 --server.port $PORT
```

Notes:

- First build can take a few minutes because PyTorch is large
- The app does not need the training dataset on Render
- The bundled checkpoint avoids downloading model weights at runtime

## Conclusion

This project gives a clean and modular example of medical object detection for a university assignment. It demonstrates data loading, Pascal VOC annotation parsing, Faster R-CNN inference, box visualization, and basic metric calculation in both a Python script and a Streamlit interface.

For the best results, use a fine-tuned BCCD checkpoint with the replaced detection head.
