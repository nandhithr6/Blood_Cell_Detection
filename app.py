import numpy as np
import streamlit as st

from dataset import ID_TO_CLASS
from utils import (
    draw_boxes,
    ensure_checkpoint_available,
    get_device,
    load_detection_model,
    preprocess_image,
    resolve_checkpoint_path,
    run_inference,
)


st.set_page_config(page_title="Blood Cell Detection", layout="wide")


@st.cache_resource
def get_model(checkpoint_path: str, checkpoint_url: str):
    device = get_device()
    resolved_checkpoint_path = ensure_checkpoint_available(
        checkpoint_path=checkpoint_path or None,
        checkpoint_url=checkpoint_url or None,
    )
    model = load_detection_model(checkpoint_path=resolved_checkpoint_path or None, device=device)
    return model, device


def get_secret_checkpoint_url() -> str:
    """Read an optional checkpoint URL from Streamlit secrets."""
    return str(st.secrets.get("checkpoint_url", ""))


def main():
    default_checkpoint_path = resolve_checkpoint_path()
    checkpoint_url = get_secret_checkpoint_url()

    st.title("Blood Cell Detection with Faster R-CNN")
    st.write(
        "Upload a microscopy image and visualize predicted blood cell bounding boxes for RBC, WBC, and Platelets."
    )
    st.info(
        "For meaningful class predictions, load fine-tuned BCCD weights. Without a checkpoint, the project still "
        "demonstrates the full inference pipeline but the replaced classifier head is not trained."
    )
    st.caption("If checkpoints/bccd_fasterrcnn.pth exists, it is used automatically.")
    st.caption("For cloud deployment, you can also provide a checkpoint download URL through Streamlit secrets.")

    checkpoint_path = st.text_input("Optional checkpoint path", value=default_checkpoint_path or "")
    threshold = st.slider("Confidence threshold", min_value=0.1, max_value=0.95, value=0.6, step=0.05)
    uploaded_file = st.file_uploader("Upload a blood cell image", type=["png", "jpg", "jpeg"])

    if checkpoint_url:
        st.success("A hosted checkpoint URL is configured for this app.")
    elif default_checkpoint_path:
        st.success(f"Using local checkpoint: {default_checkpoint_path}")
    else:
        st.warning("No fine-tuned checkpoint is configured. Predictions will not be reliable.")

    if uploaded_file is None:
        st.stop()

    try:
        model, device = get_model(checkpoint_path, checkpoint_url)
    except Exception as error:
        st.error(f"Failed to load the detection model: {error}")
        st.stop()

    # Resize the uploaded image and convert it to a tensor for Faster R-CNN.
    resized_image, image_tensor = preprocess_image(uploaded_file, resize_to=(512, 512))

    # Run detection and keep only confident predictions.
    predictions = run_inference(model, image_tensor, device=device, score_threshold=threshold)

    image_np = np.array(resized_image)
    detected_image = draw_boxes(
        image_np,
        predictions["boxes"],
        labels=predictions["labels"],
        scores=predictions["scores"],
        color=(255, 0, 0),
    )

    left_column, right_column = st.columns(2)

    with left_column:
        st.subheader("Uploaded Image")
        st.image(image_np, width="stretch")

    with right_column:
        st.subheader("Detected Boxes")
        st.image(detected_image, width="stretch")

    st.subheader("Prediction Summary")
    st.write(f"Number of detections: {len(predictions['boxes'])}")

    if len(predictions["boxes"]) == 0:
        st.warning("No detections passed the confidence threshold.")
    else:
        for index, score in enumerate(predictions["scores"], start=1):
            label_name = ID_TO_CLASS.get(int(predictions["labels"][index - 1]), "Unknown")
            st.write(f"{index}. {label_name} | confidence = {float(score):.4f}")


if __name__ == "__main__":
    main()