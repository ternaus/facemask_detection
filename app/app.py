from typing import List, Dict, Any

import albumentations as A
import cv2
import numpy as np
import streamlit as st
import torch
from PIL import Image
from retinaface.pre_trained_models import get_model as get_detector

from facemask_detection.pre_trained_models import get_model as get_classifier

st.set_option("deprecation.showfileUploaderEncoding", False)


def visualize_annotations(image: np.ndarray, annotations: List[Dict[str, Any]]) -> np.ndarray:
    vis_image = image.copy()

    for prediction_id, annotation in enumerate(annotations):
        is_mask = predictions[prediction_id] > 0.5
        if is_mask:
            color = (255, 0, 0)
            text = "mask"
        else:
            color = (0, 255, 0)
            text = "no mask"

        x_min, y_min, x_max, y_max = annotation["bbox"]

        x_min = np.clip(x_min, 0, x_max - 1)
        y_min = np.clip(y_min, 0, y_max - 1)

        vis_image = cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), color=color, thickness=2)

        vis_image = cv2.putText(
            vis_image, text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2, cv2.LINE_AA
        )
    return vis_image


@st.cache
def face_detector_model():
    m = get_detector("resnet50_2020-07-20", max_size=1048, device="cpu")
    m.eval()
    return m


face_detector = face_detector_model()
face_detector.eval()

mask_classifier = get_classifier("tf_efficientnet_b0_ns_2020-07-29")
mask_classifier.eval()

transform = A.Compose(
    [
        A.SmallestMaxSize(max_size=256, p=1, interpolation=cv2.INTER_CUBIC),
        A.CenterCrop(height=224, width=224, p=1),
        A.Normalize(p=1),
    ]
)

st.title("Detect face masks")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

if uploaded_file is not None:
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Before", use_column_width=True)
    st.write("")
    st.write("Detecting faces...")
    with torch.no_grad():
        annotations = face_detector.predict_jsons(image)

    if not annotations[0]["bbox"]:
        st.write("No faces detected")
    else:
        with torch.no_grad():
            predictions: List[float] = []
            for annotation in annotations:
                x_min, y_min, x_max, y_max = annotation["bbox"]

                x_min = np.clip(x_min, 0, x_max)
                y_min = np.clip(y_min, 0, y_max)

                crop = image[y_min:y_max, x_min:x_max]

                crop_transformed = transform(image=crop)["image"]
                model_input = torch.from_numpy(np.transpose(crop_transformed, (2, 0, 1)))

                predictions += [mask_classifier(model_input.unsqueeze(0))[0].item()]

                vis_image = image.copy()

        vis_image = visualize_annotations(image, annotations)

        st.image(vis_image, caption="After", use_column_width=True)
