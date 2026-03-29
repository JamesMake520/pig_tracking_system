import os
import supervision as sv
from rfdetr.detr import RFDETRBase
from PIL import Image
from io import BytesIO
import requests

url = "https://media.roboflow.com/dog.jpeg"
image = Image.open(BytesIO(requests.get(url).content))

model = RFDETRBase()

detections = model.predict(image, threshold=0.5)

# Get class names from the model
class_names_dict = model.class_names
labels = [class_names_dict.get(class_id, f"class_{class_id}") for class_id in detections.class_id]

annotated_image = image.copy()
annotated_image = sv.BoxAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections)
annotated_image = sv.LabelAnnotator(color=sv.ColorPalette.ROBOFLOW).annotate(annotated_image, detections, labels)