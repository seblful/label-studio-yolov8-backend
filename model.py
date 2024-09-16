from typing import List, Dict, Optional

import os

from PIL import Image, ImageOps

import numpy as np
import ultralytics

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
from label_studio_ml.utils import get_single_tag_keys
from label_studio_sdk._extensions.label_studio_tools.core.utils.io import get_local_path


class YOLO(LabelStudioMLBase):
    """Label Studio ML Backend based on Ultralytics YOLO"""

    def __init__(self,
                 **kwargs) -> None:
        super(YOLO, self).__init__(**kwargs)

        # Task type
        self.task_types = ["detection", "segmentation"]
        self.task_type = os.getenv("TASK_TYPE")
        print(f"Task type is {self.task_type}.")

        # Model and labels
        self.model = ultralytics.YOLO("models/best_seg.pt")
        self.labels = self.model.names

    def setup(self) -> None:
        """Configure any parameters of your model here"""
        self.set("model_version", "yolov8m-seg")

    def load_image(self,
                   task: Dict) -> Image.Image:
        # Get image path and task id
        image_path = task.get("data").get("image")
        task_id = task.get("id")

        # Extract local image path
        file_path = self.get_local_path(image_path,
                                        task_id=task_id)

        # Open image
        image = Image.open(file_path)
        image = ImageOps.exif_transpose(image)

        return image

    def predict(self,
                tasks: List[Dict],
                context: Optional[Dict] = None,
                **kwargs) -> ModelResponse:
        assert self.task_type in self.task_types, \
            f"Task type must be one \
                of {self.task_types}, set TASK_TYPE in your .env file."

        if self.task_type == "detection":
            predictions = self.predict_det(tasks)
        else:
            predictions = self.predict_seg(tasks)
        print('.' * 20, "Returned prediction", '.' * 20)
        return predictions

    def predict_det(self,
                    tasks: List[Dict]) -> ModelResponse:
        pass

    def predict_seg(self,
                    tasks: List[Dict]) -> ModelResponse:
        # Create blank list with results
        results = []

        # Create variable to calcualte scores
        score = 0
        counter = 0

        for task in tasks:
            # Load image
            image = self.load_image(task=task)

            # Height and width of image
            image_width, image_height = image.size

            # Getting prediction using model
            model_prediction = self.model.predict(image)

            # Getting mask segments, boxes from model prediction
            for pred in model_prediction:
                for i, (box, segm) in enumerate(zip(pred.boxes, pred.masks.xy)):

                    # 2D array with poligon points
                    points = segm / \
                        np.array([image_width, image_height]) * 100
                    points = points.tolist()

                    # Label
                    labels = [self.labels[int(box.cls.item())]]

                    # Regions and predictions
                    result = {"from_name": "label",
                              "to_name": "image",
                              "id": str(i),
                              "type": "polygonlabels",
                              "score": box.conf.item(),
                              "original_width": image_width,
                              "original_height": image_height,
                              "image_rotation": 0,
                              "value": {"points": points,
                                        "polygonlabels": labels}}

                    # Append prediction to predictions
                    results.append(result)

                    # Add score
                    score += box.conf.item()
                    counter += 1

        predictions = [{"result": results,
                       "score": score / counter,
                        "model_version": self.model_version}]

        return ModelResponse(predictions=predictions)

    def fit(self, event, data, **kwargs):
        raise NotImplementedError("Training is not implemented yet")
