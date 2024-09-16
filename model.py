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
        print('.' * 20, "Returned predictions", '.' * 20)
        return predictions

    def fit(self, event, data, **kwargs):
        raise NotImplementedError("Training is not implemented yet")
