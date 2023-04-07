import os
import random
import requests
from PIL import Image
from io import BytesIO

from ultralytics import YOLO
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_single_tag_keys, get_local_path

# URL with host
# LS_URL =  "http://127.0.0.1:8080"
LS_URL = "http://192.168.100.3:8080"
LS_API_TOKEN = "your api token"


# Initialize class inhereted from LabelStudioMLBase
class YOLOv8Model(LabelStudioMLBase):
    def __init__(self, **kwargs):
        # Call base class constructor
        super(YOLOv8Model, self).__init__(**kwargs)

        # Initialize self variables
        self.from_name, self.to_name, self.value, self.classes = get_single_tag_keys(
            self.parsed_label_config, 'PolygonLabels', 'Image')
        self.labels = ['capsules', 'tablets']
        # Load model
        self.model = YOLO("best.pt")

    # Function to predict
    def predict(self, tasks, **kwargs):
        """
        Returns the list of predictions based on input list of tasks for 1 image
        """
        task = tasks[0]

        # Getting URL of the image
        image_url = task['data'][self.value]
        full_url = LS_URL + image_url
        print("FULL URL: ", full_url)

        # Header to get request
        header = {
            "Authorization": "Token " + LS_API_TOKEN}
        
        # Getting URL and loading image
        image = Image.open(BytesIO(requests.get(
            full_url, headers=header).content))
        # Height and width of image
        original_width, original_height = image.size
        
        # Creating list for predictions and variable for scores
        predictions = []
        score = 0
        

        # Getting prediction using model
        results = self.model.predict(image)
        

        # Getting mask segments, boxes from model prediction
        for result in results:
            for i, (box, segm) in enumerate(zip(result.boxes, result.masks.segments)):
                
                # 2D array with poligon points 
                polygon_points = (segm * 100).tolist()

                # Adding dict to prediction
                predictions.append({
                    "from_name" : self.from_name,
                    "to_name" : self.to_name,
                    "id": str(i),
                    "type": "polygonlabels",
                    "score": box.conf.item(),
                    "original_width": original_width,
                    "original_height": original_height,
                    "image_rotation": 0,
                    "value": {
                        "points": polygon_points,
                        "polygonlabels": [self.labels[int(box.cls.item())]]
                    }})

                # Calculating score
                score += box.conf.item()


        print(10*"#", "Returned Prediction", 10*"#")

        # Dict with final dicts with predictions
        final_prediction = [{
            "result": predictions,
            "score": score / (i + 1),
            "model_version": "v8s"
        }]

        return final_prediction
    
    def fit(self, completions, workdir=None, **kwargs):
        """ 
        Dummy function to train model
        """
        return {'random': random.randint(1, 10)}