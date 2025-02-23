# Custom YOLO Backend for Label Studio

This backend provides ML-assisted labeling capabilities to accelerate your annotation workflow, supporting both **object detection** and **instance segmentation** tasks.

## 🏗️ Project Structure

- **Dockerfile**: The Dockerfile for building the backend container.

- **docker-compose.yml**: The docker-compose file for running the backend.

- **_wsgi.py**: WSGI app initializer.

- **start.sh**: bash script to start the whole process.

- **model.py**: The Python code for the ML backend model.

- **requirements.txt**: The list of Python dependencies for the backend.

## 🚀 Quick Start

1. **Clone the repository**:

   ```bash
   git clone https://github.com/seblful/label-studio-yolo-backend.git
   cd label-studio-yolo-backend
   ```

2. **Create and prepare your model directory:**
   
    ```bash
    mkdir models
    cp /path/to/your/model.pt models/
    ```

3. **Edit `.env` with your settings:**
   
    ```yaml
    BASIC_AUTH_USER=  # Optional
    BASIC_AUTH_PASS=  # Optional
    LOG_LEVEL=DEBUG

    MODEL_FILENAME=model.pt

    PORT=8080

    LABEL_STUDIO_API_KEY= # API key from LS
    TASK_TYPE=segmentation # segmentation or detection
    ```


4. **Deploy using the following command:**
   
    ```bash
    docker compose up
    ```

5. **Add the model in project settings:**

    From the project settings, select the **Model** page and click [**Connect Model**](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).
    
    Add the URL `http://locallhost:9090` and save the model as an ML backend.

   ![Connect Model](https://github.com/seblful/label-studio-yolo-backend/raw/main/assets/images/connect_model.png)
   ![Connected model](https://github.com/seblful/label-studio-yolo-backend/raw/main/assets/images/connected_model.png)


6. **Label in interactive mode**

    To use this functionality, activate **Auto-Annotation**.

  ![Example annotation](https://github.com/seblful/label-studio-yolo-backend/raw/main/assets/images/annotation.png)

  ### For users with internet restrictions:
  
  Configure Docker daemon with proxy:
  ```json
  {
    "registry-mirrors": ["https://registry.docker-cn.com"]
  }
  ```

## 📋 TODO

- Add support for obb and keypoints.


## 💁 Contributing

Contributions to this project are welcome. To contribute, please submit an issue or pull request.
