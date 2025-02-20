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
    mkdir model
    cp /path/to/your/model.pt model/
    mv model/your_model.pt model/best.pt
    ```

3. **Edit `.env` with your settings:**
   
    ```yaml
    - LABEL_STUDIO_URL=http://host.docker.internal:${PORT}
    - LABEL_STUDIO_API_KEY=<your_api_key>
    - TASK_TYPE=<detection> or <segmentation>
    ```

    > **Important Notes:**
    > - Default LABEL_STUDIO_URL is `http://host.docker.internal:8080`
    > - Get IP using 'ifconfig' (Linux/Mac) or 'ipconfig' (Windows)
    > - ⚠️ Never use `localhost` as the container is isolated from the host
    > - Get your API_KEY from: Label Studio -> Account Settings -> Access Token


4. **Deploy using the following commands:**
   
    ```bash
    docker pull pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
    docker compose build
    docker compose up
    ```

5. **Add model in project settings:**

    From the project settings, select the **Model** page and click [**Connect Model**](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).
    
    Add the URL `http://locallhost:{PORT}` and save the model as an ML backend.

   ![model_connected](https://github.com/user-attachments/assets/2f240905-f093-42c1-bad8-7b90efc4fcab)


6. **Label in interactive mode**

    To use this functionality, activate **Auto-Annotation**.


  ### 🇨🇳 For Users in China
  
  To ensure smooth deployment in regions with internet restrictions:
  
  1. Configure Docker daemon with proxy:
  ```json
  {
    "registry-mirrors": ["https://registry.docker-cn.com"]
  }
  ```

### 📋 TODO

- add support for keypoints and obb.


## Training

Model training is **not included** in this project. This will probably be added later.

## Contributing

Contributions to this project are welcome. To contribute, please submit an issue or pull request.
