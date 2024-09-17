# YOLOv8 ML backend for the Label Studio


YOLOv8 interactive ML-assisted labeling, facilitating faster 
annotation for **image detection**, ***instance* image segmentation**.

Tested against Label Studio 1.13.1.

## Project Structure

- **Dockerfile**: The Dockerfile for building the backend container.

- **docker-compose.yml**: The docker-compose file for running the backend.

- **_wsgi.py**: WSGI app initializer.

- **start.sh**: bash script to start the whole process.

- **model.py**: The Python code for the ML backend model.

- **requirements.txt**: The list of Python dependencies for the backend.

## Setup process

Before you begin:
* Ensure git is installed
* Ensure Docker Compose is installed.


### 1. Install Label Studio

Launch Label Studio. You can follow the guide from the [official documentation](https://labelstud.io/guide/install.html) or use the following commands:


If you're using local file serving, be sure to [get a copy of the API token](https://labelstud.io/guide/user_account#Access-token) from
Label Studio to connect the model.

### 2. Create a Label Studio project

Create a new project.

In the project **Settings** set up the **Labeling Interface** for **image detection** (RectangleLabels) or **image segmentation** (PolygonLabels). 

### 3. Install label-studio-yolov8-backend

Download the Label Studio YOLOv8 backend repository.
   ```
   git clone https://github.com/seblful/label-studio-yolov8-backend.git
   cd label-studio-yolov8-backend
   ```

Configure parameters in `.env` file:

   ```
   LABEL_STUDIO_URL=<IPv4 Address> (check your ipconfig)
   LABEL_STUDIO_API_KEY=<Label Studio API token>
   TASK_TYPE=<segmentation> or <detection>
   ```

### 4. Start the servers

   ```
   docker compose up
   ```

### 5. Upload tasks

   Upload images directly to Label Studio using the Label Studio interface.


### 6. Add model in project settings

From the project settings, select the **Model** page and click [**Connect Model**](https://labelstud.io/guide/ml#Connect-the-model-to-Label-Studio).

   Add the URL `http://locallhost:9090` and save the model as an ML backend.

### 7. Label in interactive mode

To use this functionality, activate **Auto-Annotation** for drawing boxes.


## Training

Model training is **not included** in this project. This will probably be added later.

## Contributing

Contributions to this project are welcome. To contribute, please submit an issue or pull request.
