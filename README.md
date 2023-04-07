# YOLOv8 ML backend for the Label Studio

This project contains an ML backend for classifying pills in Label Studio. It uses the YOLOv8 model and can segment and classify pills as capsules or tablets.

## Project Structure
### The repository contains the following files and directories:

- **Dockerfile**: The Dockerfile for building the backend container.

- **docker-compose.yml**: The docker-compose file for running the backend.

- **model.py**: The Python code for the ML backend model.

- **best.pt**: The pre-trained YOLOv8 model for pill classification.

- **uwsgi.ini**: The uWSGI configuration file for running the backend.

- **supervisord.conf**: The supervisord configuration file for running the backend processes.

- **requirements.txt**: The list of Python dependencies for the backend.


## Getting Started
1. Clone the Label Studio Machine Learning Backend git repository. From the command line, run the following:

    ```git clone https://github.com/seblful/label-studio-yolov8-backend.git```

2. To use this backend, you'll need to have Docker and docker-compose installed. Then, run the following command to start the backend:

    ```docker-compose up```

This will start the backend on localhost:9090.

Check if it works:

    ```$ curl http://localhost:9090/health```
    ```{"status":"UP"}```

3. Connect running backend to Label Studio:

    ```label-studio start --init new_project --ml-backends http://localhost:9090```

4. Start the labeling process.

## Training
Model training is **not included** in this project. This will probably be added later.

## Contributing
Contributions to this project are welcome. To contribute, please submit an issue or pull request.
