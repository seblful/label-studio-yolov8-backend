FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_CACHE_DIR=/.cache \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    MKL_THREADING_LAYER=GNU \
    PORT=9090

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

# Install linux packages
RUN apt update \
    && apt install --no-install-recommends -y gcc git zip unzip wget curl htop libgl1 libglib2.0-0 libpython3-dev gnupg g++ libusb-1.0-0 libsm6

RUN apt upgrade --no-install-recommends -y openssl tar

# Create working directory
WORKDIR /app

# Install pip packages
RUN python3 -m pip install --upgrade pip wheel
COPY requirements.txt .
RUN --mount=type=cache,target=${PIP_CACHE_DIR},sharing=locked \
    pip3 install -r requirements.txt

WORKDIR /app

COPY . ./

CMD ["/app/start.sh"]