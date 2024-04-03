FROM nvidia/cuda:11.0.3-base-ubuntu20.04

WORKDIR /app

COPY requirements.txt ./

RUN apt-get update && apt-get install -y \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install -r requirements.txt

COPY . .
