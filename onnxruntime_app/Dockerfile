#FROM openvino/ubuntu20_dev:latest
#ARG DEBIAN_FRONTEND=noninteractive
FROM openvino/onnxruntime_ep_ubuntu20
ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt update -y
RUN apt install -y python3 python3-pip  git python3-opencv  python3-tk
RUN mkdir /workspace
RUN mkdir /workspace/models
RUN pip3 install psutil py-cpuinfo
RUN pip3 install pandas==1.5.3
WORKDIR  /workspace/models
ADD https://download.pytorch.org/models/vit_b_16-c867db91.pth /root/.cache/torch/hub/checkpoints/
ADD https://download.pytorch.org/models/vgg11-8a719046.pth /root/.cache/torch/hub/checkpoints/
ADD https://download.pytorch.org/models/resnet50-11ad3fa6.pth /root/.cache/torch/hub/checkpoints/

WORKDIR  /workspace/
#RUN git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
#RUN pip3 install -r ./open_model_zoo/demos/common/python/requirements.txt
RUN pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

WORKDIR  /workspace/
COPY config.json auto_benchmark_app.py vision_models_inference_speeds_mod.py  /workspace/
WORKDIR  /workspace

