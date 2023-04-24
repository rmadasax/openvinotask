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
#RUN omz_downloader --name yolox-tiny && omz_converter --name yolox-tiny
#RUN omz_downloader --name road-segmentation-adas-0001

WORKDIR  /workspace/
#RUN git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
#RUN pip3 install -r ./open_model_zoo/demos/common/python/requirements.txt
RUN pip3 install torch==1.13.0+cu116 torchvision==0.14.0+cu116 -f https://download.pytorch.org/whl/cu116/torch_stable.html

WORKDIR  /workspace/
COPY config.json vision_models_inference_speeds_mod.py  /workspace/
WORKDIR  /workspace
CMD [ "python3", "vision_models_inference_speeds_mod.py"]

