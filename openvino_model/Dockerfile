FROM openvino/ubuntu22_dev:latest
ARG DEBIAN_FRONTEND=noninteractive

USER root

RUN apt update -y
#RUN apt install -y intel-opencl-icd intel-level-zero-gpu
RUN apt install -y python3 python3-pip  git python3-opencv  python3-tk 

RUN apt -y install wget
RUN apt -y install ffmpeg

RUN mkdir /workspace
RUN mkdir /workspace/models
RUN mkdir /workspace/models/yolov7-pose-qat
RUN mkdir /workspace/models/yolov7-pose-qat/yolov7-fp16
WORKDIR  /workspace/models/yolov7-pose-qat/yolov7-fp16
RUN ["wget", "https://huggingface.co/LouisMM/yolov7-pose-qat/resolve/main/yolov7-fp16/model.xml"]
RUN ["wget", "https://huggingface.co/LouisMM/yolov7-pose-qat/resolve/main/yolov7-fp16/model.bin"]
RUN pip3 install psutil py-cpuinfo
RUN ffmpeg -h
#RUN omz_downloader --name yolox-tiny && omz_converter --name yolox-tiny
#RUN omz_downloader --name road-segmentation-adas-0001

#WORKDIR  /workspace/
#RUN git clone --recurse-submodules https://github.com/openvinotoolkit/open_model_zoo.git
#RUN pip3 install -r ./open_model_zoo/demos/common/python/requirements.txt

WORKDIR  /workspace/
COPY auto_benchmark_app.py hello_query_device.py /workspace
WORKDIR  /workspace
RUN ls
#ENTRYPOINT [ "python3", "auto_benchmark_app.py", "-m models/intel/road-segmentation-adas-0001/FP32/road-segmentation-adas-0001.xml", "-d CPU"]
ENTRYPOINT [ "python3", "auto_benchmark_app.py", "-m models/yolov7-pose-qat/yolov7-fp16/model.xml", "-d GPU", "-hint latency"]
#CMD [ "python3", "auto_benchmark_app.py -m m models/yolov7-pose-qat/yolov7-fp16/model.xml -d GPU -hint latency"]
#CMD [ "python3", "hello_query_device.py"]
