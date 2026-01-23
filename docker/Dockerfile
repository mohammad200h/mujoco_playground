
# FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt install -y mesa-utils
# Install EGL libraries for headless MuJoCo rendering
RUN apt-get update && apt-get install -y libegl1-mesa libgl1-mesa-glx libglib2.0-0
# Set MuJoCo to use EGL for headless rendering
RUN apt-get update && apt-get install -y ffmpeg 
ENV MUJOCO_GL=egl
##### python3.11 ######
RUN apt-get update && add-apt-repository ppa:deadsnakes/ppa
RUN apt-get install -y python3.11 python3.11-venv python3.11-dev
RUN apt-get install -y python3-pip curl
# Set python3.11 as the default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1
RUN update-alternatives --set python3 /usr/bin/python3.11


#####Installing terminator ######
RUN apt-get update
RUN  apt-get install -y terminator
RUN mkdir /etc/xdg/terminator/

###### WorkDir #######
WORKDIR workspace
###### 
COPY . /workspace/mujoco_playground
# RUN cd /workspace/mujoco_playground && pip3 install -e ".[all,gpu]"
RUN cd /workspace/mujoco_playground && pip3 install -r docker/requirements.txt


###### Shell #######
CMD /bin/bash
