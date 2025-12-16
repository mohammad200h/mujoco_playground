
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu22.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

RUN apt-get update && apt-get install -y software-properties-common
RUN add-apt-repository universe

RUN apt-get update && apt-get install -y git
RUN apt-get update && apt install -y mesa-utils
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
RUN cd /workspace/mujoco_playground && pip install -e ".[all,gpu]"

###### Shell #######
CMD /bin/bash
