FROM ubuntu:focal

WORKDIR /usr/src/app
# install general utils
RUN apt update && apt install -y git iputils-ping && apt autoclean

# install python utils
RUN apt update && apt install -y python3 python3-pip && apt autoclean 

RUN pip3 install git+https://github.com/EmbodiedCognition/py-c3d
RUN pip3 install matplotlib

COPY . .