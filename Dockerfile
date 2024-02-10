FROM ubuntu:22.04

COPY . /home/evrun/evorobot-integrated/
WORKDIR /home/evrun/evorobot-integrated/
ENV DEBIAN_FRONTEND noninteractive

RUN apt-get update
RUN apt-get install -y \
    git-all \
    python3.10 \
    python3-pip \ 
    python2-dev \
    python3-dev \
    python-is-python3 \
    libgsl-dev \
    g++

RUN apt update
RUN apt install -y libopenmpi-dev

RUN pip install -r requirements.txt

RUN make compile_all
RUN mkdir data logs

ENTRYPOINT ["bash"]