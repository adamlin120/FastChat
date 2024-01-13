FROM nvidia/cuda:12.2.0-runtime-ubuntu20.04

RUN apt-get update -y && apt-get install -y python3.9 python3.9-distutils curl
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3.9 get-pip.py

RUN apt-get update -y && apt-get install -y git
# Clone the FastChat repository from GitHub
RUN git clone https://github.com/adamlin120/FastChat.git

# Navigate into the cloned directory and install fschat
RUN cd FastChat && \
    pip3 install .[model_worker,webui,arena] pydantic==1.10.13
