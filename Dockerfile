FROM nvcr.io/nvidia/pytorch:20.12-py3

ADD . /workspace/
WORKDIR /workspace/

RUN conda env create -f environment.yml 
RUN conda init bash
