FROM python:3.11-slim

ARG CAREAMICS_VERSION=0.3.5

RUN pip install --no-cache-dir torch==2.7.1 torchvision==0.22.1 \
        --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir careamics==${CAREAMICS_VERSION}

COPY predict.py bilayers_predict.py /usr/local/bin/
RUN chmod +x /usr/local/bin/predict.py /usr/local/bin/bilayers_predict.py

WORKDIR /bilayers
