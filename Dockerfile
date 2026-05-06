FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-devel

WORKDIR /workspace/PSKV

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

COPY . .
