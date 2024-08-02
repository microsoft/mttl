FROM nvcr.io/nvidia/pytorch:24.07-py3

RUN apt update && apt install -y python3.10-venv git
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

RUN python -m pip install --upgrade pip
COPY requirements.txt .
RUN pip install -r requirements.txt
