FROM python:3.10

ENV HOST=0.0.0.0

ENV LISTEN_PORT 8080

EXPOSE 8080

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY ./requirements.txt /app/requirements.txt


RUN python -m venv venv

#RUN source venv/bin/activate
RUN ["/bin/bash", "-c", "source venv/bin/activate"]

RUN pip install --upgrade pip

RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

WORKDIR /app

HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

COPY . /app

CMD [ "chainlit", "run" , "app_llamaindex.py", "--host", "0.0.0.0", "--port", "8080" ]