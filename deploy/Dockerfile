FROM python:3.11-slim

ENV TZ=Europe/Moscow
ARG DEBIAN_FRONTEND=noninteractive

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV PIP_NO_CACHE_DIR=off
ENV PYTHONDONTWRITEBYTECODE=on
ENV PYTHONFAULTHANDLER=on
ENV PYTHONUNBUFFERED=on
ARG POETRY_VERSION=2.1.3
ENV POETRY_REQUESTS_TIMEOUT=900
ARG POETRY_MAX_WORKERS=5
ENV TQDM_MININTERVAL=5

ENV PATH="/root/.local/bin:${PATH}"

RUN apt-get update && \
    apt-get install -y make && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip install --upgrade --no-cache pip && \
    curl -sSL https://install.python-poetry.org | python3 - --version=$POETRY_VERSION
    
WORKDIR /app

ENV PYTHONPATH="/app/:${PYTHONPATH}"

COPY ["./poetry.lock", "./pyproject.toml", "./"]

RUN poetry config virtualenvs.create false &&  \
    poetry config installer.max-workers $POETRY_MAX_WORKERS && \
    poetry install --no-interaction --no-ansi -vvv && \
    rm -rf ~/.cache/pypoetry/{cache,artifacts}

COPY . .

CMD ["/bin/sh"]
