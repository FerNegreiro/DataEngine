FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN groupadd --gid 1000 dataengine \
    && useradd --uid 1000 --gid dataengine --no-create-home dataengine

COPY requirements.txt pyproject.toml ./

RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=dataengine:dataengine . .

RUN mkdir -p /app/data/raw /app/data/processed \
    && chown -R dataengine:dataengine /app/data

USER dataengine

CMD ["python", "-m", "pipelines.run_pipeline"]
