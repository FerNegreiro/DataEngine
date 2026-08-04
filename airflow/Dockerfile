ARG AIRFLOW_VERSION=3.3.0
ARG PYTHON_VERSION=3.12

FROM apache/airflow:${AIRFLOW_VERSION}-python${PYTHON_VERSION}

ARG AIRFLOW_VERSION

COPY --chown=airflow:root requirements.txt /tmp/requirements-project.txt
COPY --chown=airflow:root airflow/requirements-airflow.txt /tmp/requirements-airflow.txt

RUN pip install --no-cache-dir \
    "apache-airflow==${AIRFLOW_VERSION}" \
    -r /tmp/requirements-airflow.txt

