# Linux + Python 3.10 + OpenFHE, for exp1 and exp4 on a Windows laptop.
# The openfhe wheel on PyPI only carries a Python 3.10 binary.
#   docker build -t thesis-openfhe 2_code/experiments/openfhe_docker
#   docker run --rm -v "<path to 2_code/experiments/code>:/code" -w /code thesis-openfhe python exp4_time_gain.py time
FROM python:3.10-slim
RUN apt-get update -qq && apt-get install -y -qq libgomp1 >/dev/null && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir -q openfhe==1.6.1.0.22.4 numpy pandas
