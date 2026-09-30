FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends libopus0 libopus-dev git \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY . /app
RUN pip install --no-cache-dir -e ".[analysis]"

ENTRYPOINT ["starvoice"]
CMD ["doctor"]
