#!/bin/bash

docker pull searxng/searxng:latest
docker pull prefecthq/prefect:3-latest
docker pull ghcr.io/berriai/litellm:main-stable
# docker pull valkey/valkey:8-alpine
# docker pull memgraph/memgraph-mage:latest
# docker pull memgraph/lab:latest
# docker pull qdrant/qdrant:latest