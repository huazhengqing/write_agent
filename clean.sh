#!/bin/bash

rm -rf .litellm_cache/*
rm -rf .cache/*
rm -rf .prefect/storage/*
rm -rf .chroma_db/*
rm -rf .kuzu_db/*
rm -rf .sqlite/*
rm -rf .output/story/*.db
rm -rf .output/*/*.db
rm -rf .logs/*.log
rm -rf .pytest_cache

rm -rf __pycache__
rm -rf agents/__pycache__
rm -rf prompts/__pycache__
rm -rf prompts/story/__pycache__
rm -rf utils/__pycache__

rm -f write.log


# docker-compose down -v


