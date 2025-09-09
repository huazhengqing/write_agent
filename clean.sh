#!/bin/bash


rm -f run.log  logs/*.log
rm -rf .litellm_cache/*
rm -rf .cache/*
rm -rf .prefect/storage/*
rm -rf .pytest_cache
rm -rf output/story/*.db
rm -rf output/*/*.db

rm -rf __pycache__
rm -rf agents/__pycache__
rm -rf prompts/__pycache__
rm -rf prompts/story/__pycache__
rm -rf utils/__pycache__


docker-compose down -v
docker-compose down -v
docker-compose down -v


