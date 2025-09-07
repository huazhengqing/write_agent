#!/bin/bash


rm -f run.log  logs/*.log
rm -rf .litellm_cache/*
rm -rf .cache/*


docker-compose stop
docker-compose down -v


