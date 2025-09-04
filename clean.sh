#!/bin/bash


rm -f run.log  logs/*.log
rm -rf .litellm_cache/*
rm -rf .cache/*
rm -rf .mem0


docker-compose stop
docker-compose down -v







