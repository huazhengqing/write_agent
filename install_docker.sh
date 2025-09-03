#!/bin/bash


sudo mkdir -p /etc/docker
sudo tee /etc/docker/daemon.json <<-'EOF'
{
  "registry-mirrors": [
    "https://3xoj0j3i.mirror.aliyuncs.com",
    "https://docker.m.daocloud.io",
    "https://mirror.azure.cn",
    "https://ghcr.hub1.nat.tf",
    "https://f1361db2.m.daocloud.io"
  ]
}
EOF
systemctl daemon-reload
systemctl restart docker
docker info | grep "Registry Mirrors" -A 5





docker pull memgraph/memgraph-mage:latest
docker pull memgraph/lab:latest
docker pull qdrant/qdrant:latest
docker pull valkey/valkey:8-alpine
docker pull searxng/searxng:latest




docker volume create memgraph-data
docker volume create qdrant_storage
docker volume create valkey-data2
docker volume create searxng-data






docker-compose up -d




docker run -d -p 7687:7687 -p 7444:7444 -e MGCONSOLE="--username memgraph --password memgraph" -v memgraph-data:/var/lib/memgraph --name memgraph memgraph/memgraph-mage:latest --schema-info-enabled=True


docker run -d -p 3000:3000 --name memgraph_lab memgraph/lab:latest



docker stop -t 60   memgraph
docker restart memgraph


create user memgraph identified by 'memgraph'







