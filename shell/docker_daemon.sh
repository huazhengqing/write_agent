

echo "配置 Docker 镜像加速..."
DOCKER_DAEMON_FILE="/etc/docker/daemon.json"
if [ ! -f "$DOCKER_DAEMON_FILE" ] || ! grep -q "registry-mirrors" "$DOCKER_DAEMON_FILE"; then
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
    sudo systemctl daemon-reload
    sudo systemctl restart docker
    echo "Docker 镜像加速配置完成。"
else
    echo "Docker 镜像加速已配置, 跳过配置。"
fi

docker info | grep "Registry Mirrors" -A 5

