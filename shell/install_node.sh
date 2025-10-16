#!/bin/bash


sudo apt update
sudo apt install -y curl
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt install -y nodejs


node -v
npm -v


npm config set registry https://registry.npmmirror.com
npm config get registry


# npm create vite@latest ui -- --template vue-ts


cd ui
npm install
npm install element-plus pinia axios
npm install vue-router


