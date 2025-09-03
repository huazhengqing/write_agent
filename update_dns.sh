#!/bin/bash

# 检查是否为 root 用户（需管理员权限）
if [ "$(id -u)" -ne 0 ]; then
    echo "❌ 请使用 sudo 运行脚本（需要管理员权限）"
    exit 1
fi

# 定义要配置的公共 DNS 列表（多组提高稳定性）
DNS_LIST=(
    "223.5.5.5"   # 阿里云DNS（国内常用，稳定）
    "223.6.6.6"   # 阿里云DNS（备用）
    "114.114.114.114"  # 114DNS（国内老牌，兼容性强）
    "114.114.115.115"  # 114DNS（备用，防污染）
    "119.29.29.29"  # DNSPod DNS+（腾讯旗下，国内速度快）
    "182.254.116.116"  # DNSPod DNS+（备用）
    "8.8.8.8"     # 谷歌DNS（国际通用，备选）
    "1.1.1.1"     # Cloudflare DNS（隐私优先，国际稳定）
)

# 步骤1：配置 wsl.conf 禁用自动生成 resolv.conf
echo -e "\n1️⃣  配置 wsl.conf 禁用自动DNS生成..."
WSL_CONF="/etc/wsl.conf"
# 检查文件是否存在，不存在则创建
if [ ! -f "$WSL_CONF" ]; then
    sudo touch "$WSL_CONF"
fi
# 写入/更新 network 配置（禁用自动生成 resolv.conf）
sudo sed -i '/^\[network\]/d' "$WSL_CONF"  # 删除旧的 [network] 段
sudo sed -i '/^generateResolvConf/d' "$WSL_CONF"  # 删除旧的配置项
echo -e "[network]\ngenerateResolvConf = false" | sudo tee -a "$WSL_CONF" > /dev/null

# 步骤2：备份并重建 resolv.conf
echo -e "\n2️⃣  备份并更新 resolv.conf..."
RESOLV_CONF="/etc/resolv.conf"
# 备份旧的 resolv.conf（避免覆盖）
if [ -f "$RESOLV_CONF" ]; then
    sudo mv "$RESOLV_CONF" "$RESOLV_CONF.bak.$(date +%Y%m%d%H%M%S)"
fi
# 创建新的 resolv.conf 并写入 DNS
sudo touch "$RESOLV_CONF"
sudo chmod 644 "$RESOLV_CONF"  # 确保权限可读写
echo "# WSL2 自定义DNS配置（$(date)）" | sudo tee "$RESOLV_CONF" > /dev/null
for dns in "${DNS_LIST[@]}"; do
    echo "nameserver $dns" | sudo tee -a "$RESOLV_CONF" > /dev/null
done

# 步骤3：重启 WSL 网络服务（临时生效）
echo -e "\n3️⃣  重启网络服务..."
sudo service network-manager restart 2>/dev/null  # 部分系统可能无此服务，忽略错误

# 步骤4：验证 DNS 配置
echo -e "\n4️⃣  验证 DNS 配置是否生效..."
echo "当前 resolv.conf 内容："
cat "$RESOLV_CONF"

echo -e "\n测试域名解析（hf-mirror.com）："
nslookup hf-mirror.com 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "\n✅ DNS 更新成功！请在 Windows 中执行 'wsl --shutdown' 后重新打开 WSL 以完全生效"
else
    echo -e "\n⚠️  DNS 配置已写入，但域名解析测试失败，可能是网络环境限制"
fi


