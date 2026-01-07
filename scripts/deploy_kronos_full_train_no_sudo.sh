#!/bin/bash
# deploy_kronos_full_train_no_sudo.sh
# 无需 sudo 权限！适用于普通用户

set -e

# 自动获取当前用户（无需硬编码）
USER_NAME=$(whoami)
HOME_DIR="/home/$USER_NAME"
PROJECT_DIR="$HOME_DIR/kucoin_project/kronos_test"
VENV_PATH="$PROJECT_DIR/kronos_env"
TOKENIZER_SCRIPT="$PROJECT_DIR/finetune/train_tokenizer.py"
PREDICTOR_SCRIPT="$PROJECT_DIR/finetune/train_predictor.py"
SERVICE_NAME="kronos-full-train"
SCRIPT_DIR="$PROJECT_DIR/scripts"
LAUNCHER_SCRIPT="$SCRIPT_DIR/run_full_training.sh"

echo "🚀 为用户 $USER_NAME 部署 Kronos 训练（无 sudo 模式）..."

# 创建启动脚本
mkdir -p "$SCRIPT_DIR"
cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/bash
cd "$PROJECT_DIR" || { echo "❌ 无法进入项目目录"; exit 1; }
source "$VENV_PATH/bin/activate" || { echo "❌ 无法激活虚拟环境"; exit 1; }
export PYTHONUNBUFFERED=1

echo "[$(date)] 🚀 开始 Tokenizer 训练..."
python "$TOKENIZER_SCRIPT"
if [ \$? -ne 0 ]; then
    echo "[$(date)] ❌ Tokenizer 训练失败"
    exit 1
fi

echo "[$(date)] ✅ 开始 Predictor 训练..."
python "$PREDICTOR_SCRIPT"
if [ \$? -ne 0 ]; then
    echo "[$(date)] ❌ Predictor 训练失败"
    exit 1
fi

echo "[$(date)] 🎉 训练完成！"
EOF
chmod +x "$LAUNCHER_SCRIPT"

# 创建 systemd 服务文件（用户级，无需 sudo）
mkdir -p "$HOME_DIR/.config/systemd/user"
cat > "$HOME_DIR/.config/systemd/user/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Kronos Full Training (No sudo required)
After=network.target

[Service]
Type=simple
WorkingDirectory=$PROJECT_DIR
ExecStart=$LAUNCHER_SCRIPT
Restart=on-failure
RestartSec=30
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

# 重载并启动（纯用户级操作，无需 sudo）
echo "🔄 重载 systemd 用户配置..."
systemctl --user daemon-reload

echo "⏯️  启动训练服务..."
systemctl --user enable --now "${SERVICE_NAME}.service"

# 提示用户注意事项
echo ""
echo "✅ 部署成功！服务已在后台运行。"
echo ""
echo "📌 重要提示（因未启用 linger）："
echo "   • 请保持至少一个 SSH 会话登录（不要完全登出）"
echo "   • SSH 断开重连是安全的（训练不会中断）"
echo "   • 避免执行 'logout' 或关闭所有终端会话"
echo ""
echo "📊 查看状态："
echo "    systemctl --user status ${SERVICE_NAME}.service"
echo ""
echo "📄 查看日志："
echo "    journalctl --user -u ${SERVICE_NAME}.service -f"