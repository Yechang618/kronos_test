#!/bin/bash
# deploy_kronos_full_train.sh
# 一键部署 Kronos Tokenizer + Predictor 顺序训练为 systemd 用户服务

set -e  # 遇错立即退出

# ==========================
# 🔧 配置区（根据你的环境调整）
# ==========================
USER_NAME= "huazhang"
HOME_DIR="/home/$USER_NAME"
PROJECT_DIR="$HOME_DIR/kucoin_project/kronos_test"
VENV_PATH="$PROJECT_DIR/kronos_env"
TOKENIZER_SCRIPT="$PROJECT_DIR/finetune/train_tokenizer.py"
PREDICTOR_SCRIPT="$PROJECT_DIR/finetune/train_predictor.py"
SERVICE_NAME="kronos-full-train"
SCRIPT_DIR="$PROJECT_DIR/scripts"
LAUNCHER_SCRIPT="$SCRIPT_DIR/run_full_training.sh"

# ==========================
# 🛠️ 部署流程开始
# ==========================

echo "🚀 开始部署 Kronos 完整训练流程(Tokenizer → Predictor)..."

# 1. 创建脚本目录
mkdir -p "$SCRIPT_DIR"

# 2. 生成训练启动脚本（顺序执行 + 错误检查）
cat > "$LAUNCHER_SCRIPT" << EOF
#!/bin/bash
# 自动顺序训练脚本: Tokenizer → Predictor

cd "$PROJECT_DIR" || { echo "❌ 无法进入项目目录"; exit 1; }

# 激活虚拟环境
source "$VENV_PATH/bin/activate" || { echo "❌ 无法激活虚拟环境"; exit 1; }

export PYTHONUNBUFFERED=1

echo "[$(date)] 🚀 开始 Tokenizer 训练..."
python "$TOKENIZER_SCRIPT"
if [ \$? -ne 0 ]; then
    echo "[$(date)] ❌ Tokenizer 训练失败，退出。"
    exit 1
fi

echo "[$(date)] ✅ Tokenizer 训练成功，开始 Predictor 训练..."
python "$PREDICTOR_SCRIPT"
if [ \$? -ne 0 ]; then
    echo "[$(date)] ❌ Predictor 训练失败，退出。"
    exit 1
fi

echo "[$(date)] 🎉 全流程训练完成！"
EOF

chmod +x "$LAUNCHER_SCRIPT"
echo "✅ 训练启动脚本已创建: $LAUNCHER_SCRIPT"

# 3. 创建 systemd 用户服务目录
mkdir -p "$HOME_DIR/.config/systemd/user"

# 4. 生成 systemd 服务文件
cat > "$HOME_DIR/.config/systemd/user/${SERVICE_NAME}.service" << EOF
[Unit]
Description=Kronos Full Training (Tokenizer + Predictor)
After=network.target

[Service]
Type=simple
User=$USER_NAME
WorkingDirectory=$PROJECT_DIR
ExecStart=$LAUNCHER_SCRIPT
Restart=on-failure
RestartSec=60
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=default.target
EOF

echo "✅ systemd 服务文件已创建: ~/.config/systemd/user/${SERVICE_NAME}.service"

# 5. 启用 linger（关键！防止登出被杀）
echo "🔑 启用 linger（允许服务在用户登出后继续运行）..."
sudo loginctl enable-linger "$USER_NAME"

# 6. 重载 systemd 并启动服务
echo "🔄 重载 systemd 用户配置..."
systemctl --user daemon-reload

echo "⏯️  启用并启动 Kronos 全流程训练服务..."
systemctl --user enable --now "${SERVICE_NAME}.service"

# 7. 显示状态和日志提示
echo ""
echo "📊 服务状态:"
systemctl --user status "${SERVICE_NAME}.service" --no-pager

echo ""
echo "📄 实时查看完整训练日志:"
echo "    journalctl --user -u ${SERVICE_NAME}.service -f"
echo ""
echo "⏹️  手动停止训练:"
echo "    systemctl --user stop ${SERVICE_NAME}.service"
echo ""
echo "✅ 部署成功！训练已在后台稳定运行（Tokenizer → Predictor）。"