# download_kronos_mini.py
from huggingface_hub import snapshot_download

# 下载 tokenizer 和 model 到本地目录
snapshot_download(
    repo_id="NeoQuasar/Kronos-Tokenizer-base",
    local_dir="core/pretrained_100K/tokenizer/best_model",   
    # local_dir_use_symlinks=False  # 避免 Windows symlink 问题
)

snapshot_download(
    repo_id="NeoQuasar/Kronos-base",
    local_dir="core/pretrained_100K/basemodel/best_model",    
    # local_dir_use_symlinks=False
)

print("✅ Kronos-mini and Tokenizer downloaded to ./core/pretrained_100K/basemodel/best_model and ./core/pretrained_100K/tokenizer/best_model/")