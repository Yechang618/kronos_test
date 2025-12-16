# download_kronos_mini.py
from huggingface_hub import snapshot_download

# 下载 tokenizer 和 model 到本地目录
snapshot_download(
    repo_id="NeoQuasar/Kronos-Tokenizer-2k",
    # local_dir="pretrained/Kronos-Tokenizer-2k",
    # local_dir="batch/model/Kronos-Tokenizer-2k",    
    local_dir="trained/sol_1min_10s/tokenizer/best_model",   
    local_dir_use_symlinks=False  # 避免 Windows symlink 问题
)

snapshot_download(
    repo_id="NeoQuasar/Kronos-mini",
    # local_dir="pretrained/Kronos-mini",
    local_dir="trained/sol_1min_10s/basemodel/best_model",    
    local_dir_use_symlinks=False
)

print("✅ Kronos-mini and Tokenizer downloaded to ./trained/sol_1min_10s/basemodel/best_model and ./trained/sol_1min_10s/tokenizer/best_model/")