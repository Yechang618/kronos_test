# ./finetune_25d/create_custom_tokenizer.py
import os, sys
import torch


# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from model.kronos import KronosTokenizer
from config import Config

def main():
    config = Config()
    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name, 'best_model')
    os.makedirs(save_dir, exist_ok=True)

    # 25维输入
    d_in = len(config.feature_list)  # 应为 25
    assert d_in == 56, f"Expected 56 features, got {d_in}"

    # 初始化与 Kronos-mini 兼容的 tokenizer 结构
    tokenizer = KronosTokenizer(
        d_in=d_in,
        d_model=256,
        n_heads=4,
        ff_dim=1024,
        n_enc_layers=2,
        n_dec_layers=2,
        ffn_dropout_p=0.1,
        attn_dropout_p=0.1,
        resid_dropout_p=0.1,
        s1_bits=8,
        s2_bits=8,
        beta=0.25,
        gamma0=0.1,
        gamma=0.1,
        zeta=0.01,
        group_size=16
    )

    tokenizer.save_pretrained(save_dir)
    print(f"[INFO] Custom 56D tokenizer saved to {save_dir}")

if __name__ == "__main__":
    main()