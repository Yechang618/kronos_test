# train_sequential_multisymbol.py
import os
import sys
import shutil
import yaml
from pathlib import Path

if sys.version_info < (3, 9):
    from backports import zoneinfo
    sys.modules["zoneinfo"] = zoneinfo

sys.path.insert(0, os.path.expanduser("~/.local/lib/python3.8/site-packages"))
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import argparse
import torch
import torch.distributed as dist

# Assume these exist in your project structure
sys.path.append('../')
from model import Kronos, KronosTokenizer, KronosPredictor
from config_loader import CustomFinetuneConfig
from finetune_tokenizer import train_tokenizer, set_seed, setup_logging as setup_tokenizer_logging
from finetune_base_model import train_model, create_dataloaders, setup_logging as setup_basemodel_logging


class MultiSymbolSequentialTrainer:
    def __init__(self, base_config_path: str, symbols, data_root="batch/data/OCHL"):
        self.base_config_path = base_config_path
        self.symbols = symbols
        self.data_root = data_root
        self._load_base_config()

    def _load_base_config(self):
        with open(self.base_config_path, 'r', encoding='utf-8') as f:
            self.base_config = yaml.safe_load(f)

    def _generate_symbol_config(self, symbol: str, temp_dir: str = "temp_configs") -> str:
        os.makedirs(temp_dir, exist_ok=True)
        config = self.base_config.copy()

        # Update data path
        config['data']['data_path'] = f"{self.data_root}/{symbol}USDT_task3.csv"

        # Update experiment & model paths
        exp_name = f"task3_{symbol}"
        config['model_paths']['exp_name'] = exp_name
        config['experiment']['name'] = f"kronos_{symbol}_finetune"

        # Ensure output directories are symbol-specific
        output_config = config.copy()
        output_path = os.path.join(temp_dir, f"config_task3_{symbol}.yaml")
        with open(output_path, 'w', encoding='utf-8') as f:
            yaml.dump(output_config, f, allow_unicode=True, indent=2)
        return output_path

    def train_symbol(self, symbol: str):
        print(f"\n{'='*70}")
        print(f"🚀 STARTING TRAINING FOR SYMBOL: {symbol}")
        print(f"{'='*70}")

        config_path = self._generate_symbol_config(symbol)

        # Reuse original SequentialTrainer per symbol
        trainer = SequentialTrainer(config_path)
        success = trainer.run_training()

        if not success:
            print(f"❌ TRAINING FAILED FOR {symbol}")
            return False
        else:
            print(f"✅ TRAINING SUCCESS FOR {symbol}")
            return True

    def run_all(self):
        total_start = time.time()
        results = {}

        for symbol in self.symbols:
            try:
                success = self.train_symbol(symbol)
                results[symbol] = success
            except Exception as e:
                print(f"💥 CRITICAL ERROR FOR {symbol}: {e}")
                results[symbol] = False
                continue

        total_time = time.time() - total_start

        print("\n" + "="*70)
        print("📊 FINAL TRAINING SUMMARY")
        print("="*70)
        for sym, ok in results.items():
            status = "✅ SUCCESS" if ok else "❌ FAILED"
            print(f"  {sym}: {status}")
        print(f"\nTotal time: {total_time / 60:.2f} minutes")
        print("="*70)

        # Optionally clean up temp configs
        # shutil.rmtree("temp_configs", ignore_errors=True)


# Reuse your original SequentialTrainer class (unchanged)
class SequentialTrainer:
    def __init__(self, config_path: str = None):
        self.config = CustomFinetuneConfig(config_path)
        self.rank = int(os.environ.get("RANK", "0"))
        self.world_size = int(os.environ.get("WORLD_SIZE", "1"))
        self.local_rank = int(os.environ.get("LOCAL_RANK", str(self.config.device_id if hasattr(self.config, 'device_id') else 0)))
        self.device = self._setup_device()
        self.config.print_config_summary()

    def _setup_device(self):
        if self.config.use_cuda and torch.cuda.is_available():
            torch.cuda.set_device(self.local_rank)
            device = torch.device(f"cuda:{self.local_rank}")
        else:
            device = torch.device("cpu")
        if self.rank == 0:
            print(f"Using device: {device} (rank={self.rank}, world_size={self.world_size})")
        return device

    def _setup_distributed(self):
        if self.world_size > 1 and torch.cuda.is_available():
            backend = os.environ.get("DIST_BACKEND", "nccl").lower()
            if not dist.is_initialized():
                dist.init_process_group(backend=backend)
            if self.rank == 0:
                print(f"Distributed training initialized: backend={backend}")
        else:
            if self.rank == 0:
                print("Single-device training")

    def _check_existing_models(self):
        tokenizer_exists = os.path.exists(self.config.tokenizer_best_model_path)
        basemodel_exists = os.path.exists(self.config.basemodel_best_model_path)
        if self.rank == 0:
            print(f"Tokenizer exists: {tokenizer_exists}, Basemodel exists: {basemodel_exists}")
        return tokenizer_exists, basemodel_exists

    def _create_directories(self):
        os.makedirs(self.config.tokenizer_save_path, exist_ok=True)
        os.makedirs(self.config.basemodel_save_path, exist_ok=True)

    def train_tokenizer_phase(self):
        print("\n" + "="*60)
        print("🔤 Tokenizer Fine-tuning Phase")
        print("="*60)
        tokenizer_exists, _ = self._check_existing_models()
        if tokenizer_exists and self.config.skip_existing:
            print("Tokenizer already exists, skipping.")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_tokenizer_logging(self.config.exp_name, log_dir, self.rank)
        set_seed(self.config.seed)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            tokenizer = KronosTokenizer.from_pretrained(self.config.pretrained_tokenizer_path)
        else:
            import json
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = KronosTokenizer(**{k: v for k, v in arch.items() if k in [
                'd_in', 'd_model', 'n_heads', 'ff_dim', 'n_enc_layers', 'n_dec_layers',
                'ffn_dropout_p', 'attn_dropout_p', 'resid_dropout_p',
                's1_bits', 's2_bits', 'beta', 'gamma0', 'gamma', 'zeta', 'group_size'
            ]})
        tokenizer = tokenizer.to(self.device)

        best_val_loss = train_tokenizer(tokenizer, self.device, self.config, self.config.tokenizer_save_path, logger)
        if self.rank == 0:
            print(f"🔤 Tokenizer training done. Best val loss: {best_val_loss:.4f}")
        return True

    def train_basemodel_phase(self):
        print("\n" + "="*60)
        print("🧠 Basemodel Fine-tuning Phase")
        print("="*60)
        _, basemodel_exists = self._check_existing_models()
        if basemodel_exists and self.config.skip_existing:
            print("Basemodel already exists, skipping.")
            return True

        log_dir = os.path.join(self.config.base_save_path, "logs")
        logger = setup_basemodel_logging(self.config.exp_name, log_dir, self.rank)
        set_seed(self.config.seed)

        if getattr(self.config, 'pre_trained_tokenizer', True):
            tokenizer = KronosTokenizer.from_pretrained(self.config.finetuned_tokenizer_path)
        else:
            import json
            cfg_path = os.path.join(self.config.pretrained_tokenizer_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            tokenizer = KronosTokenizer(**{k: v for k, v in arch.items() if k in [
                'd_in', 'd_model', 'n_heads', 'ff_dim', 'n_enc_layers', 'n_dec_layers',
                'ffn_dropout_p', 'attn_dropout_p', 'resid_dropout_p',
                's1_bits', 's2_bits', 'beta', 'gamma0', 'gamma', 'zeta', 'group_size'
            ]})
        tokenizer = tokenizer.to(self.device)

        if getattr(self.config, 'pre_trained_predictor', True):
            model = Kronos.from_pretrained(self.config.pretrained_predictor_path)
        else:
            import json
            cfg_path = os.path.join(self.config.pretrained_predictor_path, 'config.json')
            with open(cfg_path, 'r') as f:
                arch = json.load(f)
            model = Kronos(**{k: v for k, v in arch.items() if k in [
                's1_bits', 's2_bits', 'n_layers', 'd_model', 'n_heads', 'ff_dim',
                'ffn_dropout_p', 'attn_dropout_p', 'resid_dropout_p', 'token_dropout_p', 'learn_te'
            ]})
        model = model.to(self.device)

        best_val_loss = train_model(model, tokenizer, self.device, self.config, self.config.basemodel_save_path, logger)
        if self.rank == 0:
            print(f"🧠 Basemodel training done. Best val loss: {best_val_loss:.4f}")
        return True

    def run_training(self):
        if self.rank == 0:
            print(f"🚀 Starting training for: {self.config.exp_name}")

        self._setup_distributed()
        self._create_directories()

        try:
            if self.config.train_tokenizer:
                if not self.train_tokenizer_phase():
                    return False
            if self.config.train_basemodel:
                if not self.train_basemodel_phase():
                    return False

            if self.rank == 0:
                print("✅ Training completed successfully!")
            return True

        except Exception as e:
            if self.rank == 0:
                print(f"💥 Error: {e}")
            import traceback
            traceback.print_exc()
            return False

        finally:
            if dist.is_available() and dist.is_initialized():
                dist.barrier()


def main():
    parser = argparse.ArgumentParser(description='Multi-Symbol Sequential Fine-tuning')
    parser.add_argument('--base_config', type=str, default='config_task3_SOL.yaml',
                        help='Base config template (symbol-agnostic)')
    parser.add_argument('--symbols', nargs='+', default=[
        "SOL", "BNB", "ZEC", "KAITO", "DOT", "ETH", "BTC", "LTC", "XRP", "ADA", "DOGE", "AVAX", "ETC", "TAO"
    ], help='List of symbols to train on')
    parser.add_argument('--data_root', type=str, default='batch/data/OCHL',
                        help='Root directory for dataset files')
    parser.add_argument('--skip-tokenizer', action='store_true')
    parser.add_argument('--skip-basemodel', action='store_true')
    parser.add_argument('--skip-existing', action='store_true')

    args = parser.parse_args()

    # Save original CLI args for per-symbol override
    cli_args = {
        'skip_tokenizer': args.skip_tokenizer,
        'skip_basemodel': args.skip_basemodel,
        'skip_existing': args.skip_existing
    }

    trainer = MultiSymbolSequentialTrainer(
        base_config_path=args.base_config,
        symbols=args.symbols,
        data_root=args.data_root
    )

    # Inject CLI overrides into global behavior via environment or config patching
    # (For simplicity, we assume SequentialTrainer reads from config only;
    #  so we update base_config before generating per-symbol configs)
    # But easier: just modify run_training to accept overrides → already handled in original main

    # However, to respect --skip-* flags globally, we patch the base config
    with open(args.base_config, 'r') as f:
        base_cfg = yaml.safe_load(f)

    if args.skip_tokenizer:
        base_cfg['experiment']['train_tokenizer'] = False
    if args.skip_basemodel:
        base_cfg['experiment']['train_basemodel'] = False
    if args.skip_existing:
        base_cfg['experiment']['skip_existing'] = True

    # Save modified base config temporarily
    temp_base = "temp_configs/base_modified.yaml"
    os.makedirs("temp_configs", exist_ok=True)
    with open(temp_base, 'w') as f:
        yaml.dump(base_cfg, f)

    # Re-instantiate with modified base
    trainer = MultiSymbolSequentialTrainer(
        base_config_path=temp_base,
        symbols=args.symbols,
        data_root=args.data_root
    )

    trainer.run_all()


if __name__ == "__main__":
    main()