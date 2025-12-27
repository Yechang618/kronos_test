# ./finetune/train_tokenizer.py
import os
import sys

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer

def create_dataloaders(config):
    train_dataset = QlibDataset('train')
    val_dataset = QlibDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, device, config, save_dir):
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['tokenizer_learning_rate'], weight_decay=config['adam_weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['tokenizer_learning_rate'], steps_per_epoch=len(train_loader), epochs=config['epochs'], pct_start=0.03, div_factor=10)

    best_val_loss = float('inf')
    for epoch in range(config['epochs']):
        model.train()
        train_dataset.set_epoch_seed(epoch)
        for i, (batch_x, _) in enumerate(train_loader):
            batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
            zs, bsq_loss, _, _ = model(batch_x)
            z_pre, z = zs
            loss = (F.mse_loss(z_pre, batch_x) + F.mse_loss(z, batch_x) + bsq_loss) / 2
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            scheduler.step()

            if (i + 1) % config['log_interval'] == 0:
                print(f"[Epoch {epoch+1}/{config['epochs']}, Step {i+1}] Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        total_val_loss, count = 0.0, 0
        with torch.no_grad():
            for batch_x, _ in val_loader:
                batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
                _, z = model(batch_x)[0]
                total_val_loss += F.mse_loss(z, batch_x).item() * batch_x.size(0)
                count += batch_x.size(0)
        avg_val_loss = total_val_loss / count if count > 0 else 0
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = f"{save_dir}/checkpoints/best_model"
            model.save_pretrained(save_path)
            print(f"Saved best model to {save_path}")

    return best_val_loss

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    save_dir = os.path.join(config.save_path, config.tokenizer_save_folder_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    model = KronosTokenizer.from_pretrained(config.pretrained_tokenizer_path)
    model.to(device)

    best_loss = train_model(model, device, config.__dict__, save_dir)

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump({'best_val_loss': best_loss}, f, indent=4)
    print("Tokenizer training finished.")

if __name__ == '__main__':
    main()