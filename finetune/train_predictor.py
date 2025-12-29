# ./finetune/train_predictor.py
import os
import sys

# Add project root to path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json
import torch
from torch.utils.data import DataLoader
from config import Config
from dataset import QlibDataset
from model.kronos import KronosTokenizer, Kronos

def create_dataloaders(config):
    train_dataset = QlibDataset('train')
    val_dataset = QlibDataset('val')
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'], shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
    return train_loader, val_loader, train_dataset, val_dataset

def train_model(model, tokenizer, device, config, save_dir):
    train_loader, val_loader, train_dataset, val_dataset = create_dataloaders(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config['predictor_learning_rate'], betas=(config['adam_beta1'], config['adam_beta2']), weight_decay=config['adam_weight_decay'])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config['predictor_learning_rate'], steps_per_epoch=len(train_loader), epochs=config['epochs'], pct_start=0.03, div_factor=10)

    start_epoch = 0
    best_val_loss = float('inf')
    checkpoint_path = os.path.join(save_dir, 'checkpoints', 'latest_checkpoint.pt')
    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        print(f"Resuming from epoch {start_epoch}")

    for epoch in range(start_epoch, config['epochs']):
        model.train()
        for i, (batch_x, batch_x_stamp) in enumerate(train_loader):
            batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
            batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)
            with torch.no_grad():
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
            token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
            token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
            logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
            loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            optimizer.step()
            scheduler.step()

            if (i + 1) % config['log_interval'] == 0:
                lr = optimizer.param_groups[0]['lr']
                print(f"[Epoch {epoch+1}/{config['epochs']}, Step {i+1}] LR {lr:.6f}, Loss: {loss.item():.4f}")

        # Validation
        model.eval()
        total_val_loss, count = 0.0, 0
        with torch.no_grad():
            for batch_x, batch_x_stamp in val_loader:
                batch_x = batch_x.squeeze(0).to(device, non_blocking=True)
                batch_x_stamp = batch_x_stamp.squeeze(0).to(device, non_blocking=True)
                token_seq_0, token_seq_1 = tokenizer.encode(batch_x, half=True)
                token_in = [token_seq_0[:, :-1], token_seq_1[:, :-1]]
                token_out = [token_seq_0[:, 1:], token_seq_1[:, 1:]]
                logits = model(token_in[0], token_in[1], batch_x_stamp[:, :-1, :])
                val_loss, _, _ = model.head.compute_loss(logits[0], logits[1], token_out[0], token_out[1])
                total_val_loss += val_loss.item()
                count += 1
        avg_val_loss = total_val_loss / count if count > 0 else 0
        print(f"Epoch {epoch+1} Val Loss: {avg_val_loss:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained(os.path.join(save_dir, 'checkpoints', 'best_model'))

        # Save latest checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_loss': best_val_loss,
        }
        torch.save(checkpoint, checkpoint_path)

    return best_val_loss

def main():
    config = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    save_dir = os.path.join(config.save_path, config.predictor_save_folder_name)
    os.makedirs(os.path.join(save_dir, 'checkpoints'), exist_ok=True)

    tokenizer = KronosTokenizer.from_pretrained(
        os.path.join(config.save_path, config.tokenizer_save_folder_name, 'checkpoints', 'best_model')
    ).to(device).eval()
    
    model = Kronos.from_pretrained(config.pretrained_predictor_path)
    model.to(device)

    best_loss = train_model(model, tokenizer, device, config.__dict__, save_dir)

    with open(os.path.join(save_dir, 'summary.json'), 'w') as f:
        json.dump({'best_val_loss': best_loss}, f, indent=4)
    print("Predictor training finished.")

if __name__ == '__main__':
    main()