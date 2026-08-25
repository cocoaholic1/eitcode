


# ---------------- 姝ｅ鸡浣嶇疆缂栫爜 ----------------
import argparse
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset
from scipy.io import loadmat
from tqdm import tqdm
from piq import ssim

# ---------------- 閰嶇疆 ----------------
DATASET_PATH = r"E:\EIT\DATASET_FOR_EIT_SIMU_abs"
FINETUNE_DATASET_PATH = r"E:\EIT\finetunedataset"
BATCH_SIZE = 16
NUM_EPOCHS = 150
FINETUNE_EPOCHS = 100
LEARNING_RATE = 6e-4
FINETUNE_LEARNING_RATE = 5e-4
EWC_ESTIMATE_BATCHES = 80
EWC_LAMBDA = 500.0
TARGET_MIN = 0.6
TARGET_MAX = 0.8
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("浣跨敤璁惧:", DEVICE)

# 可选: "train" / "finetune_wood"
# train: 普通训练；finetune_wood: 先用旧数据估计 Fisher，再在木头数据上做软约束 EWC 微调
RUN_MODE_KEYWORD = "train"
# ---------------- 姝ｅ鸡浣嶇疆缂栫爜 ----------------
class TriGatedFusion(nn.Module):
    def __init__(self, dim=1024, hidden_dim=2048, num_layers=4, dropout=0.1):
        super(TriGatedFusion, self).__init__()
        layers = []
        in_dim = dim * 3
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.gate_norm = nn.LayerNorm(dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, space_feat, amp_feat, phase_feat):
        concat = torch.cat([space_feat, amp_feat, phase_feat], dim=-1)  # [B, 3*dim]
        x = self.mlp(concat)
        x = self.gate_norm(x)
        gates = self.sigmoid(x)  # [B, dim]
        return gates * (space_feat + amp_feat + phase_feat)

class Simple1DViT(nn.Module):
    def __init__(self, input_len=1024, patch_size=4, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(Simple1DViT, self).__init__()
        self.input_len = input_len
        self.patch_size = patch_size

        # Linear projection of patches
        self.embedding = nn.Linear(patch_size, dim)

        # Sinusoidal Position Encoding
        self.positional_encoding = self.get_sinusoidal_encoding(input_len // patch_size, dim)

        # Transformer layers
        self.transformer_blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True)
            for _ in range(depth)
        ])

        # Instead of outputting to the original input length, we output directly to input_len
        # Add a final layer that directly maps transformer output to the same length as input_len
        self.fc_out = nn.Linear(dim, input_len)  # Match input_len instead of using patches

        self.alpha = nn.Parameter(torch.tensor(0.5))
    def get_sinusoidal_encoding(self, num_patches, dim):
        """
        Generate a sinusoidal position encoding for the number of patches and dimension size.
        """
        positions = torch.arange(num_patches, dtype=torch.float).unsqueeze(1)  # (num_patches, 1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * -(math.log(10000.0) / dim))  # (dim//2,)
        pe = torch.zeros(num_patches, dim)
        pe[:, 0::2] = torch.sin(positions * div_term)  # (num_patches, dim//2)
        pe[:, 1::2] = torch.cos(positions * div_term)  # (num_patches, dim//2)
        pe = pe.unsqueeze(0)  # (1, num_patches, dim)
        return pe

    def forward(self, x):
        residual=x
        # Move positional encoding to the same device as input tensor x
        device = x.device
        self.positional_encoding = self.positional_encoding.to(device)

        # Create patches (assuming input size is divisible by patch_size)
        patches = x.view(x.size(0), -1, self.patch_size)  # [batch_size, num_patches, patch_size]
        patches = self.embedding(patches)  # [batch_size, num_patches, dim]

        # Add positional encoding
        x = patches + self.positional_encoding

        # Pass through transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Instead of outputting the 3D tensor [batch_size, num_patches, dim], use a pooling operation or
        # modify the final layer to output [batch_size, input_len] directly.
        x = x.mean(dim=1)  # Average across patches, now shape: [batch_size, dim]

        # Map from dim to input_len
        out = self.fc_out(x)  # Final output: [batch_size, input_len]
        return self.alpha*out+residual

class HybridDomainViT(nn.Module):
    def __init__(self, input_len=1024, patch_size=4, dim=768, depth=12, heads=12, mlp_dim=3072):
        super(HybridDomainViT, self).__init__()
        # Simple 1D ViT model
        self.space_vit = Simple1DViT(input_len=input_len, patch_size=patch_size, dim=dim,
                                     depth=depth, heads=heads, mlp_dim=mlp_dim)
        self.freq_vit = Simple1DViT(input_len=input_len * 2, patch_size=patch_size * 2, dim=dim,
                                    depth=depth, heads=heads, mlp_dim=mlp_dim)
        self.tri_gated_fusion = TriGatedFusion(dim=input_len)

    def forward(self, x):
        # Step 1: Process spatial domain (direct pass)
        space_out = self.space_vit(x)  # Shape: [batch_size, input_len]

        # Step 2: Process frequency domain
        # Apply FFT to the input (convert to complex)
        freq_input = torch.fft.fft(x)

        # Extract magnitude and phase
        magnitude = freq_input.abs()
        phase = torch.angle(freq_input)

        # Combine magnitude and phase into a single sequence (2048-length)
        freq_seq = torch.stack([magnitude, phase], dim=-1).flatten(start_dim=-2)

        # Pass through the 1D ViT in the frequency domain
        freq_out = self.freq_vit(freq_seq)  # Shape: [batch_size, num_patches, 2048]
          # For debugging

        # Step 3: Extract magnitude and phase from freq_out
        # Magnitude is stored in even indices, phase is stored in odd indices
        magnitude = freq_out[:,  ::2]  # Even indices as magnitude
        phase = freq_out[:,  1::2]     # Odd indices as phase

        # Step 4: Reconstruct real and imaginary parts from magnitude and phase
        real_part = magnitude * torch.cos(phase)  # Real part from magnitude and phase
        imag_part = magnitude * torch.sin(phase)  # Imaginary part from magnitude and phase

        # Reconstruct complex numbers (2048-length for each patch)
        reconstructed_complex = torch.complex(real_part, imag_part)
        reconstructed_spatial_complex = torch.fft.ifft(reconstructed_complex)  # [batch, 1024]
        amp_out = reconstructed_spatial_complex.abs()      # 骞呭害鍚戦噺
        phase_out = torch.angle(reconstructed_spatial_complex)  # 鐩镐綅鍚戦噺

        fused_out = self.tri_gated_fusion(space_out, amp_out, phase_out)
        return fused_out

# ---------------- 鎹熷け鍑芥暟 ----------------
def total_loss(pred, target):
    mse_loss = F.mse_loss(pred, target)
    pixel_loss = F.smooth_l1_loss(pred, target, beta=0.02)
    range_loss = F.relu(-pred).mean() + F.relu(pred - 1.0).mean()
    return mse_loss + pixel_loss + 0.01 * range_loss

# ---------------- 鏁版嵁鍔犺浇 ----------------
def load_all_data(root_dir):
    xs, ys = [], []
    dirs = sorted(os.listdir(root_dir), key=lambda d: int(d))
    for d in tqdm(dirs, desc="鍔犺浇鏁版嵁"):
        f = os.path.join(root_dir, d, "workspace.mat")
        if not os.path.exists(f):
            continue
        m = loadmat(f, struct_as_record=False, squeeze_me=True)
        x = torch.tensor(m["img_recon"].elem_data.astype("float32").flatten())
        y = torch.tensor(m["img_2"].elem_data.astype("float32").flatten())
       # x = torch.clamp(x, min=-0.8, max=0.0)  #
        x_min = x.min()
        x_max = x.max()
        x = (x - x_min) / (x_max - x_min + 1e-8)
        y = (y - TARGET_MIN) / (TARGET_MAX - TARGET_MIN + 1e-8)
        y = torch.clamp(y, 0.0, 1.0)
        xs.append(x)
        ys.append(y)
    return torch.stack(xs), torch.stack(ys)

def parse_args():
    parser = argparse.ArgumentParser(description="Train or finetune HDT.")
    parser.add_argument("--mode", choices=["train", "finetune_wood"], default=RUN_MODE_KEYWORD,
                        help="train: normal training; finetune_wood: soft EWC then finetune on wood data.")
    parser.add_argument("--dataset", default=DATASET_PATH, help="Dataset used by normal training and EWC estimation.")
    parser.add_argument("--wood-dataset", default=FINETUNE_DATASET_PATH, help="Wood dataset used for finetuning.")
    parser.add_argument("--pretrained", default="best_model-vit.pth", help="Checkpoint loaded before training/finetuning.")
    parser.add_argument("--epochs", type=int, default=None, help="Override epoch count.")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate.")
    parser.add_argument("--ewc-batches", type=int, default=EWC_ESTIMATE_BATCHES,
                        help="Number of old-data batches used to estimate EWC importance.")
    parser.add_argument("--ewc-lambda", type=float, default=EWC_LAMBDA,
                        help="Soft EWC penalty weight used during finetuning.")
    return parser.parse_args()

def estimate_ewc_importance(model, loader, criterion, max_batches):
    fisher_dict = {}
    params_star = {}
    model.eval()
    for name, param in model.named_parameters():
        if param.requires_grad:
            fisher_dict[name] = torch.zeros_like(param, device="cpu")
            params_star[name] = param.detach().cpu().clone()

    for batch_idx, (xb, yb) in enumerate(tqdm(loader, desc="EWC估计关键参数")):
        if batch_idx >= max_batches:
            break
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        model.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        loss.backward()
        for name, param in model.named_parameters():
            if param.grad is not None:
                fisher_dict[name] += param.grad.detach().cpu().pow(2)

    used_batches = min(len(loader), max_batches)
    if used_batches > 0:
        fisher_dict = {name: fisher / used_batches for name, fisher in fisher_dict.items()}
    model.zero_grad(set_to_none=True)
    return fisher_dict, params_star

def ewc_penalty(model, fisher_dict, params_star):
    penalty = None
    for name, param in model.named_parameters():
        if not param.requires_grad or name not in fisher_dict:
            continue
        fisher = fisher_dict[name].to(param.device)
        star = params_star[name].to(param.device)
        item = (fisher * (param - star).pow(2)).sum()
        penalty = item if penalty is None else penalty + item
    if penalty is None:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return penalty

def save_ewc_state(fisher_dict, params_star, path):
    torch.save({
        "fisher_dict": fisher_dict,
        "params_star": params_star,
    }, path)

# ---------------- Pearson ----------------
def pearson_corr(x, y):
    vx = x - x.mean(dim=1, keepdim=True)
    vy = y - y.mean(dim=1, keepdim=True)
    corr = (vx * vy).sum(dim=1) / (vx.norm(dim=1) * vy.norm(dim=1) + 1e-8)
    return corr.mean().item()

# ---------------- 涓荤▼搴?----------------
def main():
    args = parse_args()
    print(f"运行模式: {args.mode}")
    if args.mode == "finetune_wood":
        train_dataset_path = args.wood_dataset
        ewc_dataset_path = args.dataset
        num_epochs = args.epochs if args.epochs is not None else FINETUNE_EPOCHS
        learning_rate = args.lr if args.lr is not None else FINETUNE_LEARNING_RATE
        best_model_path = "best_model-vit-wood-finetune.pth"
        last_model_path = "vit_model_last_wood_finetune.pth"
        log_path = "train_log_wood_finetune.txt"
        ewc_state_path = "ewc_state_wood.pt"
        print(f"微调模式: 先用 {ewc_dataset_path} 做 EWC 重要性估计，再在 {train_dataset_path} 上软约束微调")
    else:
        train_dataset_path = args.dataset
        ewc_dataset_path = None
        num_epochs = args.epochs if args.epochs is not None else NUM_EPOCHS
        learning_rate = args.lr if args.lr is not None else LEARNING_RATE
        best_model_path = "best_model-vit.pth"
        last_model_path = "vit_model_last.pth"
        log_path = "train_log.txt"
        ewc_state_path = None

    imgs, labels = load_all_data(train_dataset_path)
    loader = DataLoader(TensorDataset(imgs, labels),
                        batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    model =HybridDomainViT().to(DEVICE)

    # ========== 鏂板锛氳浇鍏ラ璁粌妯″瀷 ==========
    load_pretrained = True
    if args.mode == "finetune_wood" and os.path.exists(best_model_path):
        pretrained_path = best_model_path
        print(f"检测到上次木头微调模型，继续微调: {pretrained_path}")
    else:
        pretrained_path = args.pretrained
    if load_pretrained and os.path.exists(pretrained_path):
        print(f"馃攧 鍔犺浇棰勮缁冩ā鍨? {pretrained_path}")
        state_dict = torch.load(pretrained_path, map_location=DEVICE)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            print("鈿狅笍 缂哄け鐨勬潈閲?", missing)
        if unexpected:
            print("鈿狅笍 鏈娇鐢ㄧ殑鏉冮噸:", unexpected)
    else:
        print('No pretrained model loaded; training from scratch.')
    # =====================================

    criterion = total_loss
    fisher_dict = None
    params_star = None

    if args.mode == "finetune_wood":
        ewc_imgs, ewc_labels = load_all_data(ewc_dataset_path)
        ewc_loader = DataLoader(TensorDataset(ewc_imgs, ewc_labels),
                                batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
        fisher_dict, params_star = estimate_ewc_importance(model, ewc_loader, criterion, args.ewc_batches)
        save_ewc_state(fisher_dict, params_star, ewc_state_path)
        print(f"EWC状态已保存: {ewc_state_path}, lambda={args.ewc_lambda}")

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate
    )

    scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_ssim = 0.9069

    log_file = open(log_path, "w")

    for epoch in range(1, num_epochs + 1):
        model.train()
        running = 0.0
        running_data = 0.0
        running_ewc = 0.0

        for xb, yb in tqdm(loader, desc=f"Epoch {epoch}/{num_epochs}"):
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            pred = model(xb)

            data_loss = criterion(pred, yb)
            if args.mode == "finetune_wood" and fisher_dict is not None and params_star is not None:
                raw_ewc_loss = ewc_penalty(model, fisher_dict, params_star)
                ewc_loss = args.ewc_lambda * raw_ewc_loss
                loss = data_loss + ewc_loss
                running_ewc += ewc_loss.item()
            else:
                ewc_loss = torch.tensor(0.0, device=DEVICE)
                loss = data_loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running += loss.item()
            running_data += data_loss.item()

        avg = running / len(loader)
        avg_data = running_data / len(loader)
        avg_ewc = running_ewc / len(loader)
        model.eval()
        eval_loss = 0.0
        ssim_sum = 0.0
        cos_sum = 0.0
        pearson_sum = 0.0
        num_batches = 0
        with torch.no_grad():
            for eval_xb, eval_yb in loader:
                eval_xb, eval_yb = eval_xb.to(DEVICE), eval_yb.to(DEVICE)
                eval_pred = model(eval_xb)
                eval_loss += criterion(eval_pred, eval_yb).item()
                pred_img = eval_pred.detach().view(-1, 1, 32, 32).clamp(0, 1)
                gt_img = eval_yb.detach().view(-1, 1, 32, 32).clamp(0, 1)
                ssim_sum += ssim(pred_img, gt_img, data_range=1.0).item()
                cos_sum += F.cosine_similarity(eval_pred.detach(), eval_yb.detach(), dim=1).mean().item()
                pearson_sum += pearson_corr(eval_pred.detach(), eval_yb.detach())
                num_batches += 1
        eval_loss /= max(num_batches, 1)
        ssim_val = ssim_sum / max(num_batches, 1)
        cos_sim = cos_sum / max(num_batches, 1)
        pearson_r = pearson_sum / max(num_batches, 1)
        model.train()

        log_str = f"Epoch {epoch}  TrainLoss {1000 * avg:.6f} | DataLoss {1000 * avg_data:.6f} | EWCLoss {1000 * avg_ewc:.6f} | EvalLoss {1000 * eval_loss:.6f} | SSIM {ssim_val:.4f} | CosSim {cos_sim:.4f} | PearsonR {pearson_r:.4f}"
        print(log_str)
        #print(f"alphaspace: {model.space_vit.alpha.item()}, alphafreq: {model.freq_vit.alpha.item()}")
        log_file.write(log_str + "\n")
        log_file.flush()

        scheduler.step()

        if ssim_val > best_ssim:
            best_ssim = ssim_val
            torch.save(model.state_dict(), best_model_path)
            print(f"New best SSIM {best_ssim:.4f}, model saved to {best_model_path}")

    torch.save(model.state_dict(), last_model_path)
    print(f"鉁?璁粌瀹屾垚锛屾渶鍚庢ā鍨嬩繚瀛樺湪 {last_model_path}")
    log_file.close()
    #os.system("shutdown /s /t 30")

if __name__ == "__main__":
    main()
