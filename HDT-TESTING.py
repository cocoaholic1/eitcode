import os
import torch
import numpy as np
from scipy.io import loadmat, savemat
from tqdm import tqdm
from inference_metrics import print_model_profile
import math
import torch.nn as nn
# ------- 閰嶇疆 -------D
DATASET_PATH = r"E:\EIT\DATASET_FOR_EIT_SIMU_abs-t"
MODEL_PATH = ("best_model-vit-wood-finetune.pth")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("浣跨敤璁惧:", DEVICE)
N_ITER = 1
TARGET_MIN = 0.6
TARGET_MAX = 0.8
class TriGatedFusion(nn.Module):
    def __init__(self, dim=1024, hidden_dim=2048, num_layers=4, dropout=0.1):
        super(TriGatedFusion, self).__init__()
        layers = []
        in_dim = dim * 3
        for i in range(num_layers):
            out_dim = hidden_dim if i < num_layers - 1 else dim * 3
            layers.append(nn.Linear(in_dim, out_dim))
            if i < num_layers - 1:
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout))
                in_dim = hidden_dim
        self.mlp = nn.Sequential(*layers)
        self.gate_norm = nn.LayerNorm(dim * 3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, space_feat, amp_feat, phase_feat):
        concat = torch.cat([space_feat, amp_feat, phase_feat], dim=-1)  # [B, 3*dim]
        x = self.mlp(concat)
        x = self.gate_norm(x)
        gates = self.sigmoid(x)  # [B, 3*dim]

        _, D = space_feat.size()
        gate_space = gates[:, :D]
        return gate_space * space_feat


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
        space_out = self.space_vit(x) # Shape: [batch_size, input_len]

        # Step 2: Process frequency domain
        # Apply FFT to the input (convert to complex)
        freq_input = torch.fft.fft(x)

        # Extract magnitude and phase
        magnitude = freq_input.abs()
        phase = torch.angle(freq_input)

        # Combine magnitude and phase into a single sequence (2048-length)
        freq_seq = torch.stack([magnitude, phase], dim=-1).flatten(start_dim=-2)

        # Pass through the 1D ViT in the frequency domain
        freq_out = self.freq_vit(freq_seq)# Shape: [batch_size, num_patches, 2048]
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

# ------- 鍔犺浇鍗曚釜鏍锋湰骞舵帹鐞?-------
def process_folder(folder_path, model, n_iter=N_ITER):
    workspace_path = os.path.join(folder_path, "workspace.mat")
    if not os.path.exists(workspace_path):
        return

    mat = loadmat(workspace_path, struct_as_record=False, squeeze_me=True)
    x = mat["img_recon"].elem_data.astype(np.float32).flatten()
    y = mat["img_2"].elem_data.astype(np.float32).flatten()

    x_min = x.min()
    x_max = x.max()
    x_norm = torch.tensor((x - x_min) / (x_max - x_min + 1e-8), dtype=torch.float32).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        pred_norm = x_norm.clone()
        # 鉁?鍏抽敭閮ㄥ垎锛氬惊鐜緭鍏?
        for i in range(n_iter):
            pred_norm = model(pred_norm)
        pred_norm = pred_norm.clamp(0.0, 1.0)
        pred = pred_norm.cpu().squeeze(0).numpy() * (TARGET_MAX - TARGET_MIN) + TARGET_MIN

    def make_img_like(ref_obj, elem_data):
        return {
            "type": getattr(ref_obj, "type", "image"),
            "name": getattr(ref_obj, "name", ""),
            "elem_data": elem_data,
            "fwd_model": {
                "nodes": ref_obj.fwd_model.nodes,
                "elems": ref_obj.fwd_model.elems,
                "type": ref_obj.fwd_model.type,
                "name": ref_obj.fwd_model.name,
                "electrode": ref_obj.fwd_model.electrode,
            }
        }

    img_input = make_img_like(mat["img_recon"], x)
    img_gt = make_img_like(mat["img_recon"], y)
    img_pred = make_img_like(mat["img_recon"], pred)

    save_path = os.path.join(folder_path, f"vit.mat")
    savemat(save_path, {
        "img_recon_input": img_input,
        "img_recon_gt": img_gt,
        "img_recon_pred": img_pred,
    }, do_compression=True)
    print(f"鉁?鎺ㄧ悊瀹屾垚 ({n_iter} 娆¤凯浠?: {save_path}")

# ------- 涓荤▼搴?-------
def main():
    model = HybridDomainViT().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE), strict=False)
    model.eval()
    sample_input = torch.zeros(1, 1024, dtype=torch.float32, device=DEVICE)
    print_model_profile("NiloViT", model, MODEL_PATH, sample_input, n_iter=N_ITER)

    all_folders = sorted(os.listdir(DATASET_PATH), key=lambda x: int(x))
    for folder in tqdm(all_folders, desc=f"鎵归噺鎺ㄧ悊 ({N_ITER} 娆¤凯浠?"):
        folder_path = os.path.join(DATASET_PATH, folder)
        if os.path.isdir(folder_path):
            process_folder(folder_path, model, n_iter=N_ITER)

if __name__ == "__main__":
    main()
