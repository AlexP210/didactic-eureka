"""DINOv3 ViT-S/16 patch features of the saved wrist frames -> joint 3-component PCA -> RGB.

Same backbone, weights, 224 resize and ImageNet normalization as s2p's DINOV3EncoderModel.
One PCA is fit over every frame's patches so a colour means the same thing across frames."""
import sys, numpy as np, torch
from PIL import Image
sys.path.insert(0, "/path/to/project/agents/squeeze2plan")
from s2p.models.dinov3_encoder_model import make_transform

steps = [int(s) for s in sys.argv[1].split(",")]
bb = torch.hub.load("/path/to/project/dependencies/dinov3", "dinov3_vits16",
                    source="local", pretrained=True,
                    weights="/path/to/pretrained_checkpoints/dino/dinov3_vits16_pretrain_lvd1689m-08c60483.pth").cuda().eval()
imgs = torch.stack([torch.from_numpy(np.array(Image.open(f"frames/wrist_{t:03d}.png").convert("RGB"))).permute(2, 0, 1) for t in steps])
with torch.no_grad():
    f = bb.forward_features(make_transform(224)(imgs).cuda())["x_norm_patchtokens"]  # (T, 196, 384)
T, N, D = f.shape; g = int(N ** 0.5)
X = f.reshape(-1, D).float()
X = X - X.mean(0)
_, S, V = torch.linalg.svd(X, full_matrices=False)
P = (X @ V[:3].T).cpu().numpy()
print("explained variance of top 3:", (S[:3] ** 2 / (S ** 2).sum()).cpu().numpy().round(3))
lo, hi = np.percentile(P, 1, axis=0), np.percentile(P, 99, axis=0)
P = np.clip((P - lo) / (hi - lo), 0, 1).reshape(T, g, g, 3)
for t, p in zip(steps, P):
    Image.fromarray((p * 255).astype(np.uint8)).resize((224, 224), Image.NEAREST).save(f"frames/dino_{t:03d}.png")
