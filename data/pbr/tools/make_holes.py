# builds data/pbr/Metal038_holes: Metal038 with the staggered rounded-square dimples of trak5/holes1a, regenerated at 2048
import os, sys, shutil
from PIL import Image
import numpy as np
base, dest = sys.argv[1], sys.argv[2]
N, scale = 2048, 2048 / 512
os.makedirs(dest, exist_ok=True)
y, x = np.mgrid[0:N, 0:N].astype(float) + 0.5
# distance to the nearest dimple centre on the legacy layout: rows 16 apart, alternate rows shifted half the 32 pitch
def dimple(cx, cy):
    px, py = 32 * scale, 32 * scale
    dx = (x - cx * scale + px / 2) % px - px / 2
    dy = (y - cy * scale + py / 2) % py - py / 2
    # rounded square: superellipse distance, in legacy texels
    return (np.abs(dx) ** 4 + np.abs(dy) ** 4) ** 0.25 / scale
d = np.minimum(dimple(24.5, 7), dimple(8.5, 23))
radius, bevel = 4.6, 1.2
H = np.clip((d - (radius - bevel)) / bevel, 0, 1)   # 0 at the dimple floor, 1 on the plate
H = H * H * (3 - 2 * H)
detailH = np.asarray(Image.open(os.path.join(base, 'height.png')).convert('L')).astype(float) / 255
Hf = 0.88 * H + 0.12 * detailH
Image.fromarray((Hf * 255).round().astype(np.uint8)).save(os.path.join(dest, 'height.png'), optimize=True)
# pattern normal from the height, DirectX convention (green down the image), whiteout-blended over the metal's own
gy, gx = np.gradient(H * 6.0)
pn = np.dstack([-gx, -gy, np.ones_like(H)])
dn = np.asarray(Image.open(os.path.join(base, 'normal.jpg')).convert('RGB')).astype(float) / 127.5 - 1
n = np.dstack([pn[..., 0] + dn[..., 0], pn[..., 1] + dn[..., 1], pn[..., 2] * dn[..., 2]])
n /= np.linalg.norm(n, axis=2, keepdims=True)
Image.fromarray(((n * 0.5 + 0.5) * 255).round().astype(np.uint8)).save(os.path.join(dest, 'normal.jpg'), quality=95)
# dimples are darker, rougher and occluded
arm = np.asarray(Image.open(os.path.join(base, 'arm.jpg')).convert('RGB')).astype(float)
arm[..., 0] = 255 * (0.45 + 0.55 * H)
arm[..., 1] = np.clip(arm[..., 1] + (1 - H) * 70, 0, 255)
Image.fromarray(arm.round().astype(np.uint8)).save(os.path.join(dest, 'arm.jpg'), quality=95)
col = np.asarray(Image.open(os.path.join(base, 'diffuse.jpg')).convert('RGB')).astype(float)
col *= (0.55 + 0.45 * H)[..., None]
Image.fromarray(np.clip(col, 0, 255).round().astype(np.uint8)).save(os.path.join(dest, 'diffuse.jpg'), quality=95)
print('ok', Hf.mean())
