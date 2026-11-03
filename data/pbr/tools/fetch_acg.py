# downloads ambientCG 2k JPG sets and packs them like the Poly Haven ones: diffuse.jpg, normal.jpg (DirectX),
# height.png (stretched to full range), arm.jpg (AO, roughness, metalness)
import io, os, sys, zipfile, urllib.request
from PIL import Image
import numpy as np
dest, tmp, ids = sys.argv[1], sys.argv[2], sys.argv[3:]
def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'remod-pbr-fetch'}), timeout=180).read()
for id in ids:
    zpath = os.path.join(tmp, id + '_2K-JPG.zip')
    if not os.path.exists(zpath):
        open(zpath, 'wb').write(get('https://ambientcg.com/get?file=%s_2K-JPG.zip' % id))
    z = zipfile.ZipFile(zpath)
    names = {n.split('_')[-1].split('.')[0]: n for n in z.namelist() if n.lower().endswith(('.jpg', '.png'))}
    load = lambda key: Image.open(io.BytesIO(z.read(names[key])))
    d = os.path.join(dest, id); os.makedirs(d, exist_ok=True)
    load('Color').convert('RGB').save(os.path.join(d, 'diffuse.jpg'), quality=95)
    load('NormalDX').convert('RGB').save(os.path.join(d, 'normal.jpg'), quality=95)
    size = load('Color').size
    H = np.asarray(load('Displacement').convert('I;16') if 'Displacement' in names else Image.new('L', size, 128)).astype(float)
    lo, hi = np.percentile(H, 0.1), np.percentile(H, 99.9)
    Image.fromarray((np.clip((H - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(d, 'height.png'), optimize=True)
    ch = lambda key, default: np.asarray(load(key).convert('L').resize(size)) if key in names else np.full(size[::-1], default, np.uint8)
    Image.fromarray(np.dstack([ch('AmbientOcclusion', 255), ch('Roughness', 128), ch('Metalness', 0)])).save(os.path.join(d, 'arm.jpg'), quality=95)
    A = np.dstack([ch('AmbientOcclusion', 255), ch('Roughness', 128), ch('Metalness', 0)]).reshape(-1, 3).mean(0)
    print(id, size, sorted(names), 'AO %.0f rough %.0f metal %.0f' % tuple(A))
