# downloads Poly Haven 2k maps into <dest>/<id>/: diffuse.jpg, normal.jpg (DirectX), arm.jpg (AO, roughness, metalness)
# and height.png, the 16-bit displacement stretched to its full range
import io, json, os, sys, urllib.request
from PIL import Image
import numpy as np
dest, ids = sys.argv[1], sys.argv[2:]
def get(url):
    req = urllib.request.Request(url, headers={'User-Agent': 'remod-pbr-fetch'})
    return urllib.request.urlopen(req, timeout=120).read()
want = {'Diffuse': 'diffuse.jpg', 'nor_dx': 'normal.jpg', 'arm': 'arm.jpg'}
for id in ids:
    files = json.loads(get('https://api.polyhaven.com/files/' + id))
    info = json.loads(get('https://api.polyhaven.com/info/' + id))
    d = os.path.join(dest, id); os.makedirs(d, exist_ok=True)
    for key, name in want.items():
        out = os.path.join(d, name)
        if not os.path.exists(out):
            open(out, 'wb').write(get(files[key]['2k']['jpg']['url']))
    H = np.asarray(Image.open(io.BytesIO(get(files['Displacement']['2k']['png']['url'])))).astype(float)
    if H.ndim == 3: H = H[..., 0]
    lo, hi = np.percentile(H, 0.1), np.percentile(H, 99.9)
    Image.fromarray((np.clip((H - lo) / max(hi - lo, 1e-6), 0, 1) * 255).round().astype(np.uint8)).save(os.path.join(d, 'height.png'), optimize=True)
    json.dump({'name': info.get('name'), 'dimensions_mm': info.get('dimensions'), 'authors': list(info.get('authors', {})), 'url': 'https://polyhaven.com/a/' + id}, open(os.path.join(d, 'info.json'), 'w'))
    print(id, info.get('dimensions'), list(info.get('authors', {})))
