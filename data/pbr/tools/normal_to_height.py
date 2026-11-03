# makes height maps for the normal maps that have none, by integrating their slopes
# usage: python3 -I normal_to_height.py <data dir> <out dir> <cfg out> <normal map names...>
# writes <out dir>/<name without extension>.png and a cfg line for each:
#   texautoheight "<name>" <depth> <rest> <aspect>
# depth: the height range as a fraction of the texture's width, so it shows at the size the shading implies
# rest: the height most of the texture is at, which stays put while the rest moves
# aspect: width over height, since texture coordinates count each axis in its own size
import os, sys
import numpy as np
from PIL import Image

def load(data, name):
    base = name.replace('\\', '/')
    for cand in (base, os.path.splitext(base)[0] + '.png', os.path.splitext(base)[0] + '.jpg'):
        p = os.path.join(data, cand)
        if os.path.isfile(p): return Image.open(p)
    return None

LOWCUT = 2.0 # waves across the texture below which heights fade out
MINSPAN = 1.0 # texels of height below which a map is left flat
MAXDEPTH = 0.06 # of the longer side, where tubes and bulbs drawn with steep sides would sink further
MAXMISFIT = 0.4 # share of the slopes no height explains above which the map isn't a real surface

def slopes(img):
    a = np.asarray(img.convert('RGB')).astype(np.float64) / 127.5 - 1.0
    nz = np.maximum(a[..., 2], 0.25) # keeps grazing texels from blowing up
    return -a[..., 0] / nz, a[..., 1] / nz

# the integrable part of a slope field, solved over the tiling texture with the spectrum of a difference
# across one texel centred on each, so no frequency is lost or shifted
def integrate(p, q):
    h, w = p.shape
    du = 2j * np.sin(np.pi * np.fft.fftfreq(w))[None, :]
    dv = 2j * np.sin(np.pi * np.fft.fftfreq(h))[:, None]
    P, Q = np.fft.fft2(p), np.fft.fft2(q)
    den = np.abs(du) ** 2 + np.abs(dv) ** 2
    den[0, 0] = 1
    H = (np.conj(du) * P + np.conj(dv) * Q) / den
    # slopes drawn a little off add up across the whole texture into ramps, so the broadest waves are let go:
    # under about two across it, measured on the longer side so a strip keeps its length
    fu = np.fft.fftfreq(w)[None, :] * max(w, h)
    fv = np.fft.fftfreq(h)[:, None] * max(w, h)
    f2 = fu ** 2 + fv ** 2
    H *= f2 / (f2 + LOWCUT ** 2)
    H[0, 0] = 0
    return np.real(np.fft.ifft2(H))

def residual(h, p, q):
    # how much of the slope field the height fails to explain
    hu = np.roll(h, -1, 1) - h
    hv = np.roll(h, -1, 0) - h
    pu = (p + np.roll(p, -1, 1)) / 2
    qv = (q + np.roll(q, -1, 0)) / 2
    return np.mean((hu - pu) ** 2 + (hv - qv) ** 2) / max(np.mean(pu ** 2 + qv ** 2), 1e-9)

def make(img):
    p, q = slopes(img)
    p -= p.mean(); q -= q.mean()
    best = None
    # green up the image or down it: the wrong way round leaves slopes no height can have
    for sign in (1, -1):
        h = integrate(p, sign * q)
        r = residual(h, p, sign * q)
        if best is None or r < best[0]: best = (r, h, sign)
    r, h, sign = best
    lo, hi = np.percentile(h, 0.5), np.percentile(h, 99.5)
    span = max(hi - lo, 1e-6)
    hn = np.clip((h - lo) / span, 0, 1)
    return hn, span, float(np.median(hn)), r, sign

if __name__ == '__main__':
    data, out, cfg, names = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]
    lines = []
    for name in names:
        img = load(data, name)
        if img is None: print('missing', name); continue
        if img.mode in ('RGBA', 'LA') and np.ptp(np.asarray(img)[..., -1]) > 8: print('has height', name); continue
        hn, span, rest, r, sign = make(img)
        if span < MINSPAN: print('flat', name); continue
        if r > MAXMISFIT: print('no height fits', name); continue
        # in widths, as the texture coordinates count them
        depth = min(span, MAXDEPTH * max(img.size)) / img.size[0]
        key = os.path.splitext(name.replace('\\', '/'))[0]
        dest = os.path.join(out, key + '.png')
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        Image.fromarray((hn * 255).round().astype(np.uint8)).save(dest, optimize=True)
        aspect = img.size[0] / img.size[1]
        lines.append('texautoheight "%s" %.4f %.3f %.4g // fit %.2f %s' % (key, depth, rest, aspect, 1 - r, 'gl' if sign > 0 else 'dx'))
        print(lines[-1])
    with open(cfg, 'w') as f:
        f.write('// made by tools/normal_to_height.py: normal map, height range in widths, resting height, width over height\n')
        f.write('\n'.join(lines) + '\n')
