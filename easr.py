
import numpy as np
from PIL import Image

def _to_np(img):
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB"), dtype=np.float32) / 255.0
    else:
        arr = img.astype(np.float32)
        if arr.max() > 1.0:
            arr /= 255.0
    return arr

def _to_img(arr):
    arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)

def load_image(path):
    return Image.open(path).convert("RGB")

def save_image(img, path):
    if isinstance(img, np.ndarray):
        img = _to_img(img)
    img.save(path)


def box_blur_np(img, k=3):
    arr = _to_np(img)
    if k <= 1:
        return _to_img(arr)
    pad = k//2
    # horizontal pass: pad width only
    arr_pad_w = np.pad(arr, ((0,0),(pad,pad),(0,0)), mode="edge")
    kernel = np.ones((k,), dtype=np.float32) / k
    tmp = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=1, arr=arr_pad_w)
    # vertical pass: pad height only
    tmp_pad_h = np.pad(tmp, ((pad,pad),(0,0),(0,0)), mode="edge")
    out = np.apply_along_axis(lambda m: np.convolve(m, kernel, mode="valid"), axis=0, arr=tmp_pad_h)
    return _to_img(out)

def sobel_edges_gray(img):
    arr = _to_np(img)
    gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    Kx = np.array([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=np.float32)
    Ky = np.array([[-1,-2,-1],[0,0,0],[1,2,1]], dtype=np.float32)
    def conv2d(mat, K):
        pad = 1
        m = np.pad(mat, ((pad,pad),(pad,pad)), mode="edge")
        H,W = mat.shape
        out = np.zeros_like(mat)
        for i in range(H):
            for j in range(W):
                out[i,j] = np.sum(m[i:i+3, j:j+3]*K)
        return out
    gx = conv2d(gray, Kx)
    gy = conv2d(gray, Ky)
    mag = np.sqrt(gx*gx + gy*gy)
    mag /= (mag.max() + 1e-8)
    return mag

def unsharp_mask_edgeaware(img, blur_k=5, amount=0.5, edge_weight=None):
    arr = _to_np(img)
    blurred = _to_np(box_blur_np(img, k=blur_k))
    high = arr - blurred
    if edge_weight is None:
        edge_weight = sobel_edges_gray(img)
    edge_weight = np.expand_dims(edge_weight, 2)
    sharpened = arr + amount * high * (0.25 + 0.75*edge_weight)
    return _to_img(np.clip(sharpened, 0, 1))

def anisotropic_diffusion(img, niter=5, kappa=20.0, gamma=0.15):
    arr = _to_np(img)
    H,W,_ = arr.shape
    L = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
    for _ in range(int(niter)):
        Lp = np.pad(L, ((1,1),(1,1)), mode="edge")
        N = Lp[:-2,1:-1] - L
        S = Lp[2:,1:-1]  - L
        E = Lp[1:-1,2:]  - L
        W = Lp[1:-1,:-2] - L
        cN = np.exp(-(N/kappa)**2)
        cS = np.exp(-(S/kappa)**2)
        cE = np.exp(-(E/kappa)**2)
        cW = np.exp(-(W/kappa)**2)
        L = L + gamma*(cN*N + cS*S + cE*E + cW*W)
    eps = 1e-6
    Y = L
    denom = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] + eps)
    ratio = (arr + eps) / np.expand_dims(denom,2)
    out = np.clip(ratio * np.expand_dims(Y,2), 0, 1)
    return _to_img(out)

def resize_image(img, scale, prefilter=True, method_up="bicubic", method_down="lanczos",
                 sharpness=0.4, aa_strength=0.3, edge_preserve=0.6, diffusion_iters=3):
    if not isinstance(img, Image.Image):
        img = _to_img(_to_np(img))
    W, H = img.size
    target = (max(1, int(W*scale)), max(1, int(H*scale)))
    if scale < 1.0 and prefilter:
        k = max(3, int(2/scale)+1)
        img = box_blur_np(img, k=k)
    if scale >= 1.0:
        base = img.resize(target, Image.BICUBIC if method_up=="bicubic" else Image.LANCZOS)
    else:
        base = img.resize(target, Image.LANCZOS if method_down=="lanczos" else Image.BICUBIC)
    if sharpness > 1e-6 and scale >= 1.0:
        edges = sobel_edges_gray(base)
        amount = sharpness * (0.25 + 0.75*edge_preserve)
        base = unsharp_mask_edgeaware(base, blur_k=5, amount=amount, edge_weight=edges)
    if aa_strength > 1e-6:
        iters = max(0, int(diffusion_iters * (0.5 + 0.5*aa_strength)))
        if iters > 0:
            base = anisotropic_diffusion(base, niter=iters, kappa=25.0, gamma=0.15)
    return base

def process(path_in, path_out, scale=2.0, sharpness=0.4, aa_strength=0.3, edge_preserve=0.6,
            diffusion_iters=3, prefilter=True):
    img = load_image(path_in)
    out = resize_image(img, scale=scale, prefilter=prefilter, sharpness=sharpness,
                       aa_strength=aa_strength, edge_preserve=edge_preserve, diffusion_iters=diffusion_iters)
    save_image(out, path_out)
