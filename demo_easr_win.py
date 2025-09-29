
from PIL import Image, ImageDraw, ImageFont
import numpy as np, os
from easr import resize_image, save_image

# === Change this to your desired output folder ===
out_dir = r"D:\Program Files\PyCharm 2025.1.1\PythonProject\easr_outputs"
os.makedirs(out_dir, exist_ok=True)

def make_test_image(W=320, H=180):
    img = Image.new("RGB", (W,H), (128,128,128))
    px = img.load()
    tile = 16
    for y in range(H):
        for x in range(W):
            c = 64 if ((x//tile + y//tile) % 2 == 0) else 192
            r = int(c + 63*np.sin(2*np.pi*x/W))
            g = int(c)
            b = int(c + 63*np.cos(2*np.pi*y/H))
            px[x,y] = (max(0,min(255,r)), max(0,min(255,g)), max(0,min(255,b)))
    dr = ImageDraw.Draw(img)
    msg = "EASR demo"
    try:
        fnt = ImageFont.truetype("DejaVuSans.ttf", 24)
    except:
        fnt = ImageFont.load_default()
    dr.text((10,10), msg, font=fnt, fill=(255,255,255))
    return img

if __name__ == "__main__":
    src = make_test_image()
    save_image(src, os.path.join(out_dir, "src.png"))
    down = resize_image(src, scale=0.5, sharpness=0.0, aa_strength=0.8, diffusion_iters=2)
    save_image(down, os.path.join(out_dir, "down_0p5.png"))
    up = resize_image(src, scale=2.0, sharpness=0.6, aa_strength=0.3, edge_preserve=0.7, diffusion_iters=4)
    save_image(up, os.path.join(out_dir, "up_2x.png"))
    print(f"Wrote images to: {out_dir}")
