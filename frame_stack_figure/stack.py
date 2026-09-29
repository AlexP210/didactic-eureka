"""Stack frames as offset cards running top-left (oldest, at the back) to bottom-right (newest, in front).

    python stack.py STEPS OUT [PREFIX]   e.g. python stack.py 0,1,2 o_t wrist
"""
from PIL import Image, ImageDraw, ImageFilter
import sys
steps = [int(s) for s in sys.argv[1].split(",")]; out = sys.argv[2]
prefix = sys.argv[3] if len(sys.argv) > 3 else "wrist"
S = 3                      # upscale 224 -> 672 for print
F = 224 * S
off = int(0.16 * F)
border, shadow_blur, pad = 6, 14, 40
n = len(steps)
W = H = F + off * (n - 1) + 2 * pad
canvas = Image.new("RGBA", (W, H), (0, 0, 0, 0))
for k, t in enumerate(steps):  # oldest first, so each newer card is drawn over it
    resample = Image.NEAREST if prefix == "dino" else Image.LANCZOS
    im = Image.open(f"frames/{prefix}_{t:03d}.png").convert("RGB").resize((F, F), resample)
    x = y = pad + off * k
    sh = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    ImageDraw.Draw(sh).rectangle([x + 8, y + 10, x + F + 8, y + F + 10], fill=(0, 0, 0, 110))
    canvas = Image.alpha_composite(canvas, sh.filter(ImageFilter.GaussianBlur(shadow_blur)))
    card = Image.new("RGBA", (F + 2 * border, F + 2 * border), (255, 255, 255, 255))
    card.paste(im, (border, border))
    ImageDraw.Draw(card).rectangle([0, 0, F + 2 * border - 1, F + 2 * border - 1], outline=(60, 60, 60, 255), width=2)
    canvas.alpha_composite(card, (x - border, y - border))
canvas.save(out + ".png")
bg = Image.new("RGB", canvas.size, (255, 255, 255)); bg.paste(canvas, mask=canvas.split()[3]); bg.save(out + ".pdf", resolution=300)
print(out + ".png", canvas.size)
