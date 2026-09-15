#!/usr/bin/env python3
"""Render the title cards to raw RGBA files consumed by the renderer."""
import sys, struct
from PIL import Image, ImageDraw, ImageFont

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
FONTB = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

def spaced(draw, xy, text, font, spacing, fill, anchor_center_x):
    widths = [draw.textlength(ch, font=font) for ch in text]
    total = sum(widths) + spacing*(len(text)-1)
    x = anchor_center_x - total/2
    for ch, w in zip(text, widths):
        draw.text((x, xy), ch, font=font, fill=fill)
        x += w + spacing

def save_raw(img, path):
    with open(path, 'wb') as f:
        f.write(struct.pack('<ii', img.width, img.height))
        f.write(img.tobytes())

def main(outdir):
    W, H = 1080, 420
    img = Image.new('RGBA', (W, H), (0,0,0,0))
    d = ImageDraw.Draw(img)
    f1 = ImageFont.truetype(FONT, 66)
    f2 = ImageFont.truetype(FONT, 25)
    spaced(d, 132, "SINGULARITY", f1, 16, (255,255,255,255), W/2)
    # hairline rule
    d.line([(W/2-130, 232), (W/2+130, 232)], fill=(255,255,255,110), width=2)
    spaced(d, 256, "VUTRA", f2, 11, (210,225,255,215), W/2)
    save_raw(img, f"{outdir}/text0.raw")

    img2 = Image.new('RGBA', (W, 200), (0,0,0,0))
    d2 = ImageDraw.Draw(img2)
    f3 = ImageFont.truetype(FONT, 31)
    spaced(d2, 70, "EVENT HORIZON", f3, 16, (235,242,255,255), W/2)
    save_raw(img2, f"{outdir}/text1.raw")
    print("wrote text0.raw text1.raw")

main(sys.argv[1] if len(sys.argv)>1 else '.')
