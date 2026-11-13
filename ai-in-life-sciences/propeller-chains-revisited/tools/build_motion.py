#!/usr/bin/env python3
"""Render the recorded localized motion as an original MP4, GIF, and poster.

Display frames are selected from measured configurations. A single fixed camera
rotation is used throughout; playback is not a physical molecular time scale.
"""
from pathlib import Path
import json
import math
import subprocess

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "results/reference"
FIG = ROOT / "figures"
FIG.mkdir(exist_ok=True)
FONT = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
BOLD = FONT.with_name("DejaVuSans-Bold.ttf")
COLORS = ["#6baed6", "#4292c6", "#2171b5", "#084594"]
INK, MUTED, PAPER, BLUE = "#183247", "#536e83", "#ffffff", "#08519c"
z = np.load(REF / "sequential_display.npz")
XYZ = z["coordinates_angstrom"].copy()
ENERGY, PROGRESS = z["energy_above_A_kcal_mol"], z["progress"]
XYZ -= XYZ[0].mean(0)
model = json.loads((ROOT / "src/propeller_paths/data/molecule.json").read_text())
BONDS = [(a,b) for a,b,_ in model["bonds"] if a < 43 and b < 43]
profile = json.loads((REF / "sequential_profile.json").read_text())
markers = {k:v for k,v in profile["markers"].items() if k in ("A","C","N002","N015","D","M","B")}
ALIASES = {"N002":"P", "N015":"Q"}


def font(size, bold=False):
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def frame(index):
    image = Image.new("RGB", (1280,640), PAPER)
    d = ImageDraw.Draw(image)
    p = XYZ[index]
    yaw, pitch = .13, .53
    cy,sy,cp,sp = math.cos(yaw),math.sin(yaw),math.cos(pitch),math.sin(pitch)
    rotated = np.column_stack((p[:,0]*cy+p[:,2]*sy,
        p[:,1]*cp-(-p[:,0]*sy+p[:,2]*cy)*sp,
        p[:,1]*sp+(-p[:,0]*sy+p[:,2]*cy)*cp))
    scale = 47
    screen = np.column_stack((640+scale*rotated[:,0], 210-scale*rotated[:,1],rotated[:,2]))
    for a,b in sorted(BONDS,key=lambda ab:(screen[ab[0],2]+screen[ab[1],2])/2):
        xy=tuple(screen[a,:2])+tuple(screen[b,:2])
        color=COLORS[a//6] if a<24 and b<24 and a//6==b//6 else "#a7bfd1"
        d.line(xy,fill=color,width=8)
    for i in np.argsort(screen[:,2]):
        x,y,_=screen[i];radius=9 if i<24 else 6.5
        color=COLORS[i//6] if i<24 else "#9ebad0"
        d.ellipse((x-radius,y-radius,x+radius,y+radius),fill=color,outline="#ffffff",width=1)
        d.ellipse((x-4,y-4,x-1,y-1),fill="#e7eef0")
    for j,label in enumerate(("I","II","III","IV")):
        d.ellipse((54+87*j,381,65+87*j,392),fill=COLORS[j])
        d.text((73+87*j,373),"Ring "+label,fill=MUTED,font=font(16))
    d.text((935,342),f"{ENERGY[index]:.3f}",fill=BLUE,font=font(36))
    d.text((940,391),"kcal/mol above A",fill=MUTED,font=font(17))
    left,right,top,bottom=62,1208,441,558
    def point(t,e):return(left+t*(right-left),bottom-e/8.2*(bottom-top))
    for tick in (0,4,8):
        yy=point(0,tick)[1];d.line((left,yy,right,yy),fill="#dce3e7",width=1)
        d.text((30,yy-10),str(tick),fill=MUTED,font=font(14))
    coords=[point(t,e) for t,e in zip(PROGRESS,ENERGY)]
    d.line(coords,fill=BLUE,width=3)
    x,y=point(PROGRESS[index],ENERGY[index]);d.line((x,top,x,bottom),fill="#9ecae1",width=2)
    d.ellipse((x-6,y-6,x+6,y+6),fill=BLUE,outline=PAPER,width=1)
    for key,row in markers.items():
        xx,_=point(row["progress"],0);d.text((xx-7,566),ALIASES.get(key,key),fill=INK,font=font(16))
    d.text((500,605), "Recorded path from A to B", fill=MUTED, font=font(17))
    return image


def main():
    peak=int(np.argmax(ENERGY))
    poster=frame(peak);poster.save(FIG/"localized_motion_poster.png")
    indices=np.linspace(0,len(XYZ)-1,400).astype(int)
    gif=[]
    command=["ffmpeg","-y","-loglevel","error","-f","rawvideo","-pix_fmt","rgb24","-s","1280x640","-r","24","-i","-","-an","-c:v","libx264","-crf","20","-pix_fmt","yuv420p","-movflags","+faststart",str(FIG/"localized_motion.mp4")]
    process=subprocess.Popen(command,stdin=subprocess.PIPE)
    try:
        for count,index in enumerate(indices):
            image=frame(index);process.stdin.write(image.tobytes())
            if count%2==0:
                gif.append(image.resize((768,384),Image.Resampling.LANCZOS).convert("P",palette=Image.Palette.ADAPTIVE,colors=128))
    finally:
        process.stdin.close()
    if process.wait()!=0:raise RuntimeError("ffmpeg failed")
    gif[0].save(FIG/"localized_motion.gif",save_all=True,append_images=gif[1:],duration=83,loop=0,optimize=False)
    print("Wrote original molecular motion assets to",FIG)


if __name__=="__main__":main()
