"""Figure: spectral locus of codec distortion vs spectral locus of model reliance."""
import json
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
FIG=Path('exp/figures'); d=json.load(open('exp/out/band_damage.json'))
plt.rcParams.update({"figure.dpi":150,"savefig.dpi":200,"font.size":9,
 "axes.spines.top":False,"axes.spines.right":False,"axes.grid":True,
 "grid.alpha":.25,"axes.axisbelow":True,"figure.facecolor":"white",
 "axes.titlesize":10,"axes.titleweight":"bold","legend.frameon":False})
bands=list(d['attribution_ref'].keys()); x=np.arange(len(bands))
fig,ax=plt.subplots(figsize=(7.4,3.8))
nsr=[d['nsr_db']['64'][b] for b in bands]
ax.bar(x,nsr,color="#dd6b20",alpha=.8,width=.6,label="codec distortion (NSR, dB)")
ax.set_ylabel("noise-to-signal ratio (dB)\nhigher = more codec distortion",color="#a04000")
ax.set_ylim(min(nsr)-3,0)
ax2=ax.twinx(); ax2.grid(False)
attr=[d['attribution_ref'][b] for b in bands]
ax2.plot(x,attr,"o-",color="#2b6cb0",lw=2,ms=7,label="model reliance (attribution)")
ax2.set_ylabel("mean attribution\nhigher = model relies on it more",color="#2b6cb0")
ax2.set_ylim(0,max(attr)*1.35)
ax.set_xticks(x); ax.set_xticklabels([b.replace("Hz","") for b in bands],fontsize=8)
ax.set_xlabel("frequency band (Hz)")
ax.set_title("Where MP3 does its damage vs. where the model looks (64 kbps)")
h1,l1=ax.get_legend_handles_labels(); h2,l2=ax2.get_legend_handles_labels()
ax.legend(h1+h2,l1+l2,loc="upper left",fontsize=8)
ax.annotate("most distorted band,\nmoderate reliance",xy=(5,nsr[5]),xytext=(3.7,nsr[5]+6),
            fontsize=7.5,color="#a04000",ha="center",
            arrowprops=dict(arrowstyle="->",color="#a04000",lw=1))
ax2.annotate("peak reliance,\nnear-undistorted",xy=(3,attr[3]),xytext=(1.6,attr[3]*1.12),
             fontsize=7.5,color="#2b6cb0",ha="center",
             arrowprops=dict(arrowstyle="->",color="#2b6cb0",lw=1))
fig.tight_layout(); fig.savefig(FIG/"a4_band_damage_vs_attribution.png",bbox_inches="tight")
print("  a4_band_damage_vs_attribution.png")
