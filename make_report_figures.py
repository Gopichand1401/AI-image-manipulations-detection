"""
Pick test images for the report, run the final system (v1+v2) on them, and save:
  results/report_images/
      inputs/                    copies of every candidate image
      figures/<name>_app.png     8-panel report figure per image (same as the app)
      same_photo_3_versions.png  original / Flux edit / SD edit of one test photo (first row where all three are correct)
      candidates.csv             true class, predicted class, confidence, area, mask F1 -> pick your figures from here

Run (inside aienv, from C:\\AI_Image_Project, with the app closed):
    python make_report_figures.py
Takes about 2-4 minutes on the GPU.
"""
import os, sys, shutil
from pathlib import Path

sys.argv = [sys.argv[0], "--combo"]                      # use the final system (v1+v2) exactly as the app does
import numpy as np, pandas as pd
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import app                                               # loads the four networks

P = app.PROJECT_DIR
OUT = P / "results" / "report_images"; INP = OUT / "inputs"; FIG = OUT / "figures"
for d in (INP, FIG): d.mkdir(parents=True, exist_ok=True)
app.REPORT_DIR = FIG                                      # per-image figures go here
NAMES = app.CLASS_NAMES
N_PARTB, N_EACH = 15, 5


def local(p):
    p = str(p).replace("\\", "/"); k = "AI_Image_Project/"
    return str(P / p.split(k, 1)[1]) if k in p else str(p)


# ---------------------------------------------------------------- candidate list: (name, path, true class, mask, group)
cands = []
pb = pd.read_csv(P / "ood_test" / "partB" / "index.csv")
for k, r in pb.head(N_PARTB).iterrows():
    cands += [(f"B{k:02d}_original", r["original"], 0, None, "same photo: original"),
              (f"B{k:02d}_flux_edit", r["flux_edit"], 2, r["mask"], "same photo: Flux edit"),
              (f"B{k:02d}_sd_edit", r["sd_edit"], 2, r["mask"], "same photo: SD edit")]
paths = np.load(P / "features" / "image_paths_15k.npy", allow_pickle=True)
y = np.load(P / "features" / "y_labels_15k.npy"); test_idx = np.load(P / "features" / "test_idx.npy")
for lab, tag in ((1, "diffusiondb"), (0, "coco")):
    picked = [local(paths[i]) for i in test_idx if y[i] == lab and Path(local(paths[i])).exists()][:N_EACH]
    cands += [(f"{tag}_{j}", p, lab, None, tag) for j, p in enumerate(picked)]
pa = pd.read_csv(P / "ood_test" / "partA" / "index.csv")
for src in ("midjourney6", "sd3.5"):
    for j, p in enumerate(pa[pa.source == src].path.head(N_EACH)):
        cands.append((f"{src.replace('.', '')}_{j}", p, 1, None, f"unseen generator: {src}"))
own = P / "own_data" / "own_split.csv"
if own.exists():
    o = pd.read_csv(own)
    for j, r in enumerate(o[(o.split == "test") & (o.kind == "phone_real")].head(N_EACH).itertuples()):
        cands.append((f"phone_{j}", r.path, 0, None, "your phone photo"))

# ---------------------------------------------------------------- run the final system on every candidate
rows, keep = [], {}
for name, path, true, mask, group in cands:
    try:
        img0 = Image.open(path)
    except Exception as e:
        print("skip", name, "-", str(e)[:60]); continue
    shutil.copy2(path, INP / f"{name}{Path(path).suffix}")
    img = app.prepare(img0)
    arr, hu, h7, probs, cls, area = app.run_models(img)
    ela, noise = app.forensic_maps(img)
    app.report_figure(name, arr, hu, h7, probs, cls, area, ela, noise)
    f1 = true_area = np.nan
    if isinstance(mask, str) and mask and Path(mask).exists():
        g = np.asarray(Image.open(mask).convert("L").resize(img.size, Image.Resampling.NEAREST)) > 127
        pm = h7 >= app.MASK_T
        f1 = 2 * (pm & g).sum() / (pm.sum() + g.sum() + 1e-9); true_area = 100 * g.mean()
    keep[name] = (arr, h7, probs, cls, area)
    rows.append(dict(name=name, group=group, true_class=NAMES[true], predicted=NAMES[cls], correct=cls == true,
                     confidence_pct=round(100 * float(probs[cls]), 1), real_pct=round(100 * float(probs[0]), 1),
                     full_ai_pct=round(100 * float(probs[1]), 1), partial_ai_pct=round(100 * float(probs[2]), 1),
                     predicted_area_pct=round(area, 2), true_area_pct=round(true_area, 2) if true_area == true_area else None,
                     mask_F1=round(f1, 3) if f1 == f1 else None, figure=str(FIG / f"{name}_app.png")))
    print(f"{name:22s} true {NAMES[true]:10s} -> {NAMES[cls]:10s} {100 * probs[cls]:5.1f}%  area {area:5.2f}%" + (f"  mask F1 {f1:.2f}" if f1 == f1 else ""))
df = pd.DataFrame(rows); df.to_csv(OUT / "candidates.csv", index=False)

# ---------------------------------------------------------------- combined figure: same photo, three versions
best = None
for k in range(N_PARTB):
    names = [f"B{k:02d}_original", f"B{k:02d}_flux_edit", f"B{k:02d}_sd_edit"]
    sub = df.set_index("name").reindex(names)
    if sub["correct"].fillna(False).astype(bool).all():
        score = sub["mask_F1"].fillna(0).sum()
        if best is None or score > best[0]: best = (score, names)
if best:
    names = best[1]; titles = ["untouched original", "Flux edit", "Stable Diffusion edit"]
    fig, ax = plt.subplots(2, 3, figsize=(18, 11))
    for c, (n, t) in enumerate(zip(names, titles)):
        arr, h7, probs, cls, area = keep[n]; r = df.set_index("name").loc[n]
        ax[0, c].imshow(arr); ax[0, c].set_title(t, fontsize=14)
        ax[1, c].imshow(app.region_view(arr, h7, cls))
        extra = f", area {area:.1f}%" + (f" (true {r.true_area_pct:.1f}%)" if r.true_area_pct == r.true_area_pct and r.true_area_pct is not None else "") if cls == 2 else ""
        ax[1, c].set_title(f"{NAMES[cls]} ({100 * probs[cls]:.0f}%){extra}", fontsize=13)
        for a in ax[:, c]: a.axis("off")
    plt.suptitle("Same test photo, three versions - final system (v1+v2); red = predicted AI-edited region", fontsize=15)
    plt.tight_layout(); plt.savefig(OUT / "same_photo_3_versions.png", dpi=110); plt.close(fig)
    print("\nCombined figure:", OUT / "same_photo_3_versions.png", "(row", names[0][:3] + ")")
else:
    print("\nNo Part B row with all three versions correct among the first", N_PARTB, "- raise N_PARTB at the top and run again")

print("\nCorrect per group:"); print(df.groupby("group").correct.agg(["sum", "count"]).to_string())
print("\nAll results:", OUT / "candidates.csv")
print("Per-image figures:", FIG)
