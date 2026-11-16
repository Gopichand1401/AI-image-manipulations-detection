# Progress of notebook 15 - run:  python status.py   (inside aienv)
from pathlib import Path
import datetime, torch
P = Path(r"C:\AI_Image_Project")
coco, sd = P / "extra_train" / "cocoai", P / "extra_train" / "sdinpaint"
M, R = P / "saved_models" / "localization", P / "results" / "v2"
print("Time now:", datetime.datetime.now().strftime("%H:%M"))
print("Stage 1  COCO_AI download :", "done" if (coco / "index.csv").exists() else f"running ({len(list(coco.rglob('*.png')))} images so far)")
n = len(list(sd.glob("sd_*.png")))
print(f"Stage 2  SD edits         : {n} of about 660" + ("  (done)" if n >= 660 else ""))
for name, stage in [("loc_v2_resnet34", "Stage 3  edit finder      "), ("unified_v2_resnet34", "Stage 4  unified model    ")]:
    f = M / f"{name}_last.pth"
    if not f.exists():
        print(stage + ": not started"); continue
    try:
        ck = torch.load(f, map_location="cpu", weights_only=False)
        state = "finished (early stop)" if ck["bad"] >= 5 or ck["epoch"] >= 15 else "training"
        saved = datetime.datetime.fromtimestamp(f.stat().st_mtime).strftime("%H:%M")
        print(f"{stage}: {state} - epoch {ck['epoch']}/15, best score {ck['best']:.4f}, last epoch saved {saved}")
        for h in ck["hist"][-3:]:
            print("      ", {k: (round(float(v), 4) if isinstance(v, float) else v) for k, v in h.items()})
    except Exception:
        print(stage + ": checkpoint is being saved right now - run again in a few seconds")
print("Stage 5  v1 vs v2 compare :", "done - see results\\v2\\v1_vs_v2_summary.csv" if (R / "v1_vs_v2_summary.csv").exists() else "not finished")
