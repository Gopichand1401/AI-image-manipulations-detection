"""
AI Image Detector - web interface for the final hybrid system (notebook 13, cell 8).

Upload any image and get:
  * class: Real / Full AI / Partial AI (with probabilities)
  * heatmaps: unified model (is this pixel AI?) and localization model (where was it edited?)
  * the predicted AI region and the AI-generated area in %
  * forensic maps: error level analysis and local noise level

Run on the laptop (inside the aienv environment):
    C:\\aienv\\Scripts\\activate
    pip install gradio
    python app.py            -> opens http://127.0.0.1:7860 in the browser
    python app.py --v2       -> multi-generator model (notebook 15)
    python app.py --combo    -> v1 + v2 together (notebook 16)
    python app.py --v3       -> v1+v2 fine-tuned with your phone photos and ChatGPT images (notebook 17)
    python app.py --share    -> also prints a public link you can open on a phone or send to someone (valid ~72 h)

The models are read from PROJECT_DIR/saved_models/localization/ (set AI_PROJECT_DIR to change the folder).
"""
import argparse, io, os, pickle, sys, time, warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import segmentation_models_pytorch as smp
from PIL import Image, ImageFilter, ImageOps
import matplotlib
matplotlib.use("Agg")
from matplotlib import colormaps
import matplotlib.pyplot as plt
import gradio as gr

warnings.filterwarnings("ignore")
Image.MAX_IMAGE_PIXELS = None

# ------------------------------------------------------------------ models
PROJECT_DIR = Path(os.environ.get("AI_PROJECT_DIR", r"C:\AI_Image_Project"))
MODELS_DIR = PROJECT_DIR / "saved_models" / "localization"
REPORT_DIR = PROJECT_DIR / "results" / "analysis"
CLASS_NAMES = ["Real", "Full AI", "Partial AI"]
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_AMP = DEVICE.type == "cuda"

USE_V3 = "--v3" in sys.argv                                   # python app.py --v3    -> v1+v2 plus your phone / ChatGPT images (notebook 17)
USE_COMBO = "--combo" in sys.argv and not USE_V3              # python app.py --combo -> v1 + v2 together (notebook 16)
USE_V2 = "--v2" in sys.argv and not (USE_COMBO or USE_V3)     # python app.py --v2    -> multi-generator model (notebook 15)
deploy_file = MODELS_DIR / ("final_hybrid_v3_deploy.pkl" if USE_V3 else "final_hybrid_combo_deploy.pkl" if USE_COMBO
                            else "final_hybrid_v2_deploy.pkl" if USE_V2 else "final_hybrid_deploy.pkl")
VERSION = "v3 (with phone + ChatGPT images)" if USE_V3 else "v1 + v2 combined" if USE_COMBO else "v2 (multi-generator)" if USE_V2 else "v1"
if not deploy_file.exists():
    need = " and notebook 17" if USE_V3 else " and notebook 16" if USE_COMBO else " and notebook 15" if USE_V2 else ""
    sys.exit(f"Not found: {deploy_file}\nRun notebook 13 (cell 8){need} first, or set AI_PROJECT_DIR.")
DEP = pickle.load(open(deploy_file, "rb"))
CFG, HYB = DEP["cfg"], DEP["classifier"]
MASK_T, AREA_T = DEP["mask_t"], DEP["area_t"]


def load_unet(state_dict, arch, encoder):
    m = smp.create_model(arch=arch, encoder_name=encoder, encoder_weights=None, in_channels=3, classes=1)
    m.load_state_dict(state_dict)
    return m.to(DEVICE).to(memory_format=torch.channels_last).eval()


def load_unified(run):
    return load_unet(torch.load(MODELS_DIR / f"{run}_best.pth", map_location="cpu", weights_only=False)["model"], CFG["ARCH"], CFG["ENCODER"])


def load_localizer(run):
    d = torch.load(MODELS_DIR / f"{run}_deploy.pth", map_location="cpu", weights_only=False)
    return load_unet(d["model"], d["cfg"]["ARCH"], d["cfg"]["ENCODER"])


UNI_RUNS = DEP.get("unified_runs", [DEP["unified_run"]])     # combined system: [v1, v2]; otherwise one model each
LOC_RUNS = DEP.get("loc_runs", [DEP["loc07_run"]])
UNIFIEDS = [load_unified(r) for r in UNI_RUNS]
LOCALIZERS = [load_localizer(r) for r in LOC_RUNS]
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1) * 255
print(f"[{VERSION}] Models loaded on {DEVICE}: {' + '.join(UNI_RUNS + LOC_RUNS)} | mask threshold {MASK_T}, area threshold {AREA_T}")


# ------------------------------------------------------------------ same computations as notebook 13
@torch.no_grad()
def predict_prob(model, img_arr, tta=True):
    H, W = img_arr.shape[:2]
    x = torch.from_numpy(np.array(img_arr)).to(DEVICE).permute(2, 0, 1)[None].float()
    x = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
    x = F.pad(x, (0, (-W) % 32, 0, (-H) % 32), mode="reflect")
    with torch.autocast(DEVICE.type, dtype=torch.float16, enabled=USE_AMP):
        p = torch.sigmoid(model(x).float())
        if tta:
            p = 0.5 * (p + torch.sigmoid(model(x.flip(-1)).float()).flip(-1))
    return p[0, 0, :H, :W].float()


def image_stats(p):
    f = p.flatten(); k = max(1, int(0.01 * f.numel()))
    q = torch.quantile(f[:: max(1, f.numel() // 200000)], torch.tensor([0.99, 0.95, 0.5], device=f.device))
    return np.array([(f > t).float().mean().item() for t in (0.1, 0.3, 0.5, 0.7, 0.9)] +
                    [f.mean().item(), f.std().item(), f.max().item()] + q.tolist() + [f.topk(k).values.mean().item()], np.float32)


def forensic_maps(img, block=16, quality=90):
    rgb = img.convert("RGB"); a = np.asarray(rgb).astype(np.float32)
    b = io.BytesIO(); rgb.save(b, "JPEG", quality=quality); b.seek(0)
    ela = np.abs(a - np.asarray(Image.open(b).convert("RGB")).astype(np.float32)).mean(2)
    g = rgb.convert("L")
    res = np.asarray(g).astype(np.float32) - np.asarray(g.filter(ImageFilter.MedianFilter(3))).astype(np.float32)
    H, W = (ela.shape[0] // block) * block, (ela.shape[1] // block) * block
    blk = lambda m, f: f(m[:H, :W].reshape(H // block, block, W // block, block), axis=(1, 3))
    return blk(ela, np.mean), blk(res, np.std)


def prepare(img):
    img = ImageOps.exif_transpose(img).convert("RGB")          # upright phone photos (orientation only)
    s = min(1.0, CFG["MAX_SIDE"] / max(img.size))
    if s < 1:
        img = img.resize((round(img.size[0] * s), round(img.size[1] * s)), Image.Resampling.BICUBIC)
    return img


def run_models(img):
    arr = np.asarray(img)
    pus = [predict_prob(m, arr, CFG.get("TTA", True)) for m in UNIFIEDS]
    pls = [predict_prob(m, arr, True) for m in LOCALIZERS]
    if DEP.get("combo"):                                      # same feature order as notebook 16: v1 unified, v1 loc, v2 unified, v2 loc
        feats = np.hstack([image_stats(pus[0]), image_stats(pls[0]), image_stats(pus[1]), image_stats(pls[1])])[DEP["feat_idx"]]
        w = DEP["loc_weight"]
        p7 = torch.maximum(pls[0], pls[1]) if w is None else (1 - w) * pls[0] + w * pls[1]
        pu = 0.5 * (pus[0] + pus[1])                          # shown on the page only
    else:
        pu, p7 = pus[0], pls[0]
        feats = np.hstack([image_stats(pu), image_stats(p7)])
    probs = HYB.predict_proba(feats[None])[0]
    cls = int(np.argmax(probs))
    h7, hu = p7.cpu().numpy(), pu.cpu().numpy()
    area = 0.0 if cls == 0 else 100.0 if cls == 1 else 100 * float((h7 >= AREA_T).mean())
    return arr, hu, h7, probs, cls, area


# ------------------------------------------------------------------ pictures for the page
def colorize(m, cmap, vmax=1.0):
    return (colormaps[cmap](np.clip(m / max(vmax, 1e-6), 0, 1))[..., :3] * 255).astype(np.uint8)


def blend(arr, heat, cmap="jet", alpha=0.5):
    return (arr * (1 - alpha) + colorize(heat, cmap) * alpha).astype(np.uint8)


def block_map(m, size, cmap):
    pic = Image.fromarray(colorize(m, cmap, np.percentile(m, 99)))
    return np.asarray(pic.resize(size, Image.Resampling.NEAREST))


def region_view(arr, h7, cls):
    out = arr.copy()
    if cls == 2:
        mask = h7 >= MASK_T
        out[mask] = (0.45 * out[mask] + [140, 0, 0]).astype(np.uint8)
    elif cls == 1:
        out = (0.55 * out + [115, 0, 0]).astype(np.uint8)
    return out


def report_figure(name, arr, hu, h7, probs, cls, area, ela, noise):
    fig, ax = plt.subplots(2, 4, figsize=(24, 12))
    ax[0, 0].imshow(arr); ax[0, 0].set_title("input")
    ax[0, 1].imshow(hu, cmap="jet", vmin=0, vmax=1); ax[0, 1].set_title("unified heatmap (Real / Full / Partial)")
    ax[0, 2].imshow(h7, cmap="jet", vmin=0, vmax=1); ax[0, 2].set_title("localization heatmap (edit finder)")
    ax[0, 3].imshow(region_view(arr, h7, cls)); ax[0, 3].set_title({0: "no AI region (Real)", 1: "whole image AI (Full AI)",
                                                                   2: f"edited region: {area:.1f}% of image"}[cls])
    ax[1, 0].imshow(ela, cmap="inferno", vmax=np.percentile(ela, 99)); ax[1, 0].set_title("error level analysis (re-save q90)")
    ax[1, 1].imshow(noise, cmap="viridis", vmax=np.percentile(noise, 99)); ax[1, 1].set_title("local noise level (16x16 blocks)")
    ax[1, 2].barh(CLASS_NAMES, probs * 100, color=["#2ca02c", "#d62728", "#ff7f0e"]); ax[1, 2].set_xlim(0, 100)
    for k, q in enumerate(probs):
        ax[1, 2].text(min(q * 100 + 1, 85), k, f"{q*100:.1f}%", va="center")
    ax[1, 2].set_title("class probabilities")
    ax[1, 3].axis("off")
    ax[1, 3].text(0.02, 0.6, f"Decision: {CLASS_NAMES[cls]}\nConfidence: {probs[cls]*100:.0f}%\nAI-generated area: {area:.1f}%",
                  fontsize=18, va="center")
    for a in [ax[0, 0], ax[0, 1], ax[0, 2], ax[0, 3], ax[1, 0], ax[1, 1]]:
        a.axis("off")
    plt.suptitle(f"{name}  ->  {CLASS_NAMES[cls]} ({probs[cls]*100:.0f}%),  AI area = {area:.1f}%", fontsize=16)
    plt.tight_layout()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / f"{Path(name).stem}_app.png"
    plt.savefig(out, dpi=90); plt.close(fig)
    return str(out)


def explain(cls, probs, area):
    conf = probs[cls] * 100
    lines = {0: "No AI-generated content was found. The image looks like a real photograph.",
             1: "The whole image appears to be AI-generated.",
             2: f"Part of the image appears to be AI-edited: about **{area:.1f}%** of the image (red region)."}
    note = ""
    if conf < 60:
        note = "\n\n> **Low confidence.** The image may be unlike the training data (other AI tools, heavy compression)."
    return (f"## {CLASS_NAMES[cls]}  ({conf:.0f}% confidence)\n\n{lines[cls]}\n\n"
            f"| Real | Full AI | Partial AI | AI-generated area |\n|---|---|---|---|\n"
            f"| {probs[0]*100:.1f}% | {probs[1]*100:.1f}% | {probs[2]*100:.1f}% | **{area:.1f}%** |{note}")


# ------------------------------------------------------------------ app callbacks
def analyze(image, name=None):
    if image is None:
        raise gr.Error("Please upload an image first.")
    name = name or f"upload_{time.strftime('%Y%m%d_%H%M%S')}"
    img = prepare(image)
    arr, hu, h7, probs, cls, area = run_models(img)
    ela, noise = forensic_maps(img)
    size = img.size
    fig_path = report_figure(name, arr, hu, h7, probs, cls, area, ela, noise)
    return ({n: float(p) for n, p in zip(CLASS_NAMES, probs)}, explain(cls, probs, area),
            region_view(arr, h7, cls), blend(arr, hu), blend(arr, h7),
            block_map(ela, size, "inferno"), block_map(noise, size, "viridis"), fig_path)


def analyze_batch(files):
    if not files:
        raise gr.Error("Please choose one or more images.")
    rows = []
    for f in files:
        path = f if isinstance(f, str) else f.name
        try:
            img = prepare(Image.open(path))
            _, _, _, probs, cls, area = run_models(img)
            rows.append([Path(path).name, CLASS_NAMES[cls], round(float(probs[cls]) * 100, 1),
                         round(float(probs[0]) * 100, 1), round(float(probs[1]) * 100, 1), round(float(probs[2]) * 100, 1), round(area, 2)])
        except Exception as e:
            rows.append([Path(path).name, f"error: {e}", None, None, None, None, None])
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    out = REPORT_DIR / "batch_results.csv"
    import csv
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(BATCH_COLS); w.writerows(rows)
    return rows, str(out)


BATCH_COLS = ["image", "decision", "confidence %", "Real %", "Full AI %", "Partial AI %", "AI area %"]

HOW_TO_READ = """
**Unified heatmap** - trained on real, fully AI and partially edited images: red = pixel looks AI-generated.
**Localization heatmap** - trained to find edited regions inside real photos; it draws the red region and measures the area.
**Error level analysis** - how much each 16x16 block changes when re-saved as JPEG. It follows texture; look for a region that does not match similar areas.
**Noise map** - fine-grain noise per block. Camera noise is usually even; regenerated or smoothed regions are often cleaner.
The forensic maps are visual hints only; the decision comes from the two models.

**Limits:** trained on Stable Diffusion images (Full AI) and Flux inpainting (Partial AI). Edits from other AI tools, very small edits
and heavily compressed images may be missed.
"""

with gr.Blocks(title="AI Image Detector") as demo:
    gr.Markdown(f"# AI Image Detector ({VERSION})\nUpload an image to find out whether it is **Real**, **Full AI** or **Partial AI**, "
                "where it was edited and how much of it is AI-generated.")
    with gr.Tab("Single image"):
        with gr.Row():
            with gr.Column(scale=1):
                inp = gr.Image(type="pil", label="Upload an image", height=420)
                btn = gr.Button("Analyze", variant="primary")
            with gr.Column(scale=1):
                lab = gr.Label(num_top_classes=3, label="Class probabilities")
                summ = gr.Markdown()
        with gr.Row():
            o_region = gr.Image(label="Predicted AI region (red)")
            o_unified = gr.Image(label="Unified heatmap")
            o_loc = gr.Image(label="Localization heatmap")
        with gr.Row():
            o_ela = gr.Image(label="Error level analysis")
            o_noise = gr.Image(label="Local noise level")
            o_fig = gr.File(label="Full report figure (PNG)")
        with gr.Accordion("How to read the results", open=False):
            gr.Markdown(HOW_TO_READ)
        btn.click(analyze, inputs=inp, outputs=[lab, summ, o_region, o_unified, o_loc, o_ela, o_noise, o_fig], api_name="analyze")
    with gr.Tab("Many images"):
        files = gr.File(file_count="multiple", file_types=["image"], label="Choose images")
        bbtn = gr.Button("Analyze all", variant="primary")
        table = gr.Dataframe(headers=BATCH_COLS, label="Results")
        csv_out = gr.File(label="Results as CSV")
        bbtn.click(analyze_batch, inputs=files, outputs=[table, csv_out], api_name="analyze_batch")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a temporary public link")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--v2", action="store_true", help="use the multi-generator model from notebook 15")
    ap.add_argument("--combo", action="store_true", help="use v1 + v2 together (notebook 16)")
    ap.add_argument("--v3", action="store_true", help="use v3: v1+v2 plus your phone / ChatGPT images (notebook 17)")
    args = ap.parse_args()
    demo.launch(server_name="127.0.0.1", server_port=args.port, share=args.share or "google.colab" in sys.modules,
                inbrowser=not args.share)
