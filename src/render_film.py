#!/usr/bin/env python3
"""Render each shot via ComfyUI (Wan 2.2-S2V / Phantom / FLF2V), then
upscale -> face-restore -> interpolate. Resumable: per-shot checkpoints in work dir.
Rejection sampling: generate N takes, auto-pick the sharpest/most-stable.
"""
import os, sys, json, glob, argparse, subprocess

def load_cfg(root):
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg as _lc
    return _lc(root)

def load_shots(root, cfg):
    return json.load(open(os.path.join(root, cfg["shots_file"])))

def refs_for(shot, char, cfg, root):
    """Pick era-correct refs: young shots -> refs/kunal_young, m08 -> refs/kunal_40s."""
    if char == "kunal":
        era = shot.get("era", "young")
        sub = "kunal_young" if era == "young" else "kunal_40s"
        d = os.path.join(root, "refs", sub)
        if os.path.isdir(d) and glob.glob(os.path.join(d, "**", "*"), recursive=True):
            return d
    return os.path.join(root, "refs", char)

def done(work, sid):
    return os.path.exists(os.path.join(work, sid, "final.mp4"))

def render_shot(shot, cfg, root, work):
    from comfy_client import ComfyClient
    sid = shot["id"]
    outdir = os.path.join(work, sid)
    os.makedirs(outdir, exist_ok=True)
    if done(work, sid):
        print(f"[skip] {sid} already rendered.")
        return
    client = ComfyClient()
    wf_name = "s2v" if shot.get("talking") else ("phantom" if len(shot["chars"]) > 1 else "flf2v")
    graph = json.load(open(os.path.join(root, "workflows", f"{wf_name}.json")))
    ref_dirs = {c: refs_for(shot, c, cfg, root) for c in shot["chars"]}
    tone = json.load(open(os.path.join(root, cfg["shots_file"])))["globals"]
    takes = []
    for n in range(cfg["rejection_samples"]):
        out = client.run(graph, overrides={
            "positive": shot["prompt"] + (", " + shot["palette"] if shot.get("palette") else "") + ", " + shot.get("tone", tone["tone"]),
            "negative": shot.get("negative", tone["negative_prompt"]),
            "ref_dirs": ref_dirs,
            "width": cfg["width"], "height": cfg["height"],
            "steps": cfg["steps"], "cfg": cfg["cfg"],
            "seconds": min(shot["dur"], cfg["max_clip_seconds"]),
            "seed": 42 + n,
        })
        take = client.save_image(out, os.path.join(outdir, f"take_{n}.mp4"))
        takes.append(take)
    best = pick_best(takes)
    # post: upscale -> face restore -> interpolate (delegated to postprocess.sh)
    subprocess.run(["bash", os.path.join(root, "src", "postprocess.sh"), best,
                    os.path.join(outdir, "final.mp4"), str(cfg["upscale"]),
                    str(cfg["interpolate_fps"]), str(cfg["codeformer_fidelity"])], check=True)
    print(f"[ok] {sid} -> {outdir}/final.mp4")

def pick_best(takes):
    """Sharpest, most temporally stable take (Laplacian var over sampled frames)."""
    try:
        import cv2, numpy as np
    except Exception:
        return takes[0]
    best, best_score = takes[0], -1
    for t in takes:
        cap = cv2.VideoCapture(t); scores = []
        while True:
            ok, fr = cap.read()
            if not ok: break
            scores.append(cv2.Laplacian(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        cap.release()
        s = float(np.mean(scores)) if scores else 0
        if s > best_score: best, best_score = t, s
    return best

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only"); a = ap.parse_args()
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(root, "src"))
    cfg = load_cfg(root); work = cfg["paths"]["work"]; os.makedirs(work, exist_ok=True)
    shots = load_shots(root, cfg)["shots"]
    for shot in shots:
        if a.only and shot["id"] != a.only:
            continue
        render_shot(shot, cfg, root, work)
    print("[ok] all shots rendered.")

if __name__ == "__main__":
    main()
