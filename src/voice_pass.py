#!/usr/bin/env python3
"""Cinematic TTS voice pass for THE ONE.

Text-only voice generation, but directed like a real film performance:
- one consistent cast voice per character
- close-mic dramatic processing for lead lines
- distant reverberant processing for the recurring chorus
- no cloned voice sample required
"""
import os, sys, csv, argparse, subprocess, tempfile

VOICE_CAST = {
    "kunal":    ["am_michael", "am_adam", "am_onyx", "af_heart"],
    "gunja":    ["af_bella", "af_nicole", "af_heart", "af_sarah"],
    "observer": ["am_onyx", "am_fenrir", "am_adam", "af_heart"],
    "chorus":   ["am_echo", "am_onyx", "am_puck", "af_heart"],
}

PERFORMANCE_PREFIX = {
    "kunal": "Speak as a real close-mic dramatic film confession: restrained, wounded, truthful, never theatrical. ",
    "gunja": "Speak as a real intimate dramatic film performance: warm, intelligent, emotional, dignified, with controlled tears. ",
    "observer": "Speak as a dark philosophical film narrator: deep, calm, dangerous, wounded, with a faint metallic stillness. ",
    "chorus": "Speak as a distant whispered chorus, solemn and almost subliminal. ",
}

def read_lines(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))

def run(cmd):
    return subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

def cinematic_process(src, dst, layer, character):
    """Make generated voices sit like film audio instead of raw TTS."""
    char = (character or "").lower()
    if layer == "chorus" or char == "chorus":
        af = "highpass=f=120,lowpass=f=5200,aecho=0.8:0.65:95:0.28,aecho=0.7:0.45:220:0.20,volume=-6dB,loudnorm=I=-28:LRA=8:TP=-3"
    elif char == "observer":
        af = "highpass=f=70,lowpass=f=11500,compand=attacks=0.02:decays=0.25:points=-80/-80|-35/-28|-12/-10|0/-3,dynaudnorm=f=120:g=7:p=0.45,loudnorm=I=-19:LRA=9:TP=-2"
    elif char == "gunja":
        af = "highpass=f=95,lowpass=f=13500,compand=attacks=0.01:decays=0.18:points=-80/-80|-38/-31|-14/-11|0/-3,dynaudnorm=f=120:g=6:p=0.45,loudnorm=I=-20:LRA=10:TP=-2"
    else:
        af = "highpass=f=80,lowpass=f=12500,compand=attacks=0.015:decays=0.22:points=-80/-80|-36/-29|-12/-10|0/-3,dynaudnorm=f=120:g=7:p=0.45,loudnorm=I=-20:LRA=10:TP=-2"
    r = run(["ffmpeg", "-y", "-loglevel", "error", "-i", src, "-af", af, dst])
    if r.returncode != 0:
        print(f"[warn] voice processing failed, keeping raw audio: {r.stdout}")
        subprocess.run(["cp", src, dst], check=True)

def synth_hindi(text, desc, out):
    from transformers import AutoTokenizer
    from parler_tts import ParlerTTSForConditionalGeneration
    import soundfile as sf, torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = ParlerTTSForConditionalGeneration.from_pretrained("ai4bharat/indic-parler-tts").to(dev)
    tok = AutoTokenizer.from_pretrained("ai4bharat/indic-parler-tts")
    d = tok(desc, return_tensors="pt").to(dev)
    p = tok(text, return_tensors="pt").to(dev)
    audio = model.generate(input_ids=d.input_ids, prompt_input_ids=p.input_ids)
    sf.write(out, audio.cpu().numpy().squeeze(), model.config.sampling_rate)

def synth_english(text, character, out):
    from kokoro import KPipeline
    import soundfile as sf
    pipe = KPipeline(lang_code="a")
    char = (character or "").lower().strip()
    voices = VOICE_CAST.get(char, ["af_heart"])
    last_error = None
    directed_text = PERFORMANCE_PREFIX.get(char, "Speak as a truthful dramatic film performance. ") + text
    for voice in voices:
        try:
            for _, _, audio in pipe(directed_text, voice=voice):
                sf.write(out, audio, 24000)
                return voice
        except Exception as e:
            last_error = e
            continue
    raise RuntimeError(f"Kokoro failed for {character}: {last_error}")

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--root", default="."); a = ap.parse_args()
    root = os.path.abspath(a.root)
    sys.path.insert(0, os.path.join(root, "src"))
    from cfgutil import load_cfg
    cfg = load_cfg(root)
    work = cfg["paths"]["work"]
    vdir = os.path.join(work, "voice"); os.makedirs(vdir, exist_ok=True)
    cdir = os.path.join(work, "voice_chorus"); os.makedirs(cdir, exist_ok=True)

    for r in read_lines(os.path.join(root, cfg["paths"]["script"])):
        sid, text = r["shot_id"], (r.get("text") or "").strip()
        if not text or text.startswith("(silent"):
            continue
        layer = (r.get("layer") or "lead").strip().lower()
        character = (r.get("character") or "").strip()
        out = os.path.join(cdir if layer == "chorus" else vdir, f"{sid}.wav")
        if os.path.exists(out):
            continue
        raw = out + ".raw.wav"
        try:
            if r.get("language") == "hi":
                desc = (r.get("voice_description") or "") + " Authentic dramatic film performance, close mic, real emotion."
                synth_hindi(text, desc, raw)
                used = "indic-parler"
            else:
                used = synth_english(text, character, raw)
            cinematic_process(raw, out, layer, character)
            try: os.remove(raw)
            except OSError: pass
            print(f"[ok] voice {sid} ({character}, {r.get('language')}, {layer}, {used})")
        except Exception as e:
            print(f"[warn] TTS unavailable here for {sid}: {e}. Runs in the GPU session.")
    print("[ok] cinematic voice pass complete.")

if __name__ == "__main__":
    main()
