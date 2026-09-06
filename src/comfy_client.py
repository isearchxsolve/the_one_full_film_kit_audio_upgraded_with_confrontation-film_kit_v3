#!/usr/bin/env python3
"""Minimal headless ComfyUI API client (no tunnel needed).
Talks to a local ComfyUI server started by run.sh / the notebook.
Queues a prompt graph, waits for completion, fetches outputs. Retries + resumable.
"""
import json, os, time, urllib.request, urllib.parse, uuid

class ComfyClient:
    def __init__(self, host="127.0.0.1", port=8188):
        self.base = f"http://{host}:{port}"
        self.cid = str(uuid.uuid4())

    def _post(self, path, payload):
        data = json.dumps(payload).encode()
        req = urllib.request.Request(self.base + path, data=data,
                                     headers={"Content-Type": "application/json"})
        return json.load(urllib.request.urlopen(req, timeout=60))

    def _get(self, path):
        return json.load(urllib.request.urlopen(self.base + path, timeout=60))

    def apply_overrides(self, graph, overrides):
        # naive override: set any node input whose key matches an override key
        for node in graph.get("nodes", graph).values() if isinstance(graph, dict) else []:
            ins = node.get("inputs", {})
            for k, v in overrides.items():
                if k in ins:
                    ins[k] = v
        return graph

    def run(self, graph, overrides=None, tries=3):
        if overrides:
            graph = self.apply_overrides(graph, overrides)
        last = None
        for attempt in range(tries):
            try:
                r = self._post("/prompt", {"prompt": graph, "client_id": self.cid})
                pid = r["prompt_id"]
                while True:
                    hist = self._get(f"/history/{pid}")
                    if pid in hist:
                        return hist[pid]["outputs"]
                    time.sleep(2)
            except Exception as e:
                last = e
                time.sleep(5 * (attempt + 1))
        raise RuntimeError(f"ComfyUI run failed after {tries} tries: {last}")

    def save_image(self, outputs, dest):
        for node_out in outputs.values():
            for img in node_out.get("images", []):
                q = urllib.parse.urlencode({"filename": img["filename"],
                                            "subfolder": img.get("subfolder", ""),
                                            "type": img.get("type", "output")})
                raw = urllib.request.urlopen(self.base + "/view?" + q, timeout=60).read()
                os.makedirs(os.path.dirname(dest), exist_ok=True)
                open(dest, "wb").write(raw)
                return dest
        raise RuntimeError("No image in ComfyUI outputs")
