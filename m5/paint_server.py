#!/usr/bin/env python3
# m5 ローカルGPU(SDXL/ComfyUI)画像生成の公開API。CORS + per-IPレート制限つき。
# /paint と MU MAKE のβから叩く。完全無料(自前ハード)。 cloudflared: m5-paint.chatweb.ai -> :8799
import json, time, threading, urllib.request, urllib.parse, random
from collections import defaultdict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

COMFY = "http://127.0.0.1:8188"
PORT = 8799
CKPT = "Juggernaut-XL_v9_RunDiffusionPhoto_v2.safetensors"
VAE  = "sdxl_vae.safetensors"

# ── レート制限 ──
RATE_PER_HOUR = 40          # 1 IP あたり/時
QUEUE_MAX     = 4           # 同時待ち+実行の上限(超えたら429 busy)
GEN_LOCK = threading.Semaphore(1)     # GPUは1枚ずつ
QUEUE    = threading.Semaphore(QUEUE_MAX)
_hits = defaultdict(list)
_hits_lock = threading.Lock()

def allowed(ip):
    now = time.time()
    with _hits_lock:
        h = [t for t in _hits[ip] if now - t < 3600]
        _hits[ip] = h
        if len(h) >= RATE_PER_HOUR:
            return False
        h.append(now)
        return True

def comfy_post(path, data=None):
    body = json.dumps(data).encode() if data is not None else None
    req = urllib.request.Request(COMFY + path, data=body,
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=120))

def workflow(prompt, w, h, seed, steps):
    q = "masterpiece, best quality, highly detailed, sharp focus"
    neg = "lowres, bad anatomy, worst quality, low quality, watermark, text, signature, blurry"
    return {
        "1": {"inputs": {"ckpt_name": CKPT}, "class_type": "CheckpointLoaderSimple"},
        "8": {"inputs": {"vae_name": VAE}, "class_type": "VAELoader"},
        "2": {"inputs": {"text": q + ", " + prompt, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "3": {"inputs": {"text": neg, "clip": ["1", 1]}, "class_type": "CLIPTextEncode"},
        "4": {"inputs": {"width": w, "height": h, "batch_size": 1}, "class_type": "EmptyLatentImage"},
        "5": {"inputs": {"seed": seed, "steps": steps, "cfg": 6.0,
                         "sampler_name": "dpmpp_2m", "scheduler": "karras", "denoise": 1.0,
                         "model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0]}, "class_type": "KSampler"},
        "6": {"inputs": {"samples": ["5", 0], "vae": ["8", 0]}, "class_type": "VAEDecode"},
        "7": {"inputs": {"filename_prefix": "paint", "images": ["6", 0]}, "class_type": "SaveImage"},
    }

def generate(prompt, w, h, seed, steps):
    pid = comfy_post("/prompt", {"prompt": workflow(prompt, w, h, seed, steps)})["prompt_id"]
    # poll history
    for _ in range(180):  # ~90s
        time.sleep(0.5)
        hist = json.load(urllib.request.urlopen(COMFY + "/history/" + pid, timeout=30))
        if pid in hist and hist[pid].get("outputs"):
            imgs = hist[pid]["outputs"].get("7", {}).get("images", [])
            if imgs:
                im = imgs[0]
                qs = urllib.parse.urlencode({"filename": im["filename"],
                    "subfolder": im.get("subfolder", ""), "type": im.get("type", "output")})
                return urllib.request.urlopen(COMFY + "/view?" + qs, timeout=60).read()
    raise RuntimeError("timeout waiting for ComfyUI")

class H(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    def _cors(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
    def log_message(self, *a): pass
    def do_OPTIONS(self):
        self.send_response(204); self._cors(); self.send_header("Content-Length", "0"); self.end_headers()
    def _json(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self._cors()
        self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b)))
        self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        u = urllib.parse.urlparse(self.path)
        if u.path == "/health":
            return self._json(200, {"ok": True, "model": CKPT})
        if u.path != "/gen":
            return self._json(404, {"error": "not found"})
        q = urllib.parse.parse_qs(u.query)
        prompt = (q.get("prompt", [""])[0] or "").strip()[:400]
        if not prompt:
            return self._json(400, {"error": "prompt required"})
        ip = self.headers.get("CF-Connecting-IP") or self.client_address[0]
        if not allowed(ip):
            return self._json(429, {"error": "rate limit (40/hour). 少し待ってね"})
        if not QUEUE.acquire(blocking=False):
            return self._json(429, {"error": "busy: 生成が混んでいます。少し待って再試行"})
        try:
            w = max(512, min(1024, int(q.get("w", ["768"])[0])))
            h = max(512, min(1024, int(q.get("h", ["768"])[0])))
            steps = max(8, min(30, int(q.get("steps", ["24"])[0])))
            seed = int(q.get("seed", [str(random.randint(1, 2**31))])[0])
            with GEN_LOCK:
                png = generate(prompt, w, h, seed, steps)
            self.send_response(200); self._cors()
            self.send_header("Content-Type", "image/png")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(png)))
            self.end_headers(); self.wfile.write(png)
        except Exception as e:
            self._json(500, {"error": str(e)[:200]})
        finally:
            QUEUE.release()

if __name__ == "__main__":
    print(f"paint_server on :{PORT} -> ComfyUI {COMFY} ({CKPT})")
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
