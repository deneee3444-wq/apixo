import random
import time
import requests
import string
import re
import hashlib
import os
import secrets
import tempfile
import threading
import queue as _queue
from concurrent.futures import ThreadPoolExecutor
from functools import wraps
from flask import (
    Flask, render_template, render_template_string,
    request, jsonify, session, redirect, url_for
)

# ════════════════════════════════════════════════════════════════════════════
#  MODEL TANIMLAMALARI
# ════════════════════════════════════════════════════════════════════════════

MODEL_CONFIGS = {
    "wan-2-5-video": {
        "display_name": "Wan 2.5 Video",
        "type": "video",
        "modes": {
            "text-to-video": {
                "display_name": "Text → Video",
                "default_params": {
                    "request_type": "async", "mode": "text-to-video",
                    "prompt": "", "resolution": "480p", "duration": 5,
                    "aspect_ratio": "16:9", "watermark": False,
                    "enable_prompt_expansion": False, "negative_prompt": "",
                    "seed": "", "image_urls": [], "audio_urls": [],
                },
                "cost_table": {
                    "480p":  {5: 0.2, 10: 0.4},
                    "720p":  {5: 0.4, 10: 0.8},
                    "1080p": {5: 0.7, 10: 1.4},
                },
                "supported_resolutions": ["480p", "720p", "1080p"],
                "supported_durations":   [5, 10],
                "supported_ratios":      ["16:9", "9:16", "1:1", "4:3", "3:4"],
                "needs_image": False,
                "supports_audio": True,
            },
            "image-to-video": {
                "display_name": "Image → Video",
                "default_params": {
                    "request_type": "async", "mode": "image-to-video",
                    "prompt": "", "resolution": "480p", "duration": 5,
                    "aspect_ratio": "16:9", "watermark": False,
                    "enable_prompt_expansion": False, "negative_prompt": "",
                    "seed": "", "image_urls": [], "audio_urls": [],
                },
                "cost_table": {
                    "480p":  {5: 0.2, 10: 0.4},
                    "720p":  {5: 0.4, 10: 0.8},
                    "1080p": {5: 0.7, 10: 1.4},
                },
                "supported_resolutions": ["480p", "720p", "1080p"],
                "supported_durations":   [5, 10],
                "supported_ratios":      ["16:9", "9:16", "1:1", "4:3", "3:4"],
                "needs_image": True,
                "max_ref_images": 1,
                "supports_audio": True,
            },
        },
    },
    "wan-2-7-image": {
        "display_name": "Wan 2.7 Image",
        "type": "image",
        "modes": {
            "omni-image": {
                "display_name": "Omni Image (Standart)",
                "default_params": {
                    "request_type": "async", "mode": "omni-image",
                    "prompt": "", "resolution": "2k", "num_images": 1,
                    "enable_sequential": False, "watermark": False,
                    "thinking_mode": True, "image_urls": [],
                    "negative_prompt": "", "seed": "",
                },
                "estimated_credits": 0.024,
                "supported_resolutions_t2i":     ["1k", "2k"],
                "supported_resolutions_img2img": ["1k", "2k"],
                "max_ref_images": 9,
            },
            "omni-image-pro": {
                "display_name": "Omni Image Pro",
                "default_params": {
                    "request_type": "async", "mode": "omni-image-pro",
                    "prompt": "", "resolution": "4k", "num_images": 1,
                    "enable_sequential": False, "watermark": False,
                    "thinking_mode": True, "image_urls": [],
                    "negative_prompt": "", "seed": "",
                },
                "estimated_credits": 0.06,
                "supported_resolutions_t2i":     ["1k", "2k", "4k"],
                "supported_resolutions_img2img": ["1k", "2k"],
                "max_ref_images": 9,
            },
        },
    },
}

# ════════════════════════════════════════════════════════════════════════════
#  PROXY SİSTEMİ
# ════════════════════════════════════════════════════════════════════════════

PROXYSCRAPE_URL = (
    "https://api.proxyscrape.com/v4/free-proxy-list/get"
    "?request=display_proxies"
    "&proxy_format=protocolipport"
    "&format=text"
)

def fetch_proxies() -> list:
    """ProxyScrape'den proxy listesinin TAMAMINI çeker."""
    print("[*] Proxy listesi çekiliyor...")
    try:
        r = requests.get(PROXYSCRAPE_URL, timeout=10)
        proxies = [line.strip() for line in r.text.splitlines() if line.strip()]
        random.shuffle(proxies)
        print(f"[*] {len(proxies)} proxy bulundu, tümü taranacak.")
        return proxies
    except Exception as e:
        print(f"[-] Proxy listesi çekilemedi: {e}")
        return []

def test_proxy(proxy_url: str, test_url: str = "https://apixo.ai", timeout: int = 5) -> bool:
    """Proxy'nin apixo.ai'ye ulaşabildiğini test eder."""
    try:
        proxies = {"http": proxy_url, "https": proxy_url}
        r = requests.get(test_url, proxies=proxies, timeout=timeout)
        return r.status_code < 500
    except Exception:
        return False

def find_working_proxy(max_workers: int = 30):
    """Tüm proxy listesini çok thread'li olarak tarar. İlk çalışanı döndürür."""
    proxy_list = fetch_proxies()
    if not proxy_list:
        return None

    result_q    = _queue.Queue()
    found_event = threading.Event()
    counter_lock = threading.Lock()
    tested_count = [0]
    total        = len(proxy_list)

    def probe(proxy: str):
        if found_event.is_set():
            return

        ok = test_proxy(proxy)

        with counter_lock:
            tested_count[0] += 1
            idx = tested_count[0]
            last = (idx == total)

        if ok and not found_event.is_set():
            found_event.set()
            result_q.put(proxy)
            print(f"  [+] Çalışan proxy bulundu [{idx}/{total}]: {proxy}")
        else:
            if last:
                result_q.put(None)

    print(f"[*] Paralel tarama başlıyor ({max_workers} thread)...")
    executor = ThreadPoolExecutor(max_workers=max_workers)
    executor.map(lambda p: probe(p), proxy_list)

    working = result_q.get()

    found_event.set()
    executor.shutdown(wait=False, cancel_futures=True)

    if working:
        return working

    print("[-] Çalışan proxy bulunamadı.")
    return None

# ════════════════════════════════════════════════════════════════════════════
#  SPAMOK + OTP
# ════════════════════════════════════════════════════════════════════════════

class ApixoTemp:
    def random_email(self, length=15) -> str:
        return ''.join(
            random.SystemRandom().choice(string.ascii_lowercase + string.digits)
            for _ in range(length)
        ) + '@spamok.com'

    def generate_fingerprint(self, email: str) -> str:
        raw = f"{email}{time.time()}"
        return hashlib.md5(raw.encode()).hexdigest()

    def get_otp(self, email: str, timeout=60) -> str | None:
        address = email.replace('@spamok.com', '')
        for i in range(timeout):
            try:
                r = requests.get(f'https://api.spamok.com/v2/EmailBox/{address}', timeout=10)
                data = r.json()
                for mail in data.get('mails', []):
                    subject = mail.get('subject', '')
                    from_display = mail.get('fromDisplay', '')
                    if 'APIXO' in from_display or 'verification' in subject.lower():
                        mail_id = mail['id']
                        email_r = requests.get(
                            f'https://api.spamok.com/v2/Email/{address}/{mail_id}', timeout=10
                        )
                        body = email_r.json()
                        plain = body.get('messagePlain', '')
                        match = re.search(r'\b(\d{6})\b', plain)
                        if match: return match.group(1)
                        html = body.get('messageHtml', '')
                        match = re.search(r'letter-spacing:8px[^>]*>(\d{6})<', html)
                        if match: return match.group(1)
            except Exception:
                pass
            time.sleep(2)
        return None

# ════════════════════════════════════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════════════════════════════════════

def apixo_auto_login():
    temp = ApixoTemp()
    email = temp.random_email()
    fingerprint = temp.generate_fingerprint(email)
    base_url = "https://apixo.ai"

    s = requests.Session()
    s.headers.update({
        "Origin": base_url,
        "Referer": f"{base_url}/models/image",
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "sec-ch-ua": '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile": "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest": "empty",
        "sec-fetch-mode": "cors",
        "sec-fetch-site": "same-origin",
    })

    # ════════════════════════════════════════════════════════════════════════════
    #  DÖNGÜ: OTP GÖNDERİMİ İÇİN TEKRAR DENEME (RETRY) MEKANİZMASI
    # ════════════════════════════════════════════════════════════════════════════
    max_retries = 3  # Proxy patlarsa en fazla kaç farklı proxy denensin?
    otp_sent_successfully = False

    for attempt in range(max_retries):
        print(f"\n[*] OTP gönderim denemesi {attempt + 1}/{max_retries}...")
        
        # 1. Her denemede yeni bir çalışan proxy ara
        working_proxy = find_working_proxy(max_workers=30)
        if working_proxy:
            auth_proxies = {"http": working_proxy, "https": working_proxy}
            print(f"[*] Proxy OTP isteğinde kullanılacak: {working_proxy}")
        else:
            auth_proxies = None
            print("[-] Çalışan proxy bulunamadı, bu deneme proxysiz yapılacak.")

        # 2. OTP Gönderimini dene
        try:
            r1 = s.post(
                f"{base_url}/api/auth/otp/send",
                json={"email": email, "fingerprint": fingerprint},
                proxies=auth_proxies,
                timeout=15  # Proxy yavaşsa 15 saniyede kesip yenisine geçsin
            )
            
            # Eğer Apixo başarılı döndüyse döngüden çık
            if r1.json().get("success"):
                print("[+] OTP başarıyla gönderildi!")
                otp_sent_successfully = True
                break
            else:
                print(f"[-] Apixo hata döndürdü (IP engeli veya rate-limit). Yanıt: {r1.text}")
                print("[*] Yeni bir proxy ile tekrar denenecek...")
                
        except Exception as e:
            print(f"[-] İstek hatası (Proxy ölmüş veya yavaş): {e}")
            print("[*] Yeni bir proxy ile tekrar denenecek...")
        
        # Döngü sırasındaki başarısız istekler için kısa bir bekleme
        time.sleep(1)

    # Eğer tüm denemeler başarısız olduysa ana sisteme hata dön
    if not otp_sent_successfully:
        return None, None, f"{max_retries} farklı proxy denemesine rağmen OTP kodu gönderilemedi."

    # ════════════════════════════════════════════════════════════════════════════
    #  SONRAKİ ADIMLAR (Mevcut IP üzerinden hızlıca devam eder)
    # ════════════════════════════════════════════════════════════════════════════

    # 3. OTP Bekle (Spamok üzerinden)
    print("[*] OTP kodu bekleniyor...")
    otp = temp.get_otp(email)
    if not otp:
        return None, None, "OTP timeout."
    print(f"[+] OTP kodu yakalandı: {otp}")

    # 4. OTP Doğrula
    r2 = s.post(f"{base_url}/api/auth/otp/verify", json={"email": email, "otp": otp})
    d2 = r2.json()
    if not d2.get("success"):
        return None, None, "OTP doğrulanamadı."
    temp_token = d2["tempToken"]

    # 5. CSRF Al
    r3 = s.get(f"{base_url}/api/auth/csrf")
    csrf_token = r3.json()["csrfToken"]

    # 6. Callback (Kayıt tamamlama)
    try:
        s.post(
            f"{base_url}/api/auth/callback/email-otp",
            headers={**dict(s.headers),
                     "Content-Type": "application/x-www-form-urlencoded",
                     "x-auth-return-redirect": "1"},
            data={
                "email": email, "token": temp_token,
                "callbackUrl": f"{base_url}/models/image",
                "redirect": "false", "csrfToken": csrf_token,
            },
            allow_redirects=False,
            timeout=15
        )
    except Exception as e:
        return None, None, f"Callback isteği sırasında hata: {e}"

    # 7. Session Al
    r5 = s.get(f"{base_url}/api/auth/session")
    return s, r5.json(), None

# ════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════════════════

MIME_TYPES = {
    "jpg":  "image/jpeg", "jpeg": "image/jpeg",
    "png":  "image/png",  "webp": "image/webp",
    "mp3":  "audio/mpeg", "wav":  "audio/wav", "m4a": "audio/mp4",
}

def upload_file(sess: requests.Session, file_path: str) -> str:
    file_name = os.path.basename(file_path)
    file_size = os.path.getsize(file_path)
    ext = file_name.rsplit(".", 1)[-1].lower()
    file_type = MIME_TYPES.get(ext, "application/octet-stream")
    name_hash = hashlib.md5(file_name.encode()).hexdigest()
    upload_name = f"{name_hash}-{int(time.time() * 1000)}.{ext}"

    r = sess.post("https://apixo.ai/api/upload-presigned-url", json={
        "fileName": upload_name, "fileType": file_type, "fileSize": file_size
    })
    if r.status_code != 200:
        raise Exception(f"Presigned URL alınamadı: {r.text}")
    presigned = r.json()

    with open(file_path, "rb") as f:
        file_data = f.read()

    r2 = requests.put(
        presigned["uploadUrl"], data=file_data,
        headers={
            "Content-Type":  presigned["contentType"],
            "Cache-Control": presigned["cacheControl"],
            "Origin":  "https://apixo.ai",
            "Referer": "https://apixo.ai/",
        },
        timeout=60
    )
    if r2.status_code != 200:
        raise Exception(f"R2 upload başarısız: {r2.text}")
    return presigned["publicUrl"]

# ════════════════════════════════════════════════════════════════════════════
#  BALANCE
# ════════════════════════════════════════════════════════════════════════════

def get_balance(sess: requests.Session) -> float:
    r_info = sess.get("https://apixo.ai/api/user/basic-info")
    info = r_info.json()
    r = sess.post(
        "https://apixo.ai/api/user/balance",
        json={"userId": info["id"], "email": info["email"]}
    )
    return float(r.json()["currentBalance"])

# ════════════════════════════════════════════════════════════════════════════
#  GENERATE (video/image)
# ════════════════════════════════════════════════════════════════════════════

def generate_video(sess, prompt, mode, resolution, duration, aspect_ratio,
                   enable_prompt_expansion, image_url=None, negative_prompt="",
                   seed="", audio_url=None, model="wan-2-5-video",
                   watermark=False):
    mode_cfg = MODEL_CONFIGS[model]["modes"][mode]
    params = dict(mode_cfg["default_params"])
    params.update({
        "mode": mode, "prompt": prompt, "resolution": resolution,
        "duration": duration, "aspect_ratio": aspect_ratio,
        "enable_prompt_expansion": enable_prompt_expansion,
        "negative_prompt": negative_prompt, "seed": seed,
        "watermark": watermark,
        "image_urls": [image_url] if image_url else [],
        "audio_urls": [audio_url] if audio_url else [],
    })
    cost = mode_cfg["cost_table"].get(resolution, {}).get(duration, 0.4)
    r = sess.post(
        f"https://apixo.ai/api/playground/models/{model}/generate",
        json={"model": model, "parameters": params, "estimatedCredits": cost}
    )
    j = r.json()
    if not j.get("success"):
        raise Exception(f"Üretim başlatılamadı: {r.text}")
    return j["taskId"]


def generate_image(sess, prompt, mode, resolution, num_images=1,
                   image_urls=None, model="wan-2-7-image",
                   negative_prompt="", seed="",
                   watermark=False, thinking_mode=True,
                   enable_sequential=False):
    mode_cfg = MODEL_CONFIGS[model]["modes"][mode]
    params = dict(mode_cfg["default_params"])
    params.update({
        "mode": mode, "prompt": prompt, "resolution": resolution,
        "num_images": num_images,
        "image_urls": image_urls if image_urls else [],
        "negative_prompt": negative_prompt, "seed": seed,
        "watermark": watermark, "thinking_mode": thinking_mode,
        "enable_sequential": enable_sequential,
    })
    cost = mode_cfg["estimated_credits"] * num_images
    r = sess.post(
        f"https://apixo.ai/api/playground/models/{model}/generate",
        json={"model": model, "parameters": params, "estimatedCredits": cost}
    )
    j = r.json()
    if not j.get("success"):
        raise Exception(f"Üretim başlatılamadı: {r.text}")
    return j["taskId"]

# ════════════════════════════════════════════════════════════════════════════
#  FLASK APP
# ════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.secret_key = secrets.token_hex(32)
APP_PASSWORD = "123"

# Flask session başına apixo session (process-memory)
APIXO_STORE: dict[str, dict] = {}

# ── RAM Storage ────────────────────────────────────────────────────────────
import uuid as _uuid

jobs_store = {}
jobs_lock = threading.Lock()

prompts_store = {}
prompts_lock = threading.Lock()

gallery_store = []
gallery_lock = threading.Lock()


def _run_job(job_id, data, apixo_sess):
    """Background worker: generate + poll, update jobs_store."""
    def log(msg):
        with jobs_lock:
            if job_id in jobs_store:
                jobs_store[job_id]['logs'].append(msg)

    def update(upd):
        with jobs_lock:
            if job_id in jobs_store:
                jobs_store[job_id].update(upd)

    try:
        update({'status': 'generating'})
        log('Üretim başlatılıyor...')
        task_type = data.get('task_type')
        sess = apixo_sess

        if task_type == 'video':
            model = data.get('model', 'wan-2-5-video')
            task_id = generate_video(
                sess,
                prompt=data.get('prompt', ''),
                mode=data.get('mode', 'text-to-video'),
                resolution=data.get('resolution', '480p'),
                duration=int(data.get('duration', 5)),
                aspect_ratio=data.get('aspect_ratio', '16:9'),
                enable_prompt_expansion=bool(data.get('enable_prompt_expansion', False)),
                image_url=data.get('image_url') or None,
                negative_prompt=data.get('negative_prompt', ''),
                seed=data.get('seed', ''),
                audio_url=data.get('audio_url') or None,
                model=model,
                watermark=bool(data.get('watermark', False)),
            )
        elif task_type == 'image':
            model = data.get('model', 'wan-2-7-image')
            task_id = generate_image(
                sess,
                prompt=data.get('prompt', ''),
                mode=data.get('mode', 'omni-image'),
                resolution=data.get('resolution', '2k'),
                num_images=int(data.get('num_images', 1)),
                image_urls=data.get('image_urls') or [],
                model=model,
                negative_prompt=data.get('negative_prompt', ''),
                seed=data.get('seed', ''),
                watermark=bool(data.get('watermark', False)),
                thinking_mode=data.get('thinking_mode', True),
                enable_sequential=bool(data.get('enable_sequential', False)),
            )
        else:
            update({'status': 'error'})
            log('Geçersiz task_type')
            return

        log(f'Task ID: {task_id}')
        update({'status': 'polling', 'apixo_task_id': task_id})

        for i in range(150):
            time.sleep(4)
            try:
                r = sess.get(
                    f'https://apixo.ai/api/playground/models/{model}/status',
                    params={'taskId': task_id}
                )
                d = r.json()
                st = d.get('state')
                elapsed = (i + 1) * 4
                log(f'State: {st} ({elapsed}s)')
                if st == 'success':
                    outputs = d.get('resultUrls') or ([d.get('resultUrl')] if d.get('resultUrl') else [])
                    outputs = [u for u in outputs if u]
                    update({'status': 'done', 'result_urls': outputs})
                    log(f'Tamamlandı! {len(outputs)} çıktı')
                    return
                elif st == 'failed':
                    update({'status': 'error'})
                    log(f'Başarısız: {d.get("error", "bilinmeyen")}')
                    return
            except Exception as e:
                log(f'Poll hatası: {e}')

        update({'status': 'error'})
        log('Zaman aşımı (10 dk)')
    except Exception as e:
        update({'status': 'error'})
        log(f'Hata: {e}')


def get_sid():
    if 'sid' not in session:
        session['sid'] = secrets.token_hex(16)
    return session['sid']

def get_apixo():
    return APIXO_STORE.get(get_sid())

def set_apixo(s, u):
    APIXO_STORE[get_sid()] = {"session": s, "user": u}

def clear_apixo():
    APIXO_STORE.pop(get_sid(), None)


def require_app_login(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not session.get('logged_in'):
            if request.path.startswith('/api/'):
                return jsonify({"error": "Yetkisiz"}), 401
            return redirect(url_for('login'))
        return f(*a, **kw)
    return wrapper

def require_apixo(f):
    @wraps(f)
    def wrapper(*a, **kw):
        if not get_apixo():
            return jsonify({"error": "Önce APIXO hesabı oluşturmalısın."}), 400
        return f(*a, **kw)
    return wrapper


# ─── Login sayfası (inline) ─────────────────────────────────────────────────

LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="tr"><head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APIXO STUDIO · Giriş</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Space+Grotesk:wght@600;700&display=swap" rel="stylesheet">
<style>
*{box-sizing:border-box;margin:0;padding:0}
html,body{height:100%}
body{
  font-family:'Inter',sans-serif;background:#06070d;color:#e8eaf0;
  display:flex;align-items:center;justify-content:center;
  min-height:100vh;overflow:hidden;position:relative;
}
body::before{
  content:'';position:fixed;inset:0;pointer-events:none;
  background:
    radial-gradient(ellipse at 20% 10%, rgba(124,92,255,.30) 0%, transparent 50%),
    radial-gradient(ellipse at 80% 90%, rgba(0,212,255,.20) 0%, transparent 50%),
    radial-gradient(ellipse at 50% 50%, rgba(255,92,242,.08) 0%, transparent 70%);
}
body::after{
  content:'';position:fixed;inset:0;pointer-events:none;opacity:.04;
  background-image:linear-gradient(rgba(255,255,255,.5) 1px,transparent 1px),linear-gradient(90deg,rgba(255,255,255,.5) 1px,transparent 1px);
  background-size:42px 42px;
}
.card{
  position:relative;z-index:1;width:min(420px,92vw);padding:42px 36px;
  background:rgba(20,22,35,.55);backdrop-filter:blur(20px);
  border:1px solid rgba(120,130,200,.18);border-radius:24px;
  box-shadow:0 30px 80px rgba(0,0,0,.5),inset 0 0 0 1px rgba(255,255,255,.03);
  animation:rise .6s cubic-bezier(.2,.8,.2,1);
}
@keyframes rise{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:none}}
.brand{
  font-family:'Space Grotesk',sans-serif;font-weight:700;font-size:28px;letter-spacing:.5px;
  background:linear-gradient(135deg,#7c5cff 0%,#00d4ff 60%,#ff5cf2 100%);
  -webkit-background-clip:text;background-clip:text;color:transparent;
  margin-bottom:6px;
}
.sub{color:#8a8fa3;font-size:14px;margin-bottom:28px}
label{display:block;font-size:12px;letter-spacing:.6px;color:#8a8fa3;text-transform:uppercase;margin-bottom:8px}
input[type=password]{
  width:100%;padding:14px 16px;font-size:15px;font-family:inherit;
  background:rgba(10,11,18,.7);border:1px solid rgba(120,130,200,.2);
  border-radius:12px;color:#e8eaf0;outline:none;transition:.2s;
}
input[type=password]:focus{border-color:#7c5cff;box-shadow:0 0 0 4px rgba(124,92,255,.15)}
button{
  width:100%;margin-top:18px;padding:14px;font-size:15px;font-weight:600;font-family:inherit;
  background:linear-gradient(135deg,#7c5cff 0%,#00d4ff 100%);
  color:#fff;border:none;border-radius:12px;cursor:pointer;letter-spacing:.3px;
  transition:.2s;
}
button:hover{transform:translateY(-1px);box-shadow:0 10px 30px rgba(124,92,255,.4)}
.error{
  margin-top:14px;padding:10px 14px;background:rgba(255,85,119,.12);
  border:1px solid rgba(255,85,119,.3);border-radius:10px;
  color:#ff8aa3;font-size:13px;
}
.dot{display:inline-block;width:6px;height:6px;border-radius:50%;background:#7c5cff;margin-right:6px;animation:pulse 1.8s infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.3}}
</style></head><body>
<div class="card">
  <div class="brand">APIXO STUDIO</div>
  <div class="sub"><span class="dot"></span>Cinematic AI Studio · Erişim için şifre gerekli</div>
  <form method="POST" action="/login">
    <label for="pwd">Şifre</label>
    <input id="pwd" type="password" name="password" autofocus required placeholder="••••••">
    <button type="submit">Giriş Yap</button>
    {% if error %}<div class="error">{{ error }}</div>{% endif %}
  </form>
</div>
</body></html>"""


@app.route('/login', methods=['GET', 'POST'])
def login():
    if session.get('logged_in') and request.method == 'GET':
        return redirect(url_for('index'))
    error = None
    if request.method == 'POST':
        pwd = request.form.get('password', '')
        if pwd == APP_PASSWORD:
            session['logged_in'] = True
            return redirect(url_for('index'))
        error = "Hatalı şifre."
    return render_template_string(LOGIN_HTML, error=error)


@app.route('/logout')
def logout():
    clear_apixo()
    session.clear()
    return redirect(url_for('login'))


@app.route('/')
@require_app_login
def index():
    return render_template('index.html')


# ─── API ────────────────────────────────────────────────────────────────────

@app.route('/api/models')
@require_app_login
def api_models():
    return jsonify(MODEL_CONFIGS)


@app.route('/api/whoami')
@require_app_login
def api_whoami():
    ax = get_apixo()
    if not ax:
        return jsonify({"logged_in": False})
    u = ax["user"].get("user", {})
    return jsonify({
        "logged_in": True,
        "email": u.get("email"),
        "id": u.get("id"),
    })


@app.route('/api/apixo-login', methods=['POST'])
@require_app_login
def api_apixo_login():
    try:
        s, u, err = apixo_auto_login()
        if not s:
            return jsonify({"error": err or "Bilinmeyen hata"}), 500
        set_apixo(s, u)
        return jsonify({
            "email": u["user"]["email"],
            "id": u["user"]["id"],
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/balance')
@require_app_login
@require_apixo
def api_balance():
    try:
        bal = get_balance(get_apixo()["session"])
        return jsonify({"balance": bal})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/upload', methods=['POST'])
@require_app_login
@require_apixo
def api_upload():
    f = request.files.get('file')
    if not f:
        return jsonify({"error": "Dosya bulunamadı."}), 400
    suffix = '_' + os.path.basename(f.filename or 'file')
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    f.save(tmp.name)
    tmp.close()
    try:
        url = upload_file(get_apixo()["session"], tmp.name)
        return jsonify({"url": url, "name": f.filename})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        try: os.unlink(tmp.name)
        except: pass


@app.route('/api/generate', methods=['POST'])
@require_app_login
@require_apixo
def api_generate():
    data = request.get_json(force=True)
    task_type = data.get('task_type')
    sess = get_apixo()["session"]
    try:
        if task_type == 'video':
            model = data.get('model', 'wan-2-5-video')
            task_id = generate_video(
                sess,
                prompt=data.get('prompt', ''),
                mode=data.get('mode', 'text-to-video'),
                resolution=data.get('resolution', '480p'),
                duration=int(data.get('duration', 5)),
                aspect_ratio=data.get('aspect_ratio', '16:9'),
                enable_prompt_expansion=bool(data.get('enable_prompt_expansion', False)),
                image_url=data.get('image_url') or None,
                negative_prompt=data.get('negative_prompt', ''),
                seed=data.get('seed', ''),
                audio_url=data.get('audio_url') or None,
                model=model,
                watermark=bool(data.get('watermark', False)),
            )
            return jsonify({"task_id": task_id, "model": model})

        elif task_type == 'image':
            model = data.get('model', 'wan-2-7-image')
            task_id = generate_image(
                sess,
                prompt=data.get('prompt', ''),
                mode=data.get('mode', 'omni-image'),
                resolution=data.get('resolution', '2k'),
                num_images=int(data.get('num_images', 1)),
                image_urls=data.get('image_urls') or [],
                model=model,
                negative_prompt=data.get('negative_prompt', ''),
                seed=data.get('seed', ''),
                watermark=bool(data.get('watermark', False)),
                thinking_mode=data.get('thinking_mode', True),
                enable_sequential=bool(data.get('enable_sequential', False)),
            )
            return jsonify({"task_id": task_id, "model": model})

        return jsonify({"error": "Geçersiz task_type"}), 400
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/task-status')
@require_app_login
@require_apixo
def api_task_status():
    task_id = request.args.get('task_id')
    model   = request.args.get('model')
    if not task_id or not model:
        return jsonify({"error": "task_id ve model zorunlu."}), 400
    sess = get_apixo()["session"]
    try:
        r = sess.get(
            f"https://apixo.ai/api/playground/models/{model}/status",
            params={"taskId": task_id}
        )
        if r.status_code != 200:
            return jsonify({"error": f"Status HTTP {r.status_code}"}), 500
        d = r.json()
        state = d.get('state')
        outputs = []
        if state == 'success':
            outputs = d.get('resultUrls') or ([d.get('resultUrl')] if d.get('resultUrl') else [])
            outputs = [u for u in outputs if u]
        return jsonify({
            "state": state,
            "outputs": outputs,
            "error": d.get('error'),
            "cost_time": d.get('costTime', 0),
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ── JOB SYSTEM ─────────────────────────────────────────────────────────────

@app.route('/api/start-job', methods=['POST'])
@require_app_login
@require_apixo
def api_start_job():
    data = request.get_json(force=True)
    job_id = str(_uuid.uuid4())
    with jobs_lock:
        jobs_store[job_id] = {
            'status': 'starting',
            'logs': [],
            'result_urls': [],
            'task_type': data.get('task_type', 'video'),
            'model': data.get('model', ''),
            'mode': data.get('mode', ''),
            'prompt': data.get('prompt', ''),
            'created_at': time.time(),
        }
    sess = get_apixo()["session"]
    t = threading.Thread(target=_run_job, args=(job_id, data, sess), daemon=True)
    t.start()
    return jsonify({"job_id": job_id})


@app.route('/api/job-status/<job_id>')
@require_app_login
def api_job_status(job_id):
    with jobs_lock:
        if job_id in jobs_store:
            return jsonify(jobs_store[job_id])
    return jsonify({"error": "Job bulunamadı"}), 404


@app.route('/api/jobs')
@require_app_login
def api_jobs():
    with jobs_lock:
        return jsonify(dict(jobs_store))


@app.route('/api/delete-job/<job_id>', methods=['DELETE'])
@require_app_login
def api_delete_job(job_id):
    with jobs_lock:
        if job_id in jobs_store:
            del jobs_store[job_id]
            return jsonify({"success": True})
    return jsonify({"error": "Job bulunamadı"}), 404


# ── PROMPT LIBRARY ─────────────────────────────────────────────────────────

@app.route('/api/prompts', methods=['GET'])
@require_app_login
def api_get_prompts():
    with prompts_lock:
        result = sorted(prompts_store.values(), key=lambda p: p['timestamp'], reverse=True)
    return jsonify(result)


@app.route('/api/prompts', methods=['POST'])
@require_app_login
def api_save_prompt():
    data = request.get_json(force=True)
    text = (data.get('text') or '').strip()
    if not text:
        return jsonify({"error": "Prompt boş olamaz"}), 400
    pid = str(_uuid.uuid4())
    entry = {"id": pid, "text": text, "timestamp": int(time.time() * 1000)}
    with prompts_lock:
        prompts_store[pid] = entry
    return jsonify(entry)


@app.route('/api/prompts/<pid>', methods=['DELETE'])
@require_app_login
def api_delete_prompt(pid):
    with prompts_lock:
        if pid in prompts_store:
            del prompts_store[pid]
            return jsonify({"success": True})
    return jsonify({"error": "Prompt bulunamadı"}), 404


# ── GALLERY ────────────────────────────────────────────────────────────────

@app.route('/api/gallery', methods=['GET'])
@require_app_login
def api_get_gallery():
    with gallery_lock:
        return jsonify(list(gallery_store))


@app.route('/api/gallery', methods=['POST'])
@require_app_login
def api_add_gallery():
    data = request.get_json(force=True)
    if not data:
        return jsonify({"error": "Veri eksik"}), 400
    with gallery_lock:
        gallery_store[:] = [i for i in gallery_store if i.get('id') != data.get('id')]
        gallery_store.insert(0, data)
        if len(gallery_store) > 200:
            gallery_store[:] = gallery_store[:200]
    return jsonify({"success": True})


@app.route('/api/gallery/<item_id>', methods=['DELETE'])
@require_app_login
def api_delete_gallery(item_id):
    with gallery_lock:
        before = len(gallery_store)
        gallery_store[:] = [i for i in gallery_store if i.get('id') != item_id]
        if len(gallery_store) < before:
            return jsonify({"success": True})
    return jsonify({"error": "Öğe bulunamadı"}), 404


@app.route('/api/gallery/clear', methods=['DELETE'])
@require_app_login
def api_clear_gallery():
    with gallery_lock:
        gallery_store.clear()
    return jsonify({"success": True})


# ── MEDIA PROXY ────────────────────────────────────────────────────────────

@app.route('/api/proxy-media')
@require_app_login
def api_proxy_media():
    url = request.args.get('url', '')
    dl = request.args.get('dl', '0') == '1'
    if not url:
        return jsonify({"error": "URL gerekli"}), 400
    range_header = request.headers.get('Range', None)
    req_headers = {}
    if range_header:
        req_headers['Range'] = range_header
    try:
        from flask import Response
        resp = requests.get(url, headers=req_headers, stream=True, timeout=60)
        response_headers = {
            'Content-Type': resp.headers.get('content-type', 'application/octet-stream'),
            'Accept-Ranges': 'bytes',
        }
        if 'Content-Length' in resp.headers:
            response_headers['Content-Length'] = resp.headers['Content-Length']
        if 'Content-Range' in resp.headers:
            response_headers['Content-Range'] = resp.headers['Content-Range']
        if dl:
            response_headers['Content-Disposition'] = 'attachment; filename="download"'

        def gen():
            for chunk in resp.iter_content(chunk_size=65536):
                yield chunk

        return Response(gen(), status=resp.status_code, headers=response_headers)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    os.makedirs("templates", exist_ok=True)
    app.run(host="0.0.0.0", port=5000, debug=True)
