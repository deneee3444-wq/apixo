import random
import time
import requests
import string
import re
import hashlib
import os


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
        print(f"[*] OTP bekleniyor ({timeout}s timeout)...")

        for i in range(timeout):
            r = requests.get(f'https://api.spamok.com/v2/EmailBox/{address}')
            data = r.json()

            for mail in data.get('mails', []):
                subject      = mail.get('subject', '')
                from_display = mail.get('fromDisplay', '')

                if 'APIXO' in from_display or 'verification' in subject.lower():
                    mail_id = mail['id']
                    email_r = requests.get(f'https://api.spamok.com/v2/Email/{address}/{mail_id}')
                    body    = email_r.json()

                    plain = body.get('messagePlain', '')
                    match = re.search(r'\b(\d{6})\b', plain)
                    if match:
                        return match.group(1)

                    html  = body.get('messageHtml', '')
                    match = re.search(r'letter-spacing:8px[^>]*>(\d{6})<', html)
                    if match:
                        return match.group(1)

            time.sleep(2)
            print(f"    {(i+1)*2}s...", end='\r')

        return None


# ════════════════════════════════════════════════════════════════════════════
#  AUTH
# ════════════════════════════════════════════════════════════════════════════

def apixo_auto_login() -> tuple[requests.Session | None, dict | None]:
    temp        = ApixoTemp()
    email       = temp.random_email()
    fingerprint = temp.generate_fingerprint(email)
    base_url    = "https://apixo.ai"

    session = requests.Session()
    session.headers.update({
        "Origin":             base_url,
        "Referer":            f"{base_url}/models/image",
        "User-Agent":         "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/149.0.0.0 Safari/537.36",
        "sec-ch-ua":          '"Google Chrome";v="149", "Chromium";v="149", "Not)A;Brand";v="24"',
        "sec-ch-ua-mobile":   "?0",
        "sec-ch-ua-platform": '"Windows"',
        "sec-fetch-dest":     "empty",
        "sec-fetch-mode":     "cors",
        "sec-fetch-site":     "same-origin",
    })

    print(f"[*] Email      : {email}")
    print(f"[*] Fingerprint: {fingerprint}")

    print("\n[1/5] OTP gönderiliyor...")
    r1 = session.post(f"{base_url}/api/auth/otp/send",
                      json={"email": email, "fingerprint": fingerprint})
    print(f"    Status: {r1.status_code} | {r1.json()}")
    assert r1.json().get("success"), "OTP gönderilemedi!"

    otp = temp.get_otp(email)
    if not otp:
        print("[-] OTP alınamadı (timeout)!")
        return None, None
    print(f"\n[+] OTP alındı : {otp}")

    print("\n[2/5] OTP doğrulanıyor...")
    r2 = session.post(f"{base_url}/api/auth/otp/verify",
                      json={"email": email, "otp": otp})
    print(f"    Status: {r2.status_code} | {r2.json()}")
    data2 = r2.json()
    assert data2.get("success"), "OTP doğrulanamadı!"
    temp_token = data2["tempToken"]

    print("\n[3/5] CSRF token alınıyor...")
    r3         = session.get(f"{base_url}/api/auth/csrf")
    csrf_token = r3.json()["csrfToken"]
    print(f"    csrfToken: {csrf_token}")

    print("\n[4/5] Session başlatılıyor...")
    r4 = session.post(
        f"{base_url}/api/auth/callback/email-otp",
        headers={**dict(session.headers),
                 "Content-Type":           "application/x-www-form-urlencoded",
                 "x-auth-return-redirect": "1"},
        data={
            "email":       email,
            "token":       temp_token,
            "callbackUrl": f"{base_url}/models/image",
            "redirect":    "false",
            "csrfToken":   csrf_token,
        },
        allow_redirects=False
    )
    print(f"    Status: {r4.status_code} | {r4.text}")

    print("\n[5/5] Session doğrulanıyor...")
    r5           = session.get(f"{base_url}/api/auth/session")
    session_data = r5.json()

    print(f"\n✅ Giriş başarılı!")
    print(f"   Email : {session_data['user']['email']}")
    print(f"   ID    : {session_data['user']['id']}")

    return session, session_data


# ════════════════════════════════════════════════════════════════════════════
#  UPLOAD
# ════════════════════════════════════════════════════════════════════════════

MIME_TYPES = {
    "jpg":  "image/jpeg",
    "jpeg": "image/jpeg",
    "png":  "image/png",
    "webp": "image/webp",
    "mp3":  "audio/mpeg",
    "wav":  "audio/wav",
    "m4a":  "audio/mp4",
}

def upload_file(session: requests.Session, file_path: str) -> str:
    """Görsel veya ses dosyasını R2'ye yükler, public URL döndürür."""
    file_name   = os.path.basename(file_path)
    file_size   = os.path.getsize(file_path)
    ext         = file_name.rsplit(".", 1)[-1].lower()
    file_type   = MIME_TYPES.get(ext, "application/octet-stream")
    name_hash   = hashlib.md5(file_name.encode()).hexdigest()
    upload_name = f"{name_hash}-{int(time.time() * 1000)}.{ext}"

    print(f"\n[*] Presigned URL alınıyor: {upload_name} ({file_type})")
    r = session.post("https://apixo.ai/api/upload-presigned-url", json={
        "fileName": upload_name,
        "fileType": file_type,
        "fileSize": file_size
    })
    print(f"    Status: {r.status_code}")
    assert r.status_code == 200, f"Presigned URL alınamadı: {r.text}"
    presigned = r.json()

    with open(file_path, "rb") as f:
        file_data = f.read()

    print(f"[*] R2'ye yükleniyor... ({len(file_data)/1024:.1f} KB)")
    r2 = requests.put(
        presigned["uploadUrl"],
        data=file_data,
        headers={
            "Content-Type":  presigned["contentType"],
            "Cache-Control": presigned["cacheControl"],
            "Origin":        "https://apixo.ai",
            "Referer":       "https://apixo.ai/",
        },
        timeout=60
    )
    print(f"[+] R2 upload status: {r2.status_code}")
    assert r2.status_code == 200, f"R2 upload başarısız: {r2.text}"
    print(f"[+] Public URL: {presigned['publicUrl']}")
    return presigned["publicUrl"]


# ════════════════════════════════════════════════════════════════════════════
#  GENERATE
# ════════════════════════════════════════════════════════════════════════════

def generate_video(
    session:                  requests.Session,
    prompt:                   str,
    mode:                     str,
    resolution:               str,
    duration:                 int,
    aspect_ratio:             str,
    enable_prompt_expansion:  bool,
    image_url:                str | None = None,
    negative_prompt:          str        = "",
    seed:                     str        = "",
    audio_url:                str | None = None,
    model:                    str        = "wan-2-5-video"
) -> str:

    params = {
        "request_type":            "async",
        "mode":                    mode,
        "prompt":                  prompt,
        "resolution":              resolution,
        "duration":                duration,
        "aspect_ratio":            aspect_ratio,
        "watermark":               False,
        "enable_prompt_expansion": enable_prompt_expansion,
    }

    # image-to-video için zorunlu
    if image_url:
        params["image_urls"] = [image_url]

    # isteğe bağlılar — sadece dolu ise ekle
    if negative_prompt:
        params["negative_prompt"] = negative_prompt
    if seed:
        params["seed"] = seed
    if audio_url:
        params["audio_urls"] = [audio_url]

    print(f"\n[*] Video üretimi başlatılıyor...")
    print(f"    Model      : {model}")
    print(f"    Mod        : {mode}")
    print(f"    Prompt     : {prompt}")
    print(f"    Çözünürlük : {resolution}")
    print(f"    Süre       : {duration}s")
    print(f"    Oran       : {aspect_ratio}")
    if negative_prompt: print(f"    Neg.Prompt : {negative_prompt}")
    if seed:            print(f"    Seed       : {seed}")
    if audio_url:       print(f"    Ses        : {audio_url}")

    r = session.post(
        f"https://apixo.ai/api/playground/models/{model}/generate",
        json={"model": model, "parameters": params, "estimatedCredits": 0.4}
    )
    print(f"    Status: {r.status_code} | {r.json()}")
    assert r.json().get("success"), f"Üretim başlatılamadı: {r.text}"
    return r.json()["taskId"]


# ════════════════════════════════════════════════════════════════════════════
#  POLL
# ════════════════════════════════════════════════════════════════════════════

def poll_task(session: requests.Session, task_id: str,
              model="wan-2-5-video", timeout=300) -> list:

    print(f"\n[*] Task izleniyor: {task_id}")
    url = f"https://apixo.ai/api/playground/models/{model}/status"

    for elapsed in range(0, timeout, 5):
        r = session.get(url, params={"taskId": task_id})

        if r.status_code != 200:
            print(f"    [{elapsed}s] Sorgu hatası: {r.status_code}")
            time.sleep(5)
            continue

        data  = r.json()
        state = data.get("state")
        print(f"    [{elapsed}s] State: {state}")

        if state == "success":
            outputs = data.get("resultUrls") or ([data["resultUrl"]] if data.get("resultUrl") else [])
            outputs = [u for u in outputs if u]
            print(f"\n✅ Tamamlandı! (süre: {data.get('costTime', 0)/1000:.1f}s)")
            for i, u in enumerate(outputs):
                print(f"   [{i+1}] {u}")
            return outputs

        elif state == "failed":
            print(f"\n[-] Başarısız! Hata: {data.get('error')}")
            return []

        time.sleep(5)

    print("[-] Timeout!")
    return []


# ════════════════════════════════════════════════════════════════════════════
#  TAM AKIŞ
# ════════════════════════════════════════════════════════════════════════════

def apixo_generate(
    session:                  requests.Session,
    prompt:                   str,
    mode:                     str        = "text-to-video",
    resolution:               str        = "480p",
    duration:                 int        = 5,
    aspect_ratio:             str        = "16:9",
    enable_prompt_expansion:  bool       = False,
    image_path:               str | None = None,
    negative_prompt:          str        = "",
    seed:                     str        = "",
    audio_path:               str | None = None,
    model:                    str        = "wan-2-5-video"
) -> list:

    image_url = None
    if mode == "image-to-video":
        assert image_path, "image-to-video modu için image_path zorunlu!"
        image_url = upload_file(session, image_path)

    audio_url = None
    if audio_path:
        audio_url = upload_file(session, audio_path)

    task_id = generate_video(
        session, prompt, mode, resolution, duration, aspect_ratio,
        enable_prompt_expansion, image_url, negative_prompt, seed, audio_url, model
    )

    return poll_task(session, task_id, model)


# ════════════════════════════════════════════════════════════════════════════
#  KULLANIM — parametreleri buradan düzenle
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── Mod seçimi ─────────────────────────────────────────────────────────
    MOD        = "image-to-video"   # "image-to-video"  |  "text-to-video"

    # ── Dosya yolları ──────────────────────────────────────────────────────
    GORSEL     = "test.png"         # image-to-video için zorunlu | text-to-video için kullanılmaz
    SES        = None               # .mp3 / .wav / .m4a — istemiyorsan None (payload'a girmez)

    # ── Video ayarları ─────────────────────────────────────────────────────
    PROMPT     = "cinematic slow zoom, soft light"
    NEG_PROMPT = ""                 # boş bırak = payload'a girmez
    COZUNURLUK = "480p"             # "480p"  |  "720p"  |  "1080p"
    SURE       = 5                  # 5  |  10
    ORAN       = "16:9"             # "16:9"  |  "9:16"  |  "1:1"  |  "4:3"  |  "3:4"
    SEED       = ""                 # boş bırak = payload'a girmez | sabit sonuç için sayı gir
    PROMPT_GEN = False              # True = prompt otomatik genişletilsin

    # ── Giriş yap ve üret ──────────────────────────────────────────────────
    session, user = apixo_auto_login()

    if session:
        videos = apixo_generate(
            session                 = session,
            prompt                  = PROMPT,
            mode                    = MOD,
            resolution              = COZUNURLUK,
            duration                = SURE,
            aspect_ratio            = ORAN,
            enable_prompt_expansion = PROMPT_GEN,
            image_path              = GORSEL if MOD == "image-to-video" else None,
            negative_prompt         = NEG_PROMPT,
            seed                    = SEED,
            audio_path              = SES,
        )

        print(f"\n🎬 Toplam {len(videos)} video üretildi.")
