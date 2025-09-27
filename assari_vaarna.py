#!/usr/bin/env python3
import os, io, json, time, wave, subprocess, tempfile, sys
import sounddevice as sd
import webrtcvad
from vosk import Model, KaldiRecognizer
from openai import OpenAI

# ======= MUOKATTAVAT POLUT/ARVOT =======
VOSK_MODEL_DIR = os.path.expanduser("~/vosk/vosk-model-small-fi")  # vosk-suomimallin kansio
PIPER_BIN      = "/usr/bin/piper"
PIPER_VOICE    = os.path.expanduser("~/piper/fi_FI-aginum_low.onnx")      # päivitä äänesi mukaan
PIPER_CFG      = os.path.expanduser("~/piper/fi_FI-aginum_low.onnx.json") # päivitä äänesi mukaan

SR           = 16000     # 16 kHz mono
WAKEWORD     = "vaarna"  # herätesana
SIGNOFF_PHRASE = "ei miul muuta"  # <-- uusi lopputervehdys
MAX_LISTEN_S = 30        # max kesto kysymykselle
SIL_END_MS   = 900       # hiljaisuuden pituus lopetukseen
VAD_AGGR     = 2         # webrtcvad aggressiivisuus 0..3

# OpenAI
MODEL_TEXT = "gpt-5"     # yleismalli tekstiin
STT_MODEL  = "whisper-1" # puhe->teksti

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# ======= apurit =======
def pcm16_to_wav_bytes(pcm_bytes, samplerate=SR):
    bio = io.BytesIO()
    with wave.open(bio, 'wb') as wf:
        wf.setnchannels(1); wf.setsampwidth(2); wf.setframerate(samplerate)
        wf.writeframes(pcm_bytes)
    return bio.getvalue()

def stt_whisper_from_wav_bytes(wav_bytes):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
        tmp.write(wav_bytes); tmp.flush()
        with open(tmp.name, "rb") as f:
            tr = client.audio.transcriptions.create(
                model=STT_MODEL, file=f, language="fi"
            )
    return tr.text.strip()

def ask_openai_finnish(prompt_text):
    sys_prompt = ("Olet suomenkielinen asiantuntija-avustaja. "
                  "Vastaa ytimekkäästi ja selkeästi suomeksi.")
    r = client.responses.create(
        model=MODEL_TEXT,
        input=[{"role":"system","content":sys_prompt},
               {"role":"user","content":prompt_text}]
    )
    return r.output_text.strip()

def piper_say(text):
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpwav:
        wav_path = tmpwav.name
    cmd = [PIPER_BIN, "--model", PIPER_VOICE, "--config", PIPER_CFG,
           "--output_file", wav_path]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    proc.communicate(input=text.encode("utf-8"))
    proc.wait()
    subprocess.run(["paplay", wav_path], check=False)
    try: os.remove(wav_path)
    except: pass

# ======= kuuntelu =======
def listen_until_silence():
    """Nauhoita kysymys VAD:lla kunnes hiljaisuutta SIL_END_MS."""
    vad = webrtcvad.Vad(VAD_AGGR)
    block_ms = 30
    block_len = int(SR * block_ms / 1000.0)
    frames = []
    silence_ms = 0
    start_t = time.time()
    with sd.RawInputStream(samplerate=SR, channels=1, dtype='int16') as stream:
        while True:
            data, _ = stream.read(block_len)
            b = bytes(data)
            frames.append(b)
            if vad.is_speech(b, SR):
                silence_ms = 0
            else:
                silence_ms += block_ms
            if silence_ms >= SIL_END_MS: break
            if (time.time() - start_t) >= MAX_LISTEN_S: break
    return b"".join(frames)

def wait_for_wakeword():
    """Kuuntele jatkuvasti Voskilla ja palaa kun WAKEWORD havaitaan."""
    model = Model(VOSK_MODEL_DIR)
    rec = KaldiRecognizer(model, SR)
    rec.SetWords(False)
    block_len = int(SR * 0.5)  # 0.5 s palat
    print("Kuuntelen herätesanaa:", WAKEWORD)
    with sd.RawInputStream(samplerate=SR, channels=1, dtype='int16') as stream:
        while True:
            data, _ = stream.read(block_len)
            if rec.AcceptWaveform(bytes(data)):
                j = json.loads(rec.Result())
                txt = j.get("text","").lower().strip()
            else:
                j = json.loads(rec.PartialResult())
                txt = j.get("partial","").lower().strip()
            if not txt:
                continue
            tokens = txt.replace(",", " ").split()
            if WAKEWORD in tokens or WAKEWORD in txt:
                print("Herätesana havaittu:", txt)
                return

def main_loop():
    while True:
        wait_for_wakeword()
        # Herätteen vahvistus
        piper_say("kyl se siit")
        print("Nauhoitus käynnissä...")
        pcm = listen_until_silence()
        wav = pcm16_to_wav_bytes(pcm)
        print("Tunnistetaan kysymys…")
        try:
            user_text = stt_whisper_from_wav_bytes(wav)
        except Exception as e:
            print("STT-virhe:", e)
            piper_say("En saanut selvää. Yritä uudelleen.")
            continue
        if not user_text:
            print("Tyhjä kysymys.")
            piper_say("En kuullut kysymystä.")
            continue
        print("Sinä:", user_text)

        print("Haen vastauksen…")
        try:
            answer = ask_openai_finnish(user_text)
        except Exception as e:
            print("OpenAI-virhe:", e)
            piper_say("Palveluvirhe.")
            continue

        print("Vastaus:", answer)
        piper_say(answer)

        # --- uusi lopputervehdys ---
        # pieni tauko tekee rytmistä luonnollisemman
        time.sleep(0.2)
        piper_say(SIGNOFF_PHRASE)

if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("Puuttuu OPENAI_API_KEY.")
        sys.exit(1)
    for path in (VOSK_MODEL_DIR, PIPER_VOICE, PIPER_CFG):
        if not os.path.exists(path):
            print("Tarkista polku:", path)
    main_loop()
