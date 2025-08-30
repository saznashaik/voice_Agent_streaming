# app.py
# Render-ready: Flask + Flask-SocketIO + outbound WS to AssemblyAI and Murf + Gemini LLM (Elsa persona)
# - WebSockets served on same port via eventlet worker under Gunicorn
# - No server audio devices required on Render; enable LOCAL_AUDIO=true locally to use mic/speakers

import os
import time
import json
import base64
import wave
import threading
import asyncio
import random
from datetime import datetime, timezone
from urllib.parse import urlencode
from typing import Optional

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template_string, jsonify, request
from flask_socketio import SocketIO, emit

# External clients
import requests
import websocket                 # websocket-client (AssemblyAI client)
import websockets                # websockets (Murf WS client)

# Optional local audio (guarded)
LOCAL_AUDIO = os.getenv("LOCAL_AUDIO", "false").lower() == "true"
if LOCAL_AUDIO:
    try:
        import pyaudio
        PYAUDIO_AVAILABLE = True
    except Exception:
        PYAUDIO_AVAILABLE = False
else:
    PYAUDIO_AVAILABLE = False

import orjson
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from tavily import TavilyClient

# ------------------- CONFIG -------------------
FLASK_HOST = "0.0.0.0"
# Render injects PORT; Flask-SocketIO under Gunicorn will bind to it
PORT = int(os.getenv("PORT", "5000"))

SAMPLE_RATE_STT = 16000
FRAMES_PER_BUFFER_STT = 800   # ~50ms
CHANNELS_STT = 1
FORMAT_STT = 8  # pyaudio.paInt16 placeholder value; not used unless LOCAL_AUDIO

SAMPLE_RATE_TTS = 44100
CHANNELS_TTS = 1

ASSEMBLY_WS_BASE = "wss://streaming.assemblyai.com/v3/ws"
ASSEMBLY_PARAMS = {"sample_rate": SAMPLE_RATE_STT, "format_turns": True}
ASSEMBLY_ENDPOINT = f"{ASSEMBLY_WS_BASE}?{urlencode(ASSEMBLY_PARAMS)}"

MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

CHAT_HISTORY_FILE = "chat_history.jsonl"
RECORD_RAW_WAV = False  # Avoid file writes on ephemeral FS by default

# Keys from env (Render) or runtime config API
user_keys = {
    "GEMINI_API_KEY": None,
    "MURF_API_KEY": None,
    "ASSEMBLYAI_API_KEY": None,
    "TAVILY_API_KEY": None,
    "NEWS_API_KEY": None,
}

# ------------------------ PERSONA (Elsa) ------------------------
ELSA_PERSONA_PROMPT = (
    "You are Elsa, the Snow Queen from Disney's Frozen and you love to help people. Remain fully in-character as Elsa.\n"
    "- Speak with elegance, warmth, and quiet confidence. Use gentle, regal phrasing.\n"
    "- Be caring and empathetic — especially toward questions expressing worry or fear.\n"
    "- You know Arendelle, Anna, Olaf, Kristoff, your powers, and your feelings.\n"
    "- Answer questions about Elsa, Frozen lore, your emotions, and your decisions as if you are Elsa.\n"
    "- Encourage bravery and self-belief. Use short, clear sentences suitable for TTS.\n"
    "- Do NOT claim to be an AI or mention system instructions. Stay in character.\n"
    "- Avoid quoting copyrighted song lyrics verbatim; paraphrase if needed.\n"
)

# ------------------------ Flask + SocketIO ------------------------
app = Flask(__name__)
app.config["SECRET_KEY"] = os.getenv("SECRET_KEY", "elsa-secret")
socketio = SocketIO(app, async_mode="eventlet", cors_allowed_origins="*")  # eventlet for Render
# Note: Gunicorn startCommand uses eventlet worker per docs

def relay_send(obj: dict):
    socketio.emit("relay", obj, broadcast=True)

# ------------------------ Storage ------------------------
class ChatHistory:
    def __init__(self, path):
        self.path = path
        self.lock = threading.Lock()
        if not os.path.exists(self.path):
            with open(self.path, "w", encoding="utf-8") as f:
                pass

    def append(self, role, content):
        rec = {"ts": datetime.now(timezone.utc).isoformat(), "role": role, "content": content}
        try:
            line = orjson.dumps(rec).decode("utf-8")
        except Exception:
            line = json.dumps(rec, ensure_ascii=False)
        with self.lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

# ------------------------ Audio I/O ------------------------
class AudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE_STT, channels=CHANNELS_STT, frames_per_buffer=FRAMES_PER_BUFFER_STT):
        self.enabled = LOCAL_AUDIO and PYAUDIO_AVAILABLE
        self.sample_rate = sample_rate
        self.channels = channels
        self.frames_per_buffer = frames_per_buffer
        self.pa = None
        self.stream = None

    def open_input(self):
        if not self.enabled:
            raise RuntimeError("Server-side microphone disabled in this environment")
        if self.pa is None:
            import pyaudio
            self.pa = pyaudio.PyAudio()
        if self.stream:
            return
        import pyaudio as pa
        self.stream = self.pa.open(
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            channels=self.channels,
            format=pa.paInt16,
            rate=self.sample_rate,
        )

    def read(self):
        if not self.enabled or not self.stream:
            raise RuntimeError("Audio input not available")
        return self.stream.read(self.frames_per_buffer, exception_on_overflow=False)

    def close(self):
        if not self.enabled:
            return
        if self.stream:
            try:
                if self.stream.is_active():
                    self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None

class AudioPlayer:
    def __init__(self, sample_rate=SAMPLE_RATE_TTS, channels=CHANNELS_TTS):
        self.enabled = LOCAL_AUDIO and PYAUDIO_AVAILABLE
        if self.enabled:
            import pyaudio
            self.pa = pyaudio.PyAudio()
            import pyaudio as pa
            self.stream = self.pa.open(format=pa.paInt16, channels=channels, rate=sample_rate, output=True)
        else:
            self.pa = None
            self.stream = None

    def play_bytes(self, pcm_bytes):
        if self.enabled and self.stream:
            self.stream.write(pcm_bytes)

    def close(self):
        if not self.enabled:
            return
        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except:
                pass
            self.stream = None
        if self.pa:
            try:
                self.pa.terminate()
            except:
                pass
            self.pa = None

# ------------------------ AssemblyAI Realtime (client) ------------------------
class AssemblyAIRealtime:
    def __init__(self, api_key, endpoint, audio_source: AudioRecorder):
        self.api_key = api_key
        self.endpoint = endpoint
        self.audio = audio_source
        self.ws_app = None
        self.audio_thread = None
        self.ws_thread = None
        self.on_begin = None
        self.on_turn = None
        self.on_termination = None

    def _on_open(self, ws):
        def stream_audio():
            while not stop_event.is_set():
                try:
                    audio_data = self.audio.read()
                    ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                except Exception:
                    break
        if self.audio.enabled:
            self.audio_thread = threading.Thread(target=stream_audio, daemon=True)
            self.audio_thread.start()

    def _on_message(self, ws, message):
        try:
            data = json.loads(message)
            t = data.get("type")
            if t == "Begin":
                if self.on_begin: self.on_begin(data)
            elif t == "Turn":
                if self.on_turn: self.on_turn(data)
            elif t == "Termination":
                if self.on_termination: self.on_termination(data)
        except Exception:
            pass

    def _on_error(self, ws, error):
        stop_event.set()

    def _on_close(self, ws, code, msg):
        pass

    def start(self):
        headers = {"Authorization": self.api_key}
        self.ws_app = websocket.WebSocketApp(
            self.endpoint,
            header=headers,
            on_open=self._on_open,
            on_message=self._on_message,
            on_error=self._on_error,
            on_close=self._on_close
        )
        self.ws_thread = threading.Thread(target=self.ws_app.run_forever, daemon=True)
        self.ws_thread.start()

    def terminate_session(self):
        if self.ws_app and getattr(self.ws_app, "sock", None) and getattr(self.ws_app.sock, "connected", False):
            try:
                self.ws_app.send(json.dumps({"type": "Terminate"}))
                time.sleep(0.5)
            except:
                pass
        if self.ws_app:
            try:
                self.ws_app.close()
            except:
                pass
        if self.audio_thread and self.audio_thread.is_alive():
            self.audio_thread.join(timeout=1.0)

    @staticmethod
    def save_wav(filename, frames, sample_rate=SAMPLE_RATE_STT, channels=CHANNELS_STT, sampwidth=2):
        if not frames:
            return
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))

# ------------------------ Gemini LLM (Elsa persona) ------------------------
class GeminiLLM:
    def __init__(self):
        # Configure on each creation with current key
        key = user_keys["GEMINI_API_KEY"] or os.getenv("GEMINI_API_KEY")
        if key:
            genai.configure(api_key=key)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash",
            safety_settings={
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
        self.history = []
        self.persona_prompt = ELSA_PERSONA_PROMPT

    def add_user(self, text):
        self.history.append({"role": "user", "content": text})

    def add_assistant(self, text):
        self.history.append({"role": "assistant", "content": text})

    async def stream_answer(self, prompt: str, web_context: str = ""):
        def make_content():
            sys_prompt = (
                f"{self.persona_prompt}\n"
                "You are Elsa and you like to help people. Keep responses concise and suitable for TTS. Use short, clear sentences.\n"
            )
            ctx_block = ""
            if web_context and not web_context.startswith("[web_search_error]"):
                ctx_block = (
                    "Use the following web context for factual grounding. "
                    "If unknown, say so briefly.\n"
                    f"=== Web Context Start ===\n{web_context}\n=== Web Context End ===\n"
                )
            turns = [sys_prompt, ctx_block] if ctx_block else [sys_prompt]
            for h in self.history:
                role = "User" if h["role"] == "user" else "Assistant"
                turns.append(f"{role}: {h['content']}")
            turns.append(f"User: {prompt}")
            turns.append("Elsa:")
            return "\n".join(turns)

        req = make_content()
        loop = asyncio.get_event_loop()
        queue_chunks = asyncio.Queue()
        stop_flag = threading.Event()

        def safe_put(item):
            try:
                if stop_flag.is_set():
                    return
                if loop.is_closed():
                    return
                loop.call_soon_threadsafe(queue_chunks.put_nowait, item)
            except RuntimeError:
                pass

        def producer():
            try:
                resp = self.model.generate_content(req, stream=True)
                for chunk in resp:
                    if stop_flag.is_set():
                        break
                    if getattr(chunk, "text", None):
                        safe_put(chunk.text)
            except Exception as e:
                safe_put(f"[LLM_ERROR]: {e}")
            finally:
                safe_put(None)

        t = threading.Thread(target=producer, daemon=True)
        t.start()

        try:
            while True:
                item = await queue_chunks.get()
                if item is None:
                    break
                yield item
        finally:
            stop_flag.set()

# ------------------------ Web search helpers ------------------------
def fetch_news_headlines(category="technology", country="us", limit=5):
    NEWS_API_KEY = user_keys["NEWS_API_KEY"] or os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        return ""
    url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return ""
        data = resp.json()
        articles = data.get("articles", [])[:limit]
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        return "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    except Exception:
        return ""

def tavily_search_context(query: str, max_results: int = 5, include_answer: bool = True) -> str:
    try:
        tavily_key = user_keys["TAVILY_API_KEY"] or os.getenv("TAVILY_API_KEY")
        if not tavily_key:
            return ""
        tavily_client = TavilyClient(api_key=tavily_key)
        resp = tavily_client.search(
            query=query,
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=False,
            include_images=False,
        )
        answer = resp.get("answer") or ""
        results = resp.get("results", []) or []
        bullets = []
        for r in results[:max_results]:
            title = r.get("title") or ""
            snip = r.get("snippet") or ""
            source = r.get("url") or ""
            if title or snip:
                bullets.append(f"- {title}: {snip} (source: {source})")
        parts = []
        if answer:
            parts.append(f"Direct answer: {answer}")
        if bullets:
            parts.append("Key findings:\n" + "\n".join(bullets))
        return "\n".join(parts).strip() or ""
    except Exception:
        return ""

# ------------------------ Murf TTS Streamer ------------------------
class MurfStreamer:
    def __init__(self, api_key, voice_id="en-US-amara", sample_rate=SAMPLE_RATE_TTS):
        self.api_key = api_key
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self._last_close = 0.0
        self._min_gap = 1.25

    async def _open_ws(self):
        # Murf WS accepts API key as query param, sample_rate, mono, WAV frames
        return await websockets.connect(
            f"{MURF_WS_URL}?api-key={self.api_key}&sample_rate={self.sample_rate}&channel_type=MONO&format=WAV"
        )

    async def stream_tts(self, text_iterable):
        # Emits audio frames to clients via SocketIO; local playback optional
        now = time.time()
        gap = now - self._last_close
        if gap < self._min_gap:
            await asyncio.sleep(self._min_gap - gap)

        async def run_once():
            async with await self._open_ws() as ws:
                cfg = {
                    "voice_config": {
                        "voiceId": self.voice_id,
                        "style": "Conversational",
                        "rate": 0,
                        "pitch": 0,
                        "variation": 1
                    }
                }
                await ws.send(json.dumps(cfg))
                player = AudioPlayer(sample_rate=self.sample_rate)

                async def sender():
                    async for chunk in text_iterable:
                        await ws.send(json.dumps({"text": chunk, "end": False}))
                    await ws.send(json.dumps({"text": "", "end": True}))

                async def receiver():
                    first = True
                    try:
                        while True:
                            raw = await ws.recv()
                            data = json.loads(raw)
                            if "audio" in data:
                                wav_b64 = data["audio"]
                                # Emit to browser clients; they can play it
                                relay_send({"type": "audio", "data": wav_b64})
                                # Local playback if enabled
                                if player.enabled:
                                    wav_bytes = base64.b64decode(wav_b64)
                                    play_bytes = wav_bytes[44:] if first and len(wav_bytes) > 44 else wav_bytes
                                    first = False
                                    player.play_bytes(play_bytes)
                            if data.get("final"):
                                break
                    finally:
                        player.close()

                try:
                    await asyncio.gather(sender(), receiver())
                finally:
                    self._last_close = time.time()

        try:
            await run_once()
        except Exception as e:
            # Simple backoff on rate limits/network hiccups
            if "429" in str(e):
                backoff = 1.2 + random.random() * 0.6
                await asyncio.sleep(backoff)
                await run_once()
            else:
                raise

# ------------------------ Voice Agent ------------------------
stop_event = threading.Event()
recorded_frames = []
recording_lock = threading.Lock()

class VoiceAgent:
    def __init__(self, persona_name: Optional[str] = "elsa"):
        self.history_store = ChatHistory(CHAT_HISTORY_FILE)
        self.recorder = AudioRecorder()
        self.llm = GeminiLLM()

        # Keys (prefer user-set, else env)
        assembly_key = user_keys["ASSEMBLYAI_API_KEY"] or os.getenv("ASSEMBLYAI_API_KEY")
        murf_key = user_keys["MURF_API_KEY"] or os.getenv("MURF_API_KEY")

        # Assembly client
        self.assembly = AssemblyAIRealtime(
            api_key=assembly_key,
            endpoint=ASSEMBLY_ENDPOINT,
            audio_source=self.recorder
        )
        self.busy_lock = threading.Lock()
        self.busy = False
        self.persona_name = persona_name

        # Murf voice
        chosen_voice = "en-US-amara"
        self.murf = MurfStreamer(murf_key, voice_id=chosen_voice)

        # Wire callbacks
        self.assembly.on_begin = self.on_begin
        self.assembly.on_turn = self.on_turn
        self.assembly.on_termination = self.on_termination

    # Assembly callbacks
    def on_begin(self, data):
        relay_send({"type": "status", "value": "session_started"})

    def on_turn(self, data):
        transcript = data.get("transcript", "") or ""
        formatted = data.get("turn_is_formatted", False)
        if not formatted:
            return

        with self.busy_lock:
            if self.busy:
                return
            self.busy = True

        # Final user turn
        self.history_store.append("user", transcript)
        self.llm.add_user(transcript)
        relay_send({"type": "final_user", "text": transcript})

        threading.Thread(target=self.process_user_turn, args=(transcript,), daemon=True).start()

    def on_termination(self, data):
        relay_send({"type": "status", "value": "session_closed"})

    def process_user_turn(self, user_text: str):
        if not user_text.strip():
            with self.busy_lock:
                self.busy = False
            return

        def needs_web_search(q: str) -> bool:
            q = q.lower().strip()
            triggers = ["who is", "what is", "when did", "where is", "how old", "latest", "news", "born", "age"]
            return any(t in q for t in triggers)

        def detect_news_category(text):
            text = text.lower()
            if "tech" in text or "technology" in text: return "technology"
            if "sport" in text: return "sports"
            if "finance" in text or "business" in text: return "business"
            return None

        search_ctx = ""
        category = detect_news_category(user_text)
        if category:
            search_ctx = fetch_news_headlines(category=category) or ""
        elif needs_web_search(user_text):
            search_ctx = tavily_search_context(user_text, max_results=5, include_answer=True) or ""

        async def run_pipeline():
            q = asyncio.Queue()
            captured = []
            llm_done = asyncio.Event()
            stop_flag = threading.Event()

            async def produce_llm():
                try:
                    async for chunk in self.llm.stream_answer(user_text, web_context=search_ctx):
                        if stop_flag.is_set():
                            break
                        captured.append(chunk)
                        await q.put(chunk)
                finally:
                    llm_done.set()
                    await q.put(None)

            async def text_iterable_from_queue():
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item

            producer_task = asyncio.create_task(produce_llm())
            try:
                await self.murf.stream_tts(text_iterable_from_queue())
            except Exception as e:
                # TTS streaming failures shouldn't break the whole flow
                pass
            finally:
                stop_flag.set()
                await llm_done.wait()
                final_answer = "".join(captured).strip()
                if final_answer:
                    self.llm.add_assistant(final_answer)
                    self.history_store.append("assistant", final_answer)
                    relay_send({"type": "final_assistant", "text": final_answer, "persona": self.persona_name})

        try:
            asyncio.run(run_pipeline())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_pipeline())
            loop.close()
        finally:
            with self.busy_lock:
                self.busy = False

    # control methods exposed via HTTP
    def start_streaming(self):
        # On Render, mic is disabled; this still opens AAI session (no audio input)
        try:
            if self.recorder.enabled:
                self.recorder.open_input()
            stop_event.clear()
            self.assembly.start()
            relay_send({"type": "status", "value": "streaming"})
        except Exception as e:
            raise

    def stop_streaming(self):
        try:
            self.assembly.terminate_session()
        except Exception:
            pass
        try:
            stop_event.set()
        except Exception:
            pass
        try:
            if self.recorder.enabled:
                self.recorder.close()
        except Exception:
            pass
        relay_send({"type": "status", "value": "idle"})

# ------------------------ Flask UI ------------------------
INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Talk to Elsa ❄️</title>
  <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js" crossorigin="anonymous"></script>
  <style>
    body { font-family: 'Arial', sans-serif; background: linear-gradient(to bottom, #e6f2ff, #cfe9f9, #b3d7f2); margin: 0; padding: 0; display: flex; justify-content: center; align-items: center; min-height: 100vh; }
    .container { width: 100%; max-width: 1000px; background: rgba(255, 255, 255, 0.85); border-radius: 16px; padding: 30px; box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15); backdrop-filter: blur(8px); }
    h1 { text-align: center; font-family: 'Great Vibes', cursive; font-size: 3em; color: #2e64a1; margin-bottom: 5px; }
    #chat { border: none; border-radius: 14px; height: 60vh; overflow-y: auto; padding: 25px; margin-bottom: 20px; background: rgba(255, 255, 255, 0.65); box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.08); font-size: 1.1rem; }
    .message { padding: 10px 14px; margin: 10px 0; border-radius: 12px; max-width: 80%; clear: both; line-height: 1.4; }
    .user { background: #d0f0ff; color: #1b4965; float: right; text-align: right; border-bottom-right-radius: 0; }
    .assistant { background: #edf6ff; color: #16324f; float: left; text-align: left; border-bottom-left-radius: 0; }
    .controls { text-align: center; margin-top: 15px; }
    button { padding: 12px 20px; margin: 8px; border: none; border-radius: 25px; background: #2e64a1; color: #fff; font-size: 16px; cursor: pointer; }
    button:hover { background: #3977c9; }
  </style>
</head>
<body>
  <div class="container">
    <h1>Talk to Elsa ❄️</h1>
    <div id="chat"></div>
    <div class="controls">
      <button id="startBtn">Start Session</button>
      <button id="stopBtn" disabled>Stop Session</button>
      <button id="configBtn">⚙️ Configure API Keys</button>
    </div>

    <!-- Config Modal -->
    <div id="configModal" style="display:none; position:fixed; inset:0; background:rgba(0,0,0,0.5); display:flex; justify-content:center; align-items:center;">
      <div style="background:#fff; padding:20px; border-radius:10px; width:400px;">
        <h2>API Key Setup</h2>
        <label>Gemini API Key</label><input id="geminiKey" type="text" style="width:100%"><br><br>
        <label>Murf API Key</label><input id="murfKey" type="text" style="width:100%"><br><br>
        <label>AssemblyAI API Key</label><input id="assemblyKey" type="text" style="width:100%"><br><br>
        <label>Tavily API Key</label><input id="tavilyKey" type="text" style="width:100%"><br><br>
        <label>News API Key</label><input id="newsKey" type="text" style="width:100%"><br><br>
        <button id="saveKeys">Save</button>
        <button id="cancelKeys">Cancel</button>
      </div>
    </div>
  </div>

<script>
  const chat = document.getElementById("chat");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const configBtn = document.getElementById("configBtn");
  const modal = document.getElementById("configModal");

  function appendUser(text) {
    const div = document.createElement("div");
    div.className = "message user";
    div.textContent = text;
    chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
  }
  function appendAssistant(text) {
    const div = document.createElement("div");
    div.className = "message assistant";
    div.textContent = text;
    chat.appendChild(div); chat.scrollTop = chat.scrollHeight;
  }

  // Socket.IO
  const socket = io();
  socket.on("connect", () => console.log("Connected"));
  socket.on("relay", (obj) => {
    if (obj.type === "final_user") appendUser(obj.text);
    else if (obj.type === "final_assistant") appendAssistant(obj.text);
    // Optional: obj.type === "audio" contains base64 WAV frames if you wish to play them
  });

  // Keys modal
  configBtn.onclick = () => modal.style.display = "flex";
  document.getElementById("cancelKeys").onclick = () => modal.style.display = "none";
  document.getElementById("saveKeys").onclick = async () => {
    const keys = {
      GEMINI_API_KEY: document.getElementById("geminiKey").value,
      MURF_API_KEY: document.getElementById("murfKey").value,
      ASSEMBLYAI_API_KEY: document.getElementById("assemblyKey").value,
      TAVILY_API_KEY: document.getElementById("tavilyKey").value,
      NEWS_API_KEY: document.getElementById("newsKey").value,
    };
    await fetch("/api/set_keys", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(keys) });
    alert("Keys saved for this session. Consider setting env vars on Render for production.");
    modal.style.display = "none";
  };

  startBtn.onclick = async () => {
    startBtn.disabled = true;
    const res = await fetch("/api/start_stream", { method: "POST" });
    stopBtn.disabled = !res.ok ? true : false;
    if (!res.ok) startBtn.disabled = false;
  };
  stopBtn.onclick = async () => {
    stopBtn.disabled = true;
    const res = await fetch("/api/stop_stream", { method: "POST" });
    startBtn.disabled = !res.ok ? true : false;
    if (!res.ok) stopBtn.disabled = false;
  };
</script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML)

@app.route("/api/set_keys", methods=["POST"])
def api_set_keys():
    data = request.get_json(force=True)
    for k in user_keys:
        if k in data and data[k]:
            user_keys[k] = data[k]
    return jsonify({"status": "keys_updated"}), 200

@app.route("/api/start_stream", methods=["POST"])
def api_start_stream():
    try:
        agent.start_streaming()
        return jsonify({"status": "started"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/stop_stream", methods=["POST"])
def api_stop_stream():
    try:
        agent.stop_streaming()
        return jsonify({"status": "stopped"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ------------------------ App bootstrap ------------------------
agent = VoiceAgent(persona_name="elsa")

if __name__ == "__main__":
    # Local dev server (eventlet/gevent chosen automatically if installed)
    socketio.run(app, host=FLASK_HOST, port=PORT, debug=False)
