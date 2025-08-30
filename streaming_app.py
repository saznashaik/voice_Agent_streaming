# app.py
# Public web: Browser mic -> AssemblyAI WS -> Final transcript -> Server (Gemini -> Murf) -> Relay WS -> UI
# Render-friendly: single PORT, dynamic wss, ephemeral token for AssemblyAI.

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

# Networking / Web
import websocket                 # websocket-client (Assembly client if needed)
import websockets                # websockets (async server & client)
from flask import Flask, render_template_string, jsonify, request

# Audio (server-side playback optional; kept for Murf -> speakers on server)
import pyaudio

# JSON perf
import orjson

# Gemini (Google generative ai)
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from tavily import TavilyClient

# Store user-provided keys in memory
user_keys = {
    "GEMINI_API_KEY": None,
    "MURF_API_KEY": None,
    "ASSEMBLYAI_API_KEY": None,
    "TAVILY_API_KEY": None,
    "NEWS_API_KEY": None,
}

# ------------------- CONFIG -------------------
FLASK_HOST = "0.0.0.0"
# Render provides PORT; default 5000 locally
FLASK_PORT = int(os.getenv("PORT", "5000"))

# Important: use a single public port for both HTTP and WS on Render
RELAY_HOST = "0.0.0.0"
RELAY_PORT = FLASK_PORT

SAMPLE_RATE_STT = 16000
FRAMES_PER_BUFFER_STT = 800   # ~50ms
CHANNELS_STT = 1
FORMAT_STT = pyaudio.paInt16

SAMPLE_RATE_TTS = 44100
CHANNELS_TTS = 1
FORMAT_TTS = pyaudio.paInt16

# Keep these for potential server-side AAI client usage; browser will use v3 URL directly
ASSEMBLY_WS_BASE = "wss://streaming.assemblyai.com/v3/ws"
ASSEMBLY_PARAMS = {"sample_rate": SAMPLE_RATE_STT, "format_turns": True}
ASSEMBLY_ENDPOINT = f"{ASSEMBLY_WS_BASE}?{urlencode(ASSEMBLY_PARAMS)}"

MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

CHAT_HISTORY_FILE = "chat_history.jsonl"
RECORD_RAW_WAV = False  # disable on Render

def fetch_news_headlines(category="technology", country="us", limit=5):
    NEWS_API_KEY = user_keys["NEWS_API_KEY"] or os.getenv("NEWS_API_KEY")
    if not NEWS_API_KEY:
        return "[news_error]: Missing NEWS_API_KEY"
    url = f"https://newsapi.org/v2/top-headlines?country={country}&category={category}&apiKey={NEWS_API_KEY}"
    try:
        resp = requests.get(url, timeout=8)
        if resp.status_code != 200:
            return f"[news_error]: {resp.status_code} {resp.text}"
        data = resp.json()
        articles = data.get("articles", [])[:limit]
        if not articles:
            return "No headlines found."
        headlines = [a.get("title", "") for a in articles if a.get("title")]
        return "\n".join(f"{i+1}. {h}" for i, h in enumerate(headlines))
    except Exception as e:
        return f"[news_error]: {e}"

def tavily_search_context(query: str, max_results: int = 5, include_answer: bool = True) -> str:
    try:
        tavily_key = user_keys["TAVILY_API_KEY"] or os.getenv("TAVILY_API_KEY")
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
    except Exception as e:
        return f"[web_search_error]: {e}"

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

# ------------------------ Globals / State ------------------------
stop_event = threading.Event()
recorded_frames = []
recording_lock = threading.Lock()

# ------------------------ Relay Server ------------------------
class RelayServer:
    def __init__(self, host=RELAY_HOST, port=RELAY_PORT):
        self.host = host
        self.port = port
        self._clients = set()
        self._loop = None
        self._queue = None
        self._thread = None
        self._started = threading.Event()
        self._stopped = threading.Event()
        self._prebuffer = []
        self.last_user = None
        self.last_assistant = None

    def start(self):
        def runner():
            asyncio.run(self._main())
        self._thread = threading.Thread(target=runner, daemon=True)
        self._thread.start()
        self._started.wait(timeout=5)

    def stop(self):
        try:
            self.send({"type": "__shutdown__"})
        except:
            pass
        self._stopped.wait(timeout=5)

    def send(self, obj: dict):
        t = obj.get("type")
        if t == "final_user":
            self.last_user = obj
        elif t == "final_assistant":
            self.last_assistant = obj
        if self._queue is None or self._loop is None:
            self._prebuffer.append(obj)
            return
        try:
            asyncio.run_coroutine_threadsafe(self._queue.put(obj), self._loop)
        except RuntimeError:
            pass

    async def _main(self):
        from websockets.server import serve as ws_serve
        self._loop = asyncio.get_running_loop()
        self._queue = asyncio.Queue()
        # Bind relay WS to same public port on Render
        server = await ws_serve(self._handler, self.host, self.port)
        print(f"[Relay] WebSocket running at ws://{self.host}:{self.port}")
        self._started.set()
        if self._prebuffer:
            for item in self._prebuffer:
                await self._queue.put(item)
            self._prebuffer.clear()
        broadcaster = asyncio.create_task(self._broadcaster())
        try:
            await broadcaster
        finally:
            server.close()
            await server.wait_closed()
            self._stopped.set()

    async def _handler(self, websocket, path):
        self._clients.add(websocket)
        try:
            if self.last_user:
                await websocket.send(json.dumps(self.last_user, ensure_ascii=False))
            if self.last_assistant:
                await websocket.send(json.dumps(self.last_assistant, ensure_ascii=False))
            async for _ in websocket:
                pass
        finally:
            self._clients.discard(websocket)

    async def _broadcaster(self):
        while True:
            obj = await self._queue.get()
            if obj and obj.get("type") == "__shutdown__":
                break
            if not self._clients:
                continue
            msg = json.dumps(obj, ensure_ascii=False)
            dead = []
            for ws in list(self._clients):
                try:
                    await ws.send(msg)
                except:
                    dead.append(ws)
            for ws in dead:
                self._clients.discard(ws)

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

# ------------------------ Audio I/O (server playback) ------------------------
class AudioPlayer:
    def __init__(self, sample_rate=SAMPLE_RATE_TTS, channels=CHANNELS_TTS, format=FORMAT_TTS):
        self.pa = pyaudio.PyAudio()
        self.stream = self.pa.open(format=format, channels=channels, rate=sample_rate, output=True)

    def play_bytes(self, pcm_bytes):
        self.stream.write(pcm_bytes)

    def close(self):
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

# ------------------------ Gemini LLM (Elsa persona, streaming) ------------------------
class GeminiLLM:
    def __init__(self):
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

# ------------------------ Murf TTS Streamer ------------------------
class MurfStreamer:
    def __init__(self, api_key, voice_id="en-US-amara", sample_rate=SAMPLE_RATE_TTS, relay: Optional[RelayServer]=None):
        self.api_key = api_key
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.relay = relay
        self._last_close = 0.0
        self._min_gap = 1.25

    async def _open_ws(self):
        return await websockets.connect(
            f"{MURF_WS_URL}?api-key={self.api_key}&sample_rate={self.sample_rate}&channel_type=MONO&format=WAV"
        )

    async def stream_tts_to_speakers_and_relay(self, text_iterable):
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
                player = None
                try:
                    # Optional server speaker playback (disable on Render if no audio device)
                    try:
                        player = AudioPlayer(sample_rate=self.sample_rate)
                    except Exception:
                        player = None

                    async def sender():
                        async for chunk in text_iterable:
                            await ws.send(json.dumps({"text": chunk, "end": False}))
                        await ws.send(json.dumps({"text": "", "end": True}))

                    async def receiver():
                        first = True
                        while True:
                            raw = await ws.recv()
                            data = json.loads(raw)
                            if "audio" in data:
                                wav_b64 = data["audio"]
                                wav_bytes = base64.b64decode(wav_b64)
                                play_bytes = wav_bytes
                                if first and len(play_bytes) > 44:
                                    play_bytes = play_bytes[44:]
                                    first = False
                                if player:
                                    try:
                                        player.play_bytes(play_bytes)
                                    except Exception:
                                        pass
                                if self.relay:
                                    self.relay.send({"type": "audio", "data": wav_b64})
                            if data.get("final"):
                                break

                    await asyncio.gather(sender(), receiver())
                finally:
                    if player:
                        player.close()
                    self._last_close = time.time()

        try:
            await run_once()
        except Exception as e:
            if "429" in str(e):
                if self.relay:
                    self.relay.send({"type": "status", "value": "tts_backoff"})
                backoff = 1.2 + random.random() * 0.6
                await asyncio.sleep(backoff)
                await run_once()
                if self.relay:
                    self.relay.send({"type": "status", "value": "idle"})
            else:
                raise

# ------------------------ Voice Agent ------------------------
class VoiceAgent:
    def __init__(self, persona_name: Optional[str] = "elsa"):
        gen_key = user_keys["GEMINI_API_KEY"] or os.getenv("GEMINI_API_KEY")
        murf_key = user_keys["MURF_API_KEY"] or os.getenv("MURF_API_KEY")
        # Configure Gemini
        genai.configure(api_key=gen_key)

        self.history_store = ChatHistory(CHAT_HISTORY_FILE)
        self.llm = GeminiLLM()
        self.relay = None
        self.busy_lock = threading.Lock()
        self.busy = False
        self.persona_name = persona_name
        chosen_voice = "en-US-amara"
        self.murf = MurfStreamer(murf_key, voice_id=chosen_voice)

    def attach_relay(self, relay: RelayServer):
        self.relay = relay
        self.murf.relay = relay

    def process_user_turn(self, user_text: str):
        if not user_text.strip():
            with self.busy_lock:
                self.busy = False
            return
        search_ctx = ""

        def needs_web_search(user_text: str) -> bool:
            q = user_text.lower().strip()
            factual_triggers = ["who is", "what is", "when did", "where is", "how old", "latest", "news", "born", "age"]
            return any(t in q for t in factual_triggers)

        def detect_news_category(text):
            text = text.lower()
            if "tech" in text or "technology" in text:
                return "technology"
            if "sport" in text:
                return "sports"
            if "finance" in text or "business" in text:
                return "business"
            return None

        category = detect_news_category(user_text)
        if category:
            search_ctx = fetch_news_headlines(category=category)
        elif needs_web_search(user_text):
            search_ctx = tavily_search_context(user_text, max_results=5, include_answer=True)

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
                await self.murf.stream_tts_to_speakers_and_relay(text_iterable_from_queue())
            except Exception as e:
                print(f"[Pipeline] Murf error: {e}")
            finally:
                stop_flag.set()
                await llm_done.wait()
                final_answer = "".join(captured).strip()
                if final_answer:
                    self.llm.add_assistant(final_answer)
                    self.history_store.append("assistant", final_answer)
                    print(f"\nAssistant(final): {final_answer}\n")
                    if self.relay:
                        self.relay.send({"type": "final_assistant", "text": final_answer, "persona": self.persona_name})

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

# ------------------------ Flask UI (browser mic) ------------------------
app = Flask(__name__)

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Talk to Elsa ❄️</title>
  <link href="https://fonts.googleapis.com/css2?family=Great+Vibes&display=swap" rel="stylesheet">
  <style>
    body {
      font-family: 'Arial', sans-serif;
      background: linear-gradient(to bottom, #e6f2ff, #cfe9f9, #b3d7f2);
      margin: 0; padding: 0; display: flex; justify-content: center; align-items: center;
      height: 100vh; overflow: hidden;
    }
    .container { width: 100%; max-width: 1000px; background: rgba(255,255,255,0.85); border-radius: 16px;
      padding: 30px; box-shadow: 0 8px 20px rgba(0,0,0,0.15); backdrop-filter: blur(8px); }
    h1 { text-align: center; font-family: 'Great Vibes', cursive; font-size: 3em; color: #2e64a1; margin-bottom: 5px; }
    .avatar { display: flex; justify-content: center; margin-bottom: 10px; }
    .avatar img { width: 100px; height: 100px; border-radius: 50%; border: 3px solid #fff; box-shadow: 0 0 20px rgba(173,216,230,0.8); }
    #chat { border: none; border-radius: 14px; height: 550px; overflow-y: auto; padding: 25px; margin-bottom: 20px;
      background: rgba(255,255,255,0.65); box-shadow: inset 0 0 12px rgba(0,0,0,0.08); font-size: 1.1rem; }
    .message { padding: 10px 14px; margin: 10px 0; border-radius: 12px; max-width: 80%; clear: both; position: relative; line-height: 1.4; }
    .user { background: #d0f0ff; color: #1b4965; float: right; text-align: right; border-bottom-right-radius: 0; }
    .assistant { background: #edf6ff; color: #16324f; float: left; text-align: left; border-bottom-left-radius: 0; }
    .persona-label { font-size: 0.8em; color: #5c6f80; margin-bottom: 3px; display: block; font-style: italic; }
    .controls { text-align: center; margin-top: 15px; }
    button { padding: 12px 20px; margin: 8px; border: none; border-radius: 25px; background: #2e64a1; color: #fff; font-size: 16px; cursor: pointer; }
    button:hover { background: #3977c9; box-shadow: 0 0 10px rgba(46,100,161,0.5); }
    button:disabled { background: #a3c2dd; cursor: not-allowed; }
    .snowflake { position: fixed; top: -10px; color: white; user-select: none; pointer-events: none; font-size: 12px; animation-name: fall; animation-timing-function: linear; animation-iteration-count: infinite; }
    @keyframes fall { 0% { transform: translateY(-10px) rotate(0deg); } 100% { transform: translateY(110vh) rotate(360deg); } }
  </style>
</head>
<body>
  <div class="container">
    <h1>Talk to Elsa ❄️</h1>
    <div class="avatar">
      <img src="https://i.pinimg.com/736x/38/4f/d5/384fd5716668760e53fee4294e229936.jpg" alt="Elsa">
    </div>
    <div id="chat"></div>
    <div class="controls">
      <button id="startBtn">Start Recording</button>
      <button id="stopBtn" disabled>Stop Recording</button>
      <button id="configBtn">⚙️ Configure API Keys</button>
    </div>

    <!-- Config Modal -->
    <div id="configModal" style="display:none; position:fixed; top:0; left:0; width:100%; height:100%; background:rgba(0,0,0,0.5); display:flex; justify-content:center; align-items:center;">
      <div style="background:#fff; padding:20px; border-radius:10px; width:400px;">
        <h2>API Key Setup</h2>
        <label>Gemini API Key</label>
        <input id="geminiKey" type="text" style="width:100%"><br><br>
        <label>Murf API Key</label>
        <input id="murfKey" type="text" style="width:100%"><br><br>
        <label>AssemblyAI API Key</label>
        <input id="assemblyKey" type="text" style="width:100%"><br><br>
        <label>Tavily API Key</label>
        <input id="tavilyKey" type="text" style="width:100%"><br><br>
        <label>News API Key</label>
        <input id="newsKey" type="text" style="width:100%"><br><br>
        <button onclick="saveKeys()">Save</button>
        <button onclick="closeModal()">Cancel</button>
      </div>
    </div>
  </div>

  <script>
    // Snowflakes
    const snowCount = 50;
    for (let i = 0; i < snowCount; i++) {
      const flake = document.createElement("div");
      flake.className = "snowflake";
      flake.style.left = Math.random() * 100 + "vw";
      flake.style.fontSize = 10 + Math.random() * 20 + "px";
      flake.style.animationDuration = 5 + Math.random() * 10 + "s";
      flake.style.animationDelay = Math.random() * 10 + "s";
      flake.textContent = "❄";
      document.body.appendChild(flake);
    }

    // Modal
    const configBtn = document.getElementById("configBtn");
    const modal = document.getElementById("configModal");
    configBtn.onclick = () => modal.style.display = "flex";
    function closeModal() { modal.style.display = "none"; }
    async function saveKeys() {
      const keys = {
        GEMINI_API_KEY: document.getElementById("geminiKey").value,
        MURF_API_KEY: document.getElementById("murfKey").value,
        ASSEMBLYAI_API_KEY: document.getElementById("assemblyKey").value,
        TAVILY_API_KEY: document.getElementById("tavilyKey").value,
        NEWS_API_KEY: document.getElementById("newsKey").value,
      };
      await fetch("/api/set_keys", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(keys) });
      alert("Keys saved! You can now start recording.");
      closeModal();
    }

    // Relay URL: use same origin and wss if HTTPS (Render)
    const RELAY_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host;
    const chat = document.getElementById("chat");
    const startBtn = document.getElementById("startBtn");
    const stopBtn = document.getElementById("stopBtn");

    function appendUser(text) {
      const div = document.createElement("div");
      div.className = "message user";
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }
    function appendAssistant(text) {
      const div = document.createElement("div");
      div.className = "message assistant";
      div.textContent = text;
      chat.appendChild(div);
      chat.scrollTop = chat.scrollHeight;
    }

    // Connect Relay WS
    let ws;
    function connectRelay() {
      ws = new WebSocket(RELAY_URL);
      ws.onopen = () => console.log("Relay connected");
      ws.onmessage = (ev) => {
        let obj = JSON.parse(ev.data);
        if (obj.type === "final_user") appendUser(obj.text);
        else if (obj.type === "final_assistant") appendAssistant(obj.text);
        else if (obj.type === "audio" && obj.data) {
          // Optional: play Murf audio in browser
          try {
            const wavBytes = Uint8Array.from(atob(obj.data), c => c.charCodeAt(0));
            const blob = new Blob([wavBytes], { type: "audio/wav" });
            const url = URL.createObjectURL(blob);
            const audio = new Audio(url);
            audio.play().catch(()=>{});
          } catch (e) {}
        }
      };
      ws.onclose = () => setTimeout(connectRelay, 1000);
    }
    connectRelay();

    // Browser mic -> AssemblyAI WS v3
    let aaiWS = null;
    let recorder = null;
    let started = false;

    async function startRecording() {
      if (started) return;
      started = true;
      startBtn.disabled = true; stopBtn.disabled = false;

      // 1) get ephemeral token
      const tokRes = await fetch("/api/tokens/transcription");
      const tokJson = await tokRes.json();
      const token = tokJson && tokJson.token;
      if (!token) {
        alert("Failed to get AssemblyAI token");
        stopRecording();
        return;
      }

      // 2) connect to AAI v3
      const sampleRate = 16000;
      const url = `wss://streaming.assemblyai.com/v3/ws?sample_rate=${sampleRate}&encoding=pcm_s16le&token=${token}`;
      aaiWS = new WebSocket(url);

      aaiWS.onopen = async () => {
        console.log("AAI WS opened");
        // 3) capture mic with MediaRecorder; 250ms chunks
        try {
          const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
          const supportsPCM = MediaRecorder.isTypeSupported("audio/webm;codecs=pcm");
          const mr = new MediaRecorder(stream, supportsPCM ? { mimeType: "audio/webm;codecs=pcm" } : undefined);
          mr.addEventListener("dataavailable", async (event) => {
            if (event.data.size > 0 && aaiWS?.readyState === WebSocket.OPEN) {
              const base64 = await blobToBase64(event.data);
              const chunk = base64.split("base64,")[12];
              aaiWS.send(JSON.stringify({ audio_data: chunk }));
            }
          });
          mr.start(250);
          recorder = { mr, stream };
        } catch (e) {
          console.error(e);
          alert("Mic permission or recording failed");
          stopRecording();
        }
      };

      aaiWS.onmessage = (msg) => {
        try {
          const data = JSON.parse(msg.data);
          // Universal Streaming returns turn-based results; act on final text
          if (data && data.text && (data.message_type === "FinalTranscript" || data.type === "FinalTranscript")) {
            const finalText = (data.text || "").trim();
            if (finalText) {
              appendUser(finalText);
              fetch("/api/voice_turn", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ text: finalText })
              }).catch(console.error);
            }
          }
        } catch (e) {}
      };

      aaiWS.onerror = (e) => {
        console.error("AAI WS error", e);
        stopRecording();
      };
      aaiWS.onclose = () => {
        console.log("AAI WS closed");
        stopRecording();
      };
    }

    async function stopRecording() {
      if (!started) return;
      started = false;
      startBtn.disabled = false; stopBtn.disabled = true;
      try {
        if (recorder?.mr && recorder.mr.state !== "inactive") recorder.mr.stop();
        if (recorder?.stream) recorder.stream.getTracks().forEach(t => t.stop());
      } catch (e) {}
      try {
        if (aaiWS && aaiWS.readyState === WebSocket.OPEN) {
          aaiWS.send(JSON.stringify({ type: "Terminate" }));
          aaiWS.close();
        }
      } catch (e) {}
      aaiWS = null; recorder = null;
    }

    startBtn.onclick = startRecording;
    stopBtn.onclick = stopRecording;

    function blobToBase64(blob) {
      return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onloadend = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(blob);
      });
    }
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
    # Reconfigure Gemini on the fly if provided
    gen_key = user_keys["GEMINI_API_KEY"] or os.getenv("GEMINI_API_KEY")
    if gen_key:
        genai.configure(api_key=gen_key)
    return jsonify({"status": "keys_updated"}), 200

# New: ephemeral AssemblyAI token for browser
@app.route("/api/tokens/transcription", methods=["GET"])
def get_aai_token():
    aai_key = user_keys["ASSEMBLYAI_API_KEY"] or os.getenv("ASSEMBLYAI_API_KEY")
    if not aai_key:
        return jsonify({"error": "Missing ASSEMBLYAI_API_KEY"}), 400
    try:
        # v3 token endpoint with short expiry (e.g., 60s)
        r = requests.get(
            "https://streaming.assemblyai.com/v3/token?expires_in_seconds=60",
            headers={"authorization": aai_key},
            timeout=8,
        )
        r.raise_for_status()
        data = r.json()
        # data = { "token": "..." }
        return jsonify({"token": data.get("token")})
    except Exception as e:
        return jsonify({"error": f"token_issue: {e}"}), 500

# New: accept final transcript and run agent pipeline
@app.route("/api/voice_turn", methods=["POST"])
def api_voice_turn():
    data = request.get_json(force=True)
    user_text = (data.get("text") or "").strip()
    if not user_text:
        return jsonify({"error": "empty_text"}), 400
    # enqueue processing; relay will broadcast results
    def run():
        agent.llm.add_user(user_text)
        if agent.relay:
            agent.relay.send({"type": "final_user", "text": user_text})
        with agent.busy_lock:
            if agent.busy:
                # skip if busy to avoid overlap; real app could queue
                return
            agent.busy = True
        try:
            agent.process_user_turn(user_text)
        finally:
            with agent.busy_lock:
                agent.busy = False
    threading.Thread(target=run, daemon=True).start()
    return jsonify({"status": "processing"}), 200

# Deprecated in browser-mic mode: keep as no-op
@app.route("/api/start_stream", methods=["POST"])
def api_start_stream():
    return jsonify({"status": "no_server_mic"}), 200

@app.route("/api/stop_stream", methods=["POST"])
def api_stop_stream():
    return jsonify({"status": "no_server_mic"}), 200

# ------------------------ Run App ------------------------
if __name__ == "__main__":
    # Start relay
    relay = RelayServer(host=RELAY_HOST, port=RELAY_PORT)
    relay.start()

    # Start agent and attach relay
    agent = VoiceAgent(persona_name="elsa")
    agent.attach_relay(relay)

    # Run Flask
    def run_flask():
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print(f"UI available. Single-port server on :{FLASK_PORT} (Relay WS shares same port). Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        try:
            relay.stop()
        except:
            pass
        print("Exited.")
