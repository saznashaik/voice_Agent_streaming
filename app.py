# app.py
# Flask UI + Single-Port WebSocket bridge for browser audio
# Browser captures mic (WebRTC/MediaStream + AudioWorklet at 16 kHz PCM),
# sends binary PCM to /audio WS -> forwarded to AssemblyAI realtime WS.
# Server streams Murf TTS WAV back over the same /audio WS for WebAudio playback.
# PyAudio is removed; suitable for Render single-port deployment.

import os
import time
import json
import base64
import threading
import asyncio
from datetime import datetime, timezone
from urllib.parse import urlencode
from typing import Optional, Set

from dotenv import load_dotenv
load_dotenv()

from flask import Flask, render_template_string, jsonify, request, send_from_directory

# WebSocket server for browser <-> server
import websockets
from websockets.server import serve as ws_serve

# AssemblyAI client uses websocket-client
import websocket as ws_client

# JSON perf
import orjson

# LLM and deps
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import requests
from tavily import TavilyClient

# ------------------- CONFIG -------------------
FLASK_HOST = "0.0.0.0"
# Render sets PORT; fall back to 5000 locally
FLASK_PORT = int(os.getenv("PORT", os.getenv("FLASK_PORT", "5000")))

SAMPLE_RATE_STT = 16000

ASSEMBLY_WS_BASE = "wss://streaming.assemblyai.com/v3/ws"
ASSEMBLY_PARAMS = {"sample_rate": SAMPLE_RATE_STT, "format_turns": True}
ASSEMBLY_ENDPOINT = f"{ASSEMBLY_WS_BASE}?{urlencode(ASSEMBLY_PARAMS)}"

MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

CHAT_HISTORY_FILE = "chat_history.jsonl"
RECORD_RAW_WAV = False  # optional; off by default for Render

AUDIO_WS_PATH = "/audio"  # browser WebSocket path

# ------------------- User keys -------------------
user_keys = {
    "GEMINI_API_KEY": None,
    "MURF_API_KEY": None,
    "ASSEMBLYAI_API_KEY": None,
    "TAVILY_API_KEY": None,
    "NEWS_API_KEY": None,
}

# ------------------- Helpers -------------------
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

ELSA_PERSONA_PROMPT = (
    "You are Elsa, the Snow Queen from Disney's Frozen and you love to help people. Remain fully in-character as Elsa.\n"
    "- Speak with elegance, warmth, and quiet confidence. Use gentle, regal phrasing.\n"
    "- Be caring and empathetic.\n"
    "- Use short, clear sentences suitable for TTS.\n"
    "- Avoid quoting copyrighted song lyrics verbatim.\n"
)

# ------------------- LLM -------------------
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
                "You are Elsa and you like to help people. Keep responses concise and suitable for TTS.\n"
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
                if stop_flag.is_set() or loop.is_closed():
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

# ------------------- AssemblyAI realtime (browser-driven) -------------------
class AssemblyAIRealtime:
    def __init__(self, api_key, endpoint):
        self.api_key = api_key
        self.endpoint = endpoint
        self.ws_app: Optional[ws_client.WebSocketApp] = None
        self.ws_thread: Optional[threading.Thread] = None
        self.on_begin = None
        self.on_turn = None
        self.on_termination = None

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
        except Exception as e:
            print(f"[AssemblyAI] on_message error: {e}")

    def _on_error(self, ws, error):
        print(f"[AssemblyAI] WS error: {error}")

    def _on_close(self, ws, code, msg):
        print(f"[AssemblyAI] WS closed: code={code}, msg={msg}")

    def _on_open(self, ws):
        print("AssemblyAI WS opened (browser audio mode).")
        # No audio thread; browser hub forwards PCM binary frames.

    def start(self):
        headers = {"Authorization": self.api_key}
        self.ws_app = ws_client.WebSocketApp(
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
            except Exception as e:
                print(f"[AssemblyAI] terminate send error: {e}")
        try:
            self.ws_app and self.ws_app.close()
        except:
            pass

# ------------------- Browser hub (WS /audio) -------------------
class BrowserAudioHub:
    def __init__(self, agent: "VoiceAgent"):
        self.agent = agent
        self.clients: Set[websockets.WebSocketServerProtocol] = set()

    async def handler(self, websocket):
        self.clients.add(websocket)
        try:
            await websocket.send(orjson.dumps({"type":"ready"}).decode("utf-8"))
            async for msg in websocket:
                # Binary -> forward to AssemblyAI as binary
                if isinstance(msg, (bytes, bytearray)):
                    try:
                        aai = self.agent.assembly
                        if aai and aai.ws_app:
                            aai.ws_app.send(msg, ws_client.ABNF.OPCODE_BINARY)
                    except Exception as e:
                        print(f"[Hub] AAI forward error: {e}")
                else:
                    # Optional JSON control messages; ignore for now
                    pass
        finally:
            self.clients.discard(websocket)

    async def broadcast_json(self, obj: dict):
        if not self.clients:
            return
        msg = orjson.dumps(obj).decode("utf-8")
        dead=[]
        for ws in list(self.clients):
            try:
                await ws.send(msg)
            except:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def broadcast_tts_b64(self, wav_b64: str):
        await self.broadcast_json({"type":"tts_audio","audio":wav_b64})

# ------------------- Murf streamer (to browser) -------------------
class MurfStreamer:
    def __init__(self, api_key, voice_id="en-US-amara", sample_rate=44100, browser_hub: Optional[BrowserAudioHub]=None):
        self.api_key = api_key
        self.voice_id = voice_id
        self.sample_rate = sample_rate
        self.browser_hub = browser_hub
        self._last_close = 0.0
        self._min_gap = 1.25

    async def _open_ws(self):
        return await websockets.connect(
            f"{MURF_WS_URL}?api-key={self.api_key}&sample_rate={self.sample_rate}&channel_type=MONO&format=WAV"
        )

    async def stream_tts(self, text_iterable):
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

                async def sender():
                    async for chunk in text_iterable:
                        await ws.send(json.dumps({"text": chunk, "end": False}))
                    await ws.send(json.dumps({"text": "", "end": True}))

                async def receiver():
                    try:
                        while True:
                            raw = await ws.recv()
                            data = json.loads(raw)
                            if "audio" in data and self.browser_hub:
                                wav_b64 = data["audio"]
                                await self.browser_hub.broadcast_tts_b64(wav_b64)
                            if data.get("final"):
                                break
                    finally:
                        pass

                try:
                    await asyncio.gather(sender(), receiver())
                finally:
                    self._last_close = time.time()

        try:
            await run_once()
        except Exception as e:
            if "429" in str(e):
                backoff = 1.2
                await asyncio.sleep(backoff)
                await run_once()
            else:
                raise

# ------------------- Voice Agent -------------------
stop_event = threading.Event()

class VoiceAgent:
    def __init__(self, browser_hub: BrowserAudioHub):
        gen_key = user_keys["GEMINI_API_KEY"] or os.getenv("GEMINI_API_KEY")
        murf_key = user_keys["MURF_API_KEY"] or os.getenv("MURF_API_KEY")
        assembly_key = user_keys["ASSEMBLYAI_API_KEY"] or os.getenv("ASSEMBLYAI_API_KEY")

        genai.configure(api_key=gen_key)

        self.history_store = ChatHistory(CHAT_HISTORY_FILE)
        self.llm = GeminiLLM()
        self.browser_hub = browser_hub

        self.assembly = AssemblyAIRealtime(
            api_key=assembly_key,
            endpoint=ASSEMBLY_ENDPOINT
        )
        self.busy_lock = threading.Lock()
        self.busy = False

        self.murf = MurfStreamer(murf_key, voice_id="en-US-amara", browser_hub=browser_hub)

        # Hook AssemblyAI callbacks
        self.assembly.on_begin = self.on_begin
        self.assembly.on_turn = self.on_turn
        self.assembly.on_termination = self.on_termination

    def on_begin(self, data):
        print(f"AAI session started: {data.get('id')}")

    def on_turn(self, data):
        transcript = data.get("transcript", "") or ""
        formatted = data.get("turn_is_formatted", False)
        if not formatted:
            return
        with self.busy_lock:
            if self.busy:
                print("[Agent] Busy; skip overlapping turn")
                return
            self.busy = True

        print(f"User(final): {transcript}")
        self.history_store.append("user", transcript)
        self.llm.add_user(transcript)

        # Fire pipeline async
        threading.Thread(target=self.process_user_turn, args=(transcript,), daemon=True).start()

    def on_termination(self, data):
        print(f"AAI session terminated")

    def detect_news_category(self, text):
        t = text.lower()
        if "tech" in t or "technology" in t:
            return "technology"
        if "sport" in t:
            return "sports"
        if "finance" in t or "business" in t:
            return "business"
        return None

    def needs_web_search(self, text):
        q = text.lower().strip()
        triggers = ["who is", "what is", "when did", "where is", "how old", "latest", "news", "born", "age"]
        return any(t in q for t in triggers)

    def process_user_turn(self, user_text: str):
        if not user_text.strip():
            with self.busy_lock:
                self.busy = False
            return
        search_ctx = ""
        cat = self.detect_news_category(user_text)
        if cat:
            search_ctx = fetch_news_headlines(category=cat)
        elif self.needs_web_search(user_text):
            search_ctx = tavily_search_context(user_text, max_results=5, include_answer=True)

        async def run_pipeline():
            q = asyncio.Queue()
            captured = []
            llm_done = asyncio.Event()

            async def produce_llm():
                try:
                    async for chunk in self.llm.stream_answer(user_text, web_context=search_ctx):
                        captured.append(chunk)
                        await q.put(chunk)
                finally:
                    llm_done.set()
                    await q.put(None)

            async def text_iterable():
                while True:
                    item = await q.get()
                    if item is None:
                        break
                    yield item

            producer_task = asyncio.create_task(produce_llm())
            try:
                await self.murf.stream_tts(text_iterable())
            finally:
                await llm_done.wait()
                final_answer = "".join(captured).strip()
                if final_answer:
                    self.llm.add_assistant(final_answer)
                    self.history_store.append("assistant", final_answer)
                    print(f"\nAssistant(final): {final_answer}\n")
                    # Also push text to browser UI
                    await self.browser_hub.broadcast_json({"type":"final_assistant","text":final_answer})

            with self.busy_lock:
                self.busy = False

        try:
            asyncio.run(run_pipeline())
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(run_pipeline())
            loop.close()

    def start_streaming(self):
        # Start AssemblyAI realtime session; browser will feed PCM
        stop_event.clear()
        self.assembly.start()

    def stop_streaming(self):
        try:
            self.assembly.terminate_session()
        finally:
            stop_event.set()

# ------------------- Flask UI -------------------
app = Flask(__name__, static_folder="static", static_url_path="/static")

INDEX_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <title>Talk to Elsa ❄️</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
    body { font-family: Arial, sans-serif; max-width: 860px; margin: 20px auto; }
    #chat { border:1px solid #ddd; border-radius:12px; padding:16px; height: 420px; overflow-y:auto; }
    .message { padding: 8px 10px; margin: 8px 0; border-radius: 10px; max-width: 85%; }
    .user { background:#d0f0ff; margin-left:auto; text-align:right; }
    .assistant { background:#edf6ff; margin-right:auto; }
    .controls { margin-top:12px; }
    button { padding:10px 16px; margin-right:8px; }
  </style>
</head>
<body>
  <h1>Talk to Elsa ❄️</h1>
  <div id="chat"></div>
  <div class="controls">
    <button id="startBtn">Start</button>
    <button id="stopBtn" disabled>Stop</button>
    <button id="configBtn">Configure API Keys</button>
  </div>

  <script>
  const chat = document.getElementById("chat");
  const startBtn = document.getElementById("startBtn");
  const stopBtn = document.getElementById("stopBtn");
  const configBtn = document.getElementById("configBtn");

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

  let audioCtx = null;
  let workletNode = null;
  let micStream = null;
  let socket = null;
  let isRecording = false;

  const AUDIO_WS_URL = (location.protocol === "https:" ? "wss://" : "ws://") + location.host + "{{ audio_path }}";

  async function connectWS() {
    return new Promise((resolve, reject) => {
      socket = new WebSocket(AUDIO_WS_URL);
      socket.binaryType = 'arraybuffer';
      socket.onopen = () => resolve();
      socket.onmessage = async (ev) => {
        try {
          const msg = JSON.parse(ev.data);
          if (msg.type === "tts_audio" && msg.audio) {
            await playWavBase64(msg.audio);
          }
          if (msg.type === "final_assistant") {
            appendAssistant(msg.text);
          }
        } catch(e) { /* ignore */ }
      };
      socket.onerror = reject;
      socket.onclose = () => {};
    });
  }

  async function startMic() {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, sampleRate: 48000, echoCancellation: true, noiseSuppression: true },
      video: false
    });
    audioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 48000 });
    await audioCtx.audioWorklet.addModule("/static/pcm-worklet.js");
    const source = audioCtx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioCtx, 'pcm-writer', { processorOptions: { targetSampleRate: 16000 } });
    workletNode.port.onmessage = (ev) => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send(ev.data); // ArrayBuffer of PCM int16 mono 16k
      }
    };
    source.connect(workletNode);
    // Optional monitor: workletNode.connect(audioCtx.destination);
  }

  async function playWavBase64(b64) {
    if (!audioCtx) audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    const resp = await fetch("data:audio/wav;base64," + b64);
    const arr = await resp.arrayBuffer();
    const buf = await audioCtx.decodeAudioData(arr.slice(0));
    const src = audioCtx.createBufferSource();
    src.buffer = buf;
    src.connect(audioCtx.destination);
    src.start();
  }

  async function startAll() {
    if (isRecording) return;
    isRecording = true;
    startBtn.disabled = true;
    try {
      await connectWS();
      await startMic();
      // Start server AAI session
      await fetch("/api/start_stream", { method: "POST" });
      stopBtn.disabled = false;
    } catch (e) {
      console.error(e);
      startBtn.disabled = false;
      isRecording = false;
    }
  }

  async function stopAll() {
    if (!isRecording) return;
    isRecording = false;
    stopBtn.disabled = true;
    try {
      await fetch("/api/stop_stream", { method: "POST" });
    } catch(e){}
    try {
      if (workletNode) { workletNode.disconnect(); workletNode = null; }
      if (audioCtx) { await audioCtx.close(); audioCtx = null; }
      if (micStream) { micStream.getTracks().forEach(t => t.stop()); micStream = null; }
      if (socket && socket.readyState === WebSocket.OPEN) socket.close();
    } finally {
      startBtn.disabled = false;
    }
  }

  startBtn.onclick = startAll;
  stopBtn.onclick = stopAll;

  configBtn.onclick = async () => {
    const g = prompt("Gemini API Key (leave blank to keep)");
    const m = prompt("Murf API Key (leave blank to keep)");
    const a = prompt("AssemblyAI API Key (leave blank to keep)");
    const t = prompt("Tavily API Key (optional)");
    const n = prompt("News API Key (optional)");
    const body = {};
    if (g) body.GEMINI_API_KEY = g;
    if (m) body.MURF_API_KEY = m;
    if (a) body.ASSEMBLYAI_API_KEY = a;
    if (t) body.TAVILY_API_KEY = t;
    if (n) body.NEWS_API_KEY = n;
    await fetch("/api/set_keys", { method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(body) });
    alert("Keys updated");
  };
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, audio_path=AUDIO_WS_PATH)

@app.route("/static/<path:filename>")
def static_files(filename):
    return send_from_directory(app.static_folder, filename)

@app.route("/api/set_keys", methods=["POST"])
def api_set_keys():
    data = request.get_json(force=True)
    for k in user_keys:
        if k in data and data[k]:
            user_keys[k] = data[k]
    return jsonify({"status": "keys_updated"}), 200

# Start/Stop AAI session (browser drives the PCM flow)
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

# ------------------- Boot -------------------
async def ws_main(hub: BrowserAudioHub, host, port):
    async def ws_router(websocket, path):
        if path == AUDIO_WS_PATH:
            await hub.handler(websocket)
        else:
            await websocket.close()
    async with ws_serve(ws_router, host, port):
        await asyncio.Future()

if __name__ == "__main__":
    # init hub and agent
    hub = BrowserAudioHub(agent=None)  # temporary, will set after agent constructed
    agent = VoiceAgent(browser_hub=hub)
    hub.agent = agent

    # run Flask in thread
    def run_flask():
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
    threading.Thread(target=run_flask, daemon=True).start()

    print(f"Serving at http://{FLASK_HOST}:{FLASK_PORT} and WS on {AUDIO_WS_PATH}")

    try:
        asyncio.run(ws_main(hub, FLASK_HOST, FLASK_PORT))
    except KeyboardInterrupt:
        pass
