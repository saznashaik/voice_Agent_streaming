# app.py
# Integrated: Flask UI (mic Start/Stop) + Relay WS + Voice Agent (AssemblyAI STT -> Gemini LLM -> Murf TTS)
# NOTE: Replace API keys via environment variables or .env for security.

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
import websocket                 # websocket-client (Assembly client)
import websockets                # websockets (async server & client)
from flask import Flask, render_template_string, jsonify,request

# Audio
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
FLASK_PORT = int(os.getenv("FLASK_PORT", "5000"))

RELAY_HOST = os.getenv("RELAY_HOST", "127.0.0.1")
RELAY_PORT = int(os.getenv("RELAY_PORT", "8787"))

SAMPLE_RATE_STT = 16000
FRAMES_PER_BUFFER_STT = 800   # ~50ms
CHANNELS_STT = 1
FORMAT_STT = pyaudio.paInt16

SAMPLE_RATE_TTS = 44100
CHANNELS_TTS = 1
FORMAT_TTS = pyaudio.paInt16

ASSEMBLY_WS_BASE = "wss://streaming.assemblyai.com/v3/ws"
ASSEMBLY_PARAMS = {"sample_rate": SAMPLE_RATE_STT, "format_turns": True}
ASSEMBLY_ENDPOINT = f"{ASSEMBLY_WS_BASE}?{urlencode(ASSEMBLY_PARAMS)}"

MURF_WS_URL = "wss://api.murf.ai/v1/speech/stream-input"

CHAT_HISTORY_FILE = "chat_history.jsonl"
RECORD_RAW_WAV = True






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
    """
    Calls Tavily Search and returns a short, LLM-ready context string.
    """
    try:
        tavily_key = user_keys["TAVILY_API_KEY"] or os.getenv("TAVILY_API_KEY")
        tavily_client = TavilyClient(api_key=tavily_key)

        resp = tavily_client.search(
            query=query,
            # optional kwargs supported by Tavily:
            # search_depth="basic",  # or "advanced"
            # topic="general",
            # time_range=None,       # "day" | "week" | "month" | "year"
            max_results=max_results,
            include_answer=include_answer,
            include_raw_content=False,
            include_images=False,
        )
        # Prefer the direct answer if present; backfill with top results’ snippets
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
# System prompt updated so assistant speaks AS Elsa (in-character).
# It is instructed to avoid verbatim copyrighted lyrics and to behave in-character.
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
    """
    WebSocket relay broadcasting to browser UI on ws://host:port.
    Receives nothing from browser (UI is read-only here) but will still accept connections.
    """
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

# ------------------------ Audio I/O ------------------------
class AudioRecorder:
    def __init__(self, sample_rate=SAMPLE_RATE_STT, channels=CHANNELS_STT, format=FORMAT_STT, frames_per_buffer=FRAMES_PER_BUFFER_STT):
        self.pa = None
        self.stream = None
        self.sample_rate = sample_rate
        self.channels = channels
        self.format = format
        self.frames_per_buffer = frames_per_buffer

    def open_input(self):
        # Lazily initialize PyAudio so Flask startup doesn't require audio device
        if self.pa is None:
            self.pa = pyaudio.PyAudio()
        if self.stream:
            return
        self.stream = self.pa.open(
            input=True,
            frames_per_buffer=self.frames_per_buffer,
            channels=self.channels,
            format=self.format,
            rate=self.sample_rate,
        )

    def read(self):
        if not self.stream:
            raise RuntimeError("Stream not opened")
        return self.stream.read(self.frames_per_buffer, exception_on_overflow=False)

    def close(self):
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
        print("AssemblyAI WS opened.")
        def stream_audio():
            print("Streaming microphone to AssemblyAI...")
            while not stop_event.is_set():
                try:
                    audio_data = self.audio.read()
                    if RECORD_RAW_WAV:
                        with recording_lock:
                            recorded_frames.append(audio_data)
                    ws.send(audio_data, websocket.ABNF.OPCODE_BINARY)
                except Exception as e:
                    print(f"[AssemblyAI] audio stream error: {e}")
                    break
            print("Stopped streaming to AssemblyAI.")
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
        except Exception as e:
            print(f"[AssemblyAI] on_message error: {e}")

    def _on_error(self, ws, error):
        print(f"[AssemblyAI] WS error: {error}")
        # mark stop to break audio thread
        stop_event.set()

    def _on_close(self, ws, code, msg):
        print(f"[AssemblyAI] WS closed: code={code}, msg={msg}")

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
                time.sleep(0.8)
            except Exception as e:
                print(f"[AssemblyAI] terminate send error: {e}")
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
            print("No audio data recorded.")
            return
        with wave.open(filename, 'wb') as wf:
            wf.setnchannels(channels)
            wf.setsampwidth(sampwidth)
            wf.setframerate(sample_rate)
            wf.writeframes(b"".join(frames))
        print(f"Saved microphone recording: {filename}")

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
        # persona prompt is fixed: Elsa in-character
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

    async def stream_tts_to_speakers(self, text_iterable):
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
                                wav_bytes = base64.b64decode(wav_b64)
                                play_bytes = wav_bytes
                                if first and len(play_bytes) > 44:
                                    play_bytes = play_bytes[44:]
                                    first = False
                                player.play_bytes(play_bytes)
                                if self.relay:
                                    self.relay.send({"type": "audio", "data": wav_b64})
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
        # Use user keys if available, else env fallback
        gen_key = user_keys["GEMINI_API_KEY"] or os.getenv("GEMINI_API_KEY")
        murf_key = user_keys["MURF_API_KEY"] or os.getenv("MURF_API_KEY")
        assembly_key = user_keys["ASSEMBLYAI_API_KEY"] or os.getenv("ASSEMBLYAI_API_KEY")
        tavily_key = user_keys["TAVILY_API_KEY"] or os.getenv("TAVILY_API_KEY")
        news_key = user_keys["NEWS_API_KEY"] or os.getenv("NEWS_API_KEY")

        # Configure Gemini
        genai.configure(api_key=gen_key)

        self.history_store = ChatHistory(CHAT_HISTORY_FILE)
        self.recorder = AudioRecorder()
        self.llm = GeminiLLM()
        self.relay = None

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


        # AssemblyAI callbacks will be set after attaching relay

    def attach_relay(self, relay: RelayServer):
        self.relay = relay
        self.murf.relay = relay
        self.assembly.on_begin = self.on_begin
        self.assembly.on_turn = self.on_turn
        self.assembly.on_termination = self.on_termination

    # Assembly callbacks
    def on_begin(self, data):
        sid = data.get("id")
        exp = data.get("expires_at")
        try:
            exp_dt = datetime.fromtimestamp(exp, tz=timezone.utc) if exp else None
        except Exception:
            exp_dt = None
        print(f"Session started: {sid}, expires at {exp_dt}")
        if self.relay:
            self.relay.send({"type": "status", "value": "session_started"})

    def on_turn(self, data):
        transcript = data.get("transcript", "") or ""
        formatted = data.get("turn_is_formatted", False)

        if not formatted:
            print(f"\rUser(partial): {transcript[:120]}", end="")
            return

        with self.busy_lock:
            if self.busy:
                print("[Agent] Busy; skipping overlapping turn.")
                return
            self.busy = True

        # Final turn
        print("\r" + " " * 100 + "\r", end="")
        print(f"User(final): {transcript}")
        self.history_store.append("user", transcript)
        self.llm.add_user(transcript)

        if self.relay:
            self.relay.send({"type": "final_user", "text": transcript})

        threading.Thread(target=self.process_user_turn, args=(transcript,), daemon=True).start()

    def on_termination(self, data):
        adur = data.get("audio_duration_seconds", 0)
        sdur = data.get("session_duration_seconds", 0)
        print(f"Session terminated: audio={adur:.2f}s, session={sdur:.2f}s")
        if self.relay:
            self.relay.send({"type": "status", "value": "session_closed"})

    def process_user_turn(self, user_text: str):
        if not user_text.strip():
            with self.busy_lock:
                self.busy = False
            return
        search_ctx = ""


        def needs_web_search(user_text: str) -> bool:
            q = user_text.lower().strip()
            # very simple heuristic; refine as needed
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
            # fetch headlines instead of generic Tavily search
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
                # stream to Murf -> speakers & relay
                await self.murf.stream_tts_to_speakers(text_iterable_from_queue())
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

    # control methods to be triggered by Flask endpoints
    def start_streaming(self):
        """
        Open server mic and start AssemblyAI streaming.
        """
        try:
            self.recorder.open_input()
        except Exception as e:
            print(f"[Agent] recorder.open_input error: {e}")
            raise

        try:
            # clear global stop flag so threads can run
            stop_event.clear()
            self.assembly.start()
            if self.relay:
                self.relay.send({"type": "status", "value": "streaming"})
        except Exception as e:
            print(f"[Agent] assembly.start error: {e}")
            raise

    def stop_streaming(self):
        """
        Stop AssemblyAI session and close server mic.
        """
        try:
            self.assembly.terminate_session()
        except Exception as e:
            print(f"[Agent] assembly.terminate error: {e}")
        try:
            # set stop_event to signal audio thread to stop
            stop_event.set()
            if RECORD_RAW_WAV:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                AssemblyAIRealtime.save_wav(f"mic_capture_{ts}.wav", recorded_frames, sample_rate=SAMPLE_RATE_STT)
        except Exception as e:
            print(f"[Agent] save_wav error: {e}")
        try:
            self.recorder.close()
        except Exception as e:
            print(f"[Agent] recorder.close error: {e}")
        if self.relay:
            self.relay.send({"type": "status", "value": "idle"})

# ------------------------ Flask UI (mic controls) ------------------------
# ------------------------ Flask UI (mic controls) ------------------------
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
      margin: 0;
      padding: 0;
      display: flex;
      justify-content: center;
      align-items: center;
      height: 100vh;
      overflow: hidden;
    }

    .container {
      width: 100%;
      max-width: 1000px;
      background: rgba(255, 255, 255, 0.85);
      border-radius: 16px;
      padding: 30px;
      box-shadow: 0 8px 20px rgba(0, 0, 0, 0.15);
      backdrop-filter: blur(8px);
      animation: fadeIn 1.2s ease-in-out;
    }

    @keyframes fadeIn {
      from { opacity: 0; transform: translateY(30px); }
      to { opacity: 1; transform: translateY(0); }
    }

    h1 {
      text-align: center;
      font-family: 'Great Vibes', cursive;
      font-size: 3em;
      color: #2e64a1;
      margin-bottom: 5px;
    }

    .avatar {
      display: flex;
      justify-content: center;
      margin-bottom: 10px;
    }

    .avatar img {
      width: 100px;
      height: 100px;
      border-radius: 50%;
      border: 3px solid #fff;
      box-shadow: 0 0 20px rgba(173, 216, 230, 0.8);
    }

    #chat {
      border: none;
      border-radius: 14px;
      height: 550px;
      overflow-y: auto;
      padding: 25px;
      margin-bottom: 20px;
      background: rgba(255, 255, 255, 0.65);
      box-shadow: inset 0 0 12px rgba(0, 0, 0, 0.08);
      font-size: 1.1rem;
    }

    .message {
      padding: 10px 14px;
      margin: 10px 0;
      border-radius: 12px;
      max-width: 80%;
      clear: both;
      position: relative;
      line-height: 1.4;
      animation: fadeInUp 0.4s ease-in-out;
    }

    @keyframes fadeInUp {
      from { opacity: 0; transform: translateY(20px); }
      to { opacity: 1; transform: translateY(0); }
    }

    .user {
      background: #d0f0ff;
      color: #1b4965;
      float: right;
      text-align: right;
      border-bottom-right-radius: 0;
    }

    .assistant {
      background: #edf6ff;
      color: #16324f;
      float: left;
      text-align: left;
      border-bottom-left-radius: 0;
    }

    .persona-label {
      font-size: 0.8em;
      color: #5c6f80;
      margin-bottom: 3px;
      display: block;
      font-style: italic;
    }

    .controls {
      text-align: center;
      margin-top: 15px;
    }

    button {
      padding: 12px 20px;
      margin: 8px;
      border: none;
      border-radius: 25px;
      background: #2e64a1;
      color: #fff;
      font-size: 16px;
      cursor: pointer;
      transition: all 0.3s ease;
    }

    button:hover {
      background: #3977c9;
      box-shadow: 0 0 10px rgba(46, 100, 161, 0.5);
    }

    button:disabled {
      background: #a3c2dd;
      cursor: not-allowed;
    }
     .snowflake {
    position: fixed;
    top: -10px;
    color: white;
    user-select: none;
    pointer-events: none;
    font-size: 12px;
    animation-name: fall;
    animation-timing-function: linear;
    animation-iteration-count: infinite;
  }

  @keyframes fall {
    0% { transform: translateY(-10px) rotate(0deg); }
    100% { transform: translateY(110vh) rotate(360deg); }
  }
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
  // Snowflake generator
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
    await fetch("/api/set_keys", {
      method:"POST",
      headers:{"Content-Type":"application/json"},
      body: JSON.stringify(keys)
    });
    alert("Keys saved! You can now start recording.");
    closeModal();
  }

    const RELAY_URL = "ws://{{ relay_host }}:{{ relay_port }}";
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

    let ws;
    function connect() {
      ws = new WebSocket(RELAY_URL);
      ws.onopen = () => console.log("WS connected");
      ws.onmessage = (ev) => {
        let obj = JSON.parse(ev.data);
        if (obj.type === "final_user") appendUser(obj.text);
        else if (obj.type === "final_assistant") appendAssistant(obj.text);
      };
      ws.onclose = () => setTimeout(connect, 1000);
    }
    connect();

    startBtn.onclick = async () => {
      startBtn.disabled = true;
      const res = await fetch("/api/start_stream", { method: "POST" });
      if (res.ok) stopBtn.disabled = false;
      else startBtn.disabled = false;
    };

    stopBtn.onclick = async () => {
      stopBtn.disabled = true;
      const res = await fetch("/api/stop_stream", { method: "POST" });
      if (res.ok) startBtn.disabled = false;
      else stopBtn.disabled = false;
    };
  </script>
</body>
</html>
"""

@app.route("/")
def index():
    return render_template_string(INDEX_HTML, relay_host=RELAY_HOST, relay_port=RELAY_PORT)

@app.route("/api/set_keys", methods=["POST"])
def api_set_keys():
    data = request.get_json(force=True)
    for k in user_keys:
        if k in data and data[k]:
            user_keys[k] = data[k]
    return jsonify({"status": "keys_updated"}), 200


# API endpoints to start/stop server-side mic streaming
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

# ------------------------ Run App ------------------------
if __name__ == "__main__":
    # Start relay
    relay = RelayServer(host=RELAY_HOST, port=RELAY_PORT)
    relay.start()

    # Start agent and attach relay
    agent = VoiceAgent(persona_name="elsa")
    agent.attach_relay(relay)

    # Start Flask in a background thread (so main thread remains free)
    def run_flask():
        app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    print(f"Flask UI available at http://{FLASK_HOST}:{FLASK_PORT} (use Start Recording to stream mic to AssemblyAI)")

    # Keep main thread alive; user will control streaming via UI
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("Shutting down...")
    finally:
        try:
            agent.stop_streaming()
        except:
            pass
        try:
            relay.stop()
        except:
            pass
        print("Exited.")

