import asyncio
import os
import sys
import io
import threading
import queue
import time
import re
import numpy as np
import sounddevice as sd
import sherpa_onnx
from openai import AsyncOpenAI
import urllib.request
import tarfile
import requests
import json
import miniaudio
import datetime

# Fix Windows console encoding for Chinese output
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add memU to path for importing MemoryService
sys.path.insert(0, os.path.abspath("memU/src"))
from memu.app import MemoryService

# ==========================================
# Task & Dialogue Management
# ==========================================

class ToolRegistry:
    def __init__(self):
        self.tools = {
            "get_current_time": self.get_current_time,
            "run_command": self.run_command,
            "read_file": self.read_file,
            "list_directory": self.list_directory,
        }
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "获取当前系统时间",
                    "parameters": {"type": "object", "properties": {}}
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "run_command",
                    "description": "在本地执行一条 shell 命令并返回输出（最多 2000 字符）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "command": {"type": "string", "description": "要执行的命令"}
                        },
                        "required": ["command"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "read_file",
                    "description": "读取本地文件内容（最多 3000 字符）",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "文件路径"}
                        },
                        "required": ["path"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_directory",
                    "description": "列出目录中的文件和文件夹",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "目录路径，默认当前目录"}
                        }
                    }
                }
            },
        ]

    def get_current_time(self):
        return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def run_command(self, command):
        import subprocess
        try:
            result = subprocess.run(
                command, shell=True, capture_output=True, text=True, timeout=10
            )
            output = result.stdout or result.stderr or "(no output)"
            return output[:2000]
        except subprocess.TimeoutExpired:
            return "Command timed out (10s)"
        except Exception as e:
            return f"Error: {e}"

    def read_file(self, path):
        try:
            with open(path, 'r', encoding='utf-8', errors='replace') as f:
                return f.read(3000)
        except Exception as e:
            return f"Error reading file: {e}"

    def list_directory(self, path="."):
        try:
            entries = os.listdir(path or ".")
            return "\n".join(entries[:50])
        except Exception as e:
            return f"Error: {e}"

    def execute(self, name, args_json):
        if name in self.tools:
            try:
                args = json.loads(args_json) if args_json else {}
                print(f"[Tool] Executing {name}...")
                return self.tools[name](**args) if args else self.tools[name]()
            except Exception as e:
                return f"Error executing tool: {e}"
        return "Tool not found"

class ConversationManager:
    HISTORY_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'conversation_history.json')

    def __init__(self, max_history=20):
        self.history = []
        self.max_history = max_history
        self._load()

    def add_user_message(self, content):
        self.history.append({"role": "user", "content": content})
        self._trim()
        self._save()

    def add_assistant_message(self, content):
        self.history.append({"role": "assistant", "content": content})
        self._trim()
        self._save()

    def _trim(self):
        # Remove tool_calls/tool messages during trim to keep only user/assistant
        clean = [m for m in self.history if m.get("role") in ("user", "assistant") and m.get("content")]
        if len(clean) > self.max_history:
            clean = clean[-self.max_history:]
        self.history = clean

    def get_messages(self, system_prompt):
        return [{"role": "system", "content": system_prompt}] + self.history

    def clear(self):
        self.history = []
        self._save()

    def _save(self):
        try:
            os.makedirs(os.path.dirname(self.HISTORY_FILE), exist_ok=True)
            with open(self.HISTORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.history, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load(self):
        try:
            if os.path.exists(self.HISTORY_FILE):
                with open(self.HISTORY_FILE, 'r', encoding='utf-8') as f:
                    self.history = json.load(f)
        except Exception:
            self.history = []

# ==========================================
# Configuration (from .env or fallback)
# ==========================================
from dotenv import load_dotenv

# Load .env from project root
_env_path = os.path.join(os.path.dirname(__file__), '..', '.env')
load_dotenv(_env_path)

# 1. Brain: DeepSeek (Thinking Mode)
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
DEEPSEEK_BASE_URL = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")

# 2. Memory: Local Embedding Server
LOCAL_EMBED_URL = os.getenv("LOCAL_EMBED_URL", "http://localhost:8000/v1")

# 3. Ears: Sherpa-ONNX Models (ASR only now)
ASR_MODEL_DIR = "models/asr/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20"

# 4. Mouth: MiniMax TTS API
MINIMAX_API_KEY = os.getenv("MINIMAX_API_KEY", "")
MINIMAX_GROUP_ID = os.getenv("MINIMAX_GROUP_ID", "")
MINIMAX_URL = "https://api.minimaxi.com/v1/t2a_v2"

# 5. TTS Voice Settings
TTS_VOICE_ID = os.getenv("TTS_VOICE_ID", "male-qn-qingse")
TTS_SPEED = float(os.getenv("TTS_SPEED", "1"))
TTS_EMOTION = os.getenv("TTS_EMOTION", "happy")

# ==========================================
# Initialize Clients
# ==========================================

# DeepSeek Client
deepseek_client = AsyncOpenAI(
    api_key=DEEPSEEK_API_KEY,
    base_url=DEEPSEEK_BASE_URL,
)

# MemU Service — only init if embedding server is reachable
memory_service = None
def _check_embedding_server():
    try:
        import urllib.request
        urllib.request.urlopen(LOCAL_EMBED_URL.rstrip('/') + '/models', timeout=1)
        return True
    except Exception:
        return False

if _check_embedding_server():
    print("Initializing Memory Service...")
    try:
        memory_service = MemoryService(
            llm_profiles={
                "default": {
                    "api_key": DEEPSEEK_API_KEY,
                    "base_url": DEEPSEEK_BASE_URL,
                    "chat_model": "deepseek-chat",
                    "client_backend": "sdk",
                },
                "embedding": {
                    "api_key": "sk-dummy",
                    "base_url": LOCAL_EMBED_URL,
                    "embed_model": "all-MiniLM-L6-v2",
                }
            },
            retrieve_config={"method": "rag"}
        )
    except Exception as e:
        print(f"Warning: Memory Service init failed: {e}")
        memory_service = None
else:
    print("[Memory] Embedding server not running, memory disabled.")

# ==========================================
# Helper: Model Downloader
# ==========================================
def check_and_download_models():
    if not os.path.exists("models"):
        os.makedirs("models")
        
    # ASR Model
    if not os.path.exists(ASR_MODEL_DIR):
        print("Downloading ASR model...")
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2"
        filename = "asr_model.tar.bz2"
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:bz2") as tar:
            tar.extractall(path="models")
        os.rename("models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20", ASR_MODEL_DIR)
        os.remove(filename)

    # VAD Model (Silero VAD)
    VAD_MODEL_DIR = "models/vad"
    VAD_MODEL_PATH = f"{VAD_MODEL_DIR}/silero_vad.onnx"
    if not os.path.exists(VAD_MODEL_PATH):
        print("Downloading VAD model...")
        if not os.path.exists(VAD_MODEL_DIR):
            os.makedirs(VAD_MODEL_DIR)
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
        urllib.request.urlretrieve(url, VAD_MODEL_PATH)

    # KWS Model (Keyword Spotting / Wake Word)
    # Using a pre-trained Chinese wake word model "你好小智" (ni hao xiao zhi) as an example
    # You can train your own custom wake word later.
    # For now we use a generic Chinese KWS model if available, or just skip if no easy pre-trained one matches "Jarvis"
    # Let's use 'sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01' which supports custom keywords!
    KWS_MODEL_DIR = "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
    if not os.path.exists(KWS_MODEL_DIR):
        print("Downloading KWS model...")
        url = "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01.tar.bz2"
        filename = "kws_model.tar.bz2"
        urllib.request.urlretrieve(url, filename)
        with tarfile.open(filename, "r:bz2") as tar:
            tar.extractall(path="models/kws") # Assuming models/kws exists or Tar extracts it
        if not os.path.exists("models/kws"): os.makedirs("models/kws")
        # The tar usually contains the folder name
        if os.path.exists("sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"):
             import shutil
             shutil.move("sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01", "models/kws/")
        elif os.path.exists("models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"):
            pass # Already there
        # Cleanup
        if os.path.exists(filename): os.remove(filename)

    print("Models checked.")

# ==========================================
# Audio System
# ==========================================

class AudioAgent:
    def __init__(self, voice_mode=True):
        self.conversation = ConversationManager()
        self.tools = ToolRegistry()
        self.is_listening = False
        self.audio_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.tts_buffer = queue.Queue()
        self.is_playing = False
        self.stream_play_event = threading.Event()
        self.woken_up = False

        # Only init ASR/KWS/VAD in voice mode (saves ~2s startup)
        if voice_mode:
            check_and_download_models()
            self.init_kws()
            self.init_asr()
        else:
            self.kws_spotter = None
            self.asr_recognizer = None
            self.asr_stream = None
            self.vad = None
            self.woken_up = True

        self.init_tts()

    def init_kws(self):
        print("Initializing KWS (Wake Word)...")
        repo_dir = "models/kws/sherpa-onnx-kws-zipformer-wenetspeech-3.3M-2024-01-01"
        if not os.path.exists(repo_dir):
            print("Warning: KWS model not found, skipping wake word.")
            self.kws_spotter = None
            self.woken_up = True # Fallback: always awake
            return
            
        # Create a keywords file if not exists
        keywords_file = f"{repo_dir}/keywords.txt"
        if not os.path.exists(keywords_file):
            # Format: id @keyword
            with open(keywords_file, "w", encoding="utf-8") as f:
                f.write("你好 @你好\n")
                f.write("小智 @小智\n")

        self.kws_spotter = sherpa_onnx.KeywordSpotter(
            tokens=f"{repo_dir}/tokens.txt",
            encoder=f"{repo_dir}/encoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            decoder=f"{repo_dir}/decoder-epoch-12-avg-2-chunk-16-left-64.onnx",
            joiner=f"{repo_dir}/joiner-epoch-12-avg-2-chunk-16-left-64.onnx",
            num_threads=1,
            keywords_file=keywords_file, # Mandatory argument
        )
        self.kws_stream = self.kws_spotter.create_stream()

    def init_asr(self):
        print("Initializing ASR & VAD...")
        tokens = f"{ASR_MODEL_DIR}/tokens.txt"
        encoder = f"{ASR_MODEL_DIR}/encoder-epoch-99-avg-1.int8.onnx"
        decoder = f"{ASR_MODEL_DIR}/decoder-epoch-99-avg-1.int8.onnx"
        joiner = f"{ASR_MODEL_DIR}/joiner-epoch-99-avg-1.int8.onnx"
        
        # Check if files exist, fallback to float32 if int8 not found
        if not os.path.exists(encoder):
             encoder = f"{ASR_MODEL_DIR}/encoder-epoch-99-avg-1.onnx"
             decoder = f"{ASR_MODEL_DIR}/decoder-epoch-99-avg-1.onnx"
             joiner = f"{ASR_MODEL_DIR}/joiner-epoch-99-avg-1.onnx"

        self.asr_recognizer = sherpa_onnx.OnlineRecognizer.from_transducer(
            tokens=tokens,
            encoder=encoder,
            decoder=decoder,
            joiner=joiner,
            num_threads=1,
            sample_rate=16000,
            feature_dim=80,
            enable_endpoint_detection=True,
            rule1_min_trailing_silence=2.4,
            rule2_min_trailing_silence=1.2,
            rule3_min_utterance_length=float("inf"),
        )
        self.asr_stream = self.asr_recognizer.create_stream()

        # Initialize VAD
        vad_config = sherpa_onnx.VadModelConfig()
        vad_config.silero_vad.model = "models/vad/silero_vad.onnx"
        vad_config.silero_vad.threshold = 0.5
        vad_config.silero_vad.min_silence_duration = 0.5
        vad_config.silero_vad.min_speech_duration = 0.25
        vad_config.sample_rate = 16000
        self.vad = sherpa_onnx.VoiceActivityDetector(vad_config, buffer_size_in_seconds=30)


    def init_tts(self):
        print("Initializing TTS (MiniMax Cloud)...")
        self.tts_text_queue = queue.Queue()
        # Start TTS Worker
        self.tts_worker_thread = threading.Thread(target=self.tts_loop, daemon=True)
        self.tts_worker_thread.start()
    
    def tts_loop(self):
        """Worker thread that consumes text and generates audio chunks"""
        while True:
            text = self.tts_text_queue.get()
            if text is None: 
                break
            try:
                self._generate_minimax_audio(text)
            except Exception as e:
                print(f"TTS Worker Error: {e}")
            finally:
                self.tts_text_queue.task_done()

    def _generate_minimax_audio(self, text, retries=2):
        """Call MiniMax TTS API with streaming, accumulate MP3 chunks for better decode."""
        print(f"[Mouth Stream]: {text}")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {MINIMAX_API_KEY}"
        }
        payload = {
            "model": "speech-2.6-hd",
            "text": text,
            "stream": True,
            "voice_setting": {
                "voice_id": TTS_VOICE_ID,
                "speed": TTS_SPEED,
                "vol": 1,
                "pitch": 0,
                "emotion": TTS_EMOTION
            },
            "audio_setting": {
                "sample_rate": 32000,
                "bitrate": 128000,
                "format": "mp3",
                "channel": 1
            }
        }

        for attempt in range(retries + 1):
            try:
                with requests.post(MINIMAX_URL, json=payload, headers=headers,
                                   stream=True, timeout=15) as response:
                    if response.status_code != 200:
                        print(f"MiniMax Error ({response.status_code}): {response.text[:200]}")
                        if attempt < retries:
                            time.sleep(0.5)
                            continue
                        return

                    # Accumulate MP3 bytes and batch-decode for cleaner audio
                    mp3_acc = bytearray()
                    chunk_count = 0

                    for line in response.iter_lines():
                        if not line:
                            continue
                        line_str = line.decode('utf-8')
                        if not line_str.startswith("data:"):
                            continue
                        try:
                            data_json = json.loads(line_str[5:])
                            hex_audio = data_json.get("data", {}).get("audio", "")
                            if hex_audio:
                                mp3_acc.extend(bytes.fromhex(hex_audio))
                                chunk_count += 1

                                # Decode every ~4 chunks for smooth playback
                                if chunk_count % 4 == 0 and len(mp3_acc) > 0:
                                    self._decode_and_queue(bytes(mp3_acc))
                                    mp3_acc.clear()
                        except (json.JSONDecodeError, ValueError):
                            pass

                    # Decode remaining accumulated audio
                    if mp3_acc:
                        self._decode_and_queue(bytes(mp3_acc))

                    return  # Success

            except requests.exceptions.Timeout:
                print(f"MiniMax timeout (attempt {attempt + 1}/{retries + 1})")
            except requests.exceptions.ConnectionError:
                print(f"MiniMax connection error (attempt {attempt + 1}/{retries + 1})")
            except Exception as e:
                print(f"MiniMax Request Error: {e}")
                return

            if attempt < retries:
                time.sleep(0.5 * (attempt + 1))

    def _decode_and_queue(self, mp3_data):
        """Decode MP3 bytes to float32 PCM and queue for playback."""
        try:
            decoded = miniaudio.decode(mp3_data, nchannels=1, sample_rate=32000)
            samples = np.frombuffer(decoded.samples, dtype=np.int16).astype(np.float32) / 32768.0
            self.tts_buffer.put(samples)
        except Exception:
            pass  # Skip corrupted chunks silently

    def play_audio_callback(self, outdata, frames, time, status):
        """Callback for SoundDevice output stream"""
        if status:
            print(status)
        
        # If nothing to play, fill silence
        if self.tts_buffer.empty():
            outdata.fill(0)
            return

        n = 0
        while n < frames and not self.tts_buffer.empty():
            remaining = frames - n
            current_chunk = self.tts_buffer.queue[0]
            k = current_chunk.shape[0]

            if remaining <= k:
                outdata[n:, 0] = current_chunk[:remaining]
                self.tts_buffer.queue[0] = current_chunk[remaining:]
                n = frames
                if self.tts_buffer.queue[0].shape[0] == 0:
                    self.tts_buffer.get()
                break

            outdata[n : n + k, 0] = self.tts_buffer.get()
            n += k

        if n < frames:
            outdata[n:, 0] = 0

    def start_playback_session(self):
        """Starts the audio output stream"""
        self.is_playing = True
        self.playback_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.playback_thread.start()

    def _playback_loop(self):
        try:
            with sd.OutputStream(
                channels=1,
                callback=self.play_audio_callback,
                dtype="float32",
                samplerate=32000,
                blocksize=1024,
            ):
                while self.is_playing:
                    sd.sleep(100)
        except Exception as e:
            print(f"Playback error: {e}")

    def stop_playback_session(self):
        """Stops the audio output stream"""
        self.is_playing = False
        if hasattr(self, 'playback_thread') and self.playback_thread.is_alive():
            self.playback_thread.join(timeout=1.0)

    def speak_stream(self, text):
        """Queue text for TTS generation (strips emoji and markdown)"""
        if text:
            clean = text
            # Remove emoji and special unicode symbols
            clean = re.sub(r'[\U00010000-\U0010ffff]', '', clean)
            # Remove markdown formatting
            clean = re.sub(r'\*\*(.+?)\*\*', r'\1', clean)  # **bold**
            clean = re.sub(r'\*(.+?)\*', r'\1', clean)      # *italic*
            clean = re.sub(r'`(.+?)`', r'\1', clean)        # `code`
            clean = re.sub(r'^#+\s*', '', clean, flags=re.MULTILINE)  # # headers
            clean = re.sub(r'^[-*]\s+', '', clean, flags=re.MULTILINE)  # - list items
            clean = re.sub(r'!\[.*?\]\(.*?\)', '', clean)   # ![img](url)
            clean = re.sub(r'\[(.+?)\]\(.*?\)', r'\1', clean)  # [link](url)
            clean = clean.strip()
            if clean:
                self.tts_text_queue.put(clean)
            
    def wait_speaking_done(self):
        """Blocks until all text is processed and audio played"""
        self.tts_text_queue.join() # Wait for all text to be converted to audio
        while not self.tts_buffer.empty(): # Wait for audio to be played
            time.sleep(0.1)
        time.sleep(0.5) # Extra grace for last chunk latency

    def listen_loop(self):
        """Listen from microphone with VAD-gated ASR for reduced false triggers."""
        print("\n[Ear] Listening... (Press Ctrl+C to stop)")
        sample_rate = 16000
        samples_per_read = int(0.1 * sample_rate)  # 100ms chunks
        speech_active = False

        with sd.InputStream(channels=1, dtype="float32", samplerate=sample_rate) as s:
            while not self.stop_event.is_set():
                samples, _ = s.read(samples_per_read)
                samples = samples.reshape(-1)

                # Feed VAD with int16 samples
                samples_int16 = (samples * 32768).astype(np.int16)
                self.vad.accept_waveform(samples_int16)

                # Only feed ASR when VAD detects speech
                if self.vad.is_speech_detected():
                    if not speech_active:
                        speech_active = True
                        sys.stdout.write("🎤")
                        sys.stdout.flush()

                    self.asr_stream.accept_waveform(sample_rate, samples)

                    while self.asr_recognizer.is_ready(self.asr_stream):
                        self.asr_recognizer.decode_stream(self.asr_stream)
                else:
                    if speech_active:
                        # Speech just ended — check ASR result
                        speech_active = False
                        # Feed remaining to ASR and check
                        self.asr_stream.accept_waveform(sample_rate, samples)
                        while self.asr_recognizer.is_ready(self.asr_stream):
                            self.asr_recognizer.decode_stream(self.asr_stream)

                # Check for endpoint (works both during and after speech)
                is_endpoint = self.asr_recognizer.is_endpoint(self.asr_stream)
                text = self.asr_recognizer.get_result(self.asr_stream)

                if is_endpoint and text:
                    text = text.strip()
                    if len(text) > 0:
                        print(f"\n[User]: {text}")
                        self.audio_queue.put(text)
                        self.asr_recognizer.reset(self.asr_stream)
                        self.asr_stream = self.asr_recognizer.create_stream()
                        return text

# ==========================================
# Main Logic
# ==========================================

async def _stream_and_speak(agent, response):
    """Stream DeepSeek response, split into sentences, and send to TTS."""
    full_response = ""
    sentence_buffer = ""

    async for chunk in response:
        delta = chunk.choices[0].delta

        if delta.content:
            content = delta.content
            sys.stdout.write(content)
            sys.stdout.flush()

            full_response += content
            sentence_buffer += content

            # Sentence splitting strategy
            should_speak = False

            # Rule 1: Strong punctuation - always split
            if any(p in content for p in ['.', '?', '!', '。', '？', '！', '\n']):
                should_speak = True
            # Rule 2: First sentence - split aggressively on comma after 5 chars
            elif len(full_response) < 40 and (',' in content or '，' in content):
                if len(sentence_buffer) > 5:
                    should_speak = True
            # Rule 3: Normal - split on comma after 15 chars
            elif len(sentence_buffer) > 15 and (',' in content or '，' in content):
                should_speak = True
            # Rule 4: Emergency split - prevent infinite buffering
            elif len(sentence_buffer) > 60:
                should_speak = True

            if should_speak and sentence_buffer.strip():
                agent.speak_stream(sentence_buffer.strip())
                sentence_buffer = ""

    print()

    # Speak any leftover text
    if sentence_buffer.strip():
        agent.speak_stream(sentence_buffer.strip())

    return full_response


async def brain_process(user_text, agent: AudioAgent):
    print(f"[Brain] Thinking...")

    # 1. Retrieve Memories (with Timeout)
    context_str = ""
    if memory_service:
        try:
            print(f"[Memory] Searching...")
            retrieval_task = memory_service.retrieve(
                queries=[{"role": "user", "content": {"text": user_text}}],
                where={"user_id": "user_1"}
            )
            retrieval = await asyncio.wait_for(retrieval_task, timeout=2.0)
            items = retrieval.get("items", [])
            if items:
                print(f"[Memory] Found {len(items)} items.")
                context_str = "\n".join([f"- {i['content']}" for i in items])
            else:
                print(f"[Memory] No relevant memories found.")
        except asyncio.TimeoutError:
            print(f"[Memory] Retrieval timed out (skipped).")
        except Exception as e:
            print(f"[Memory Error] {e}")

    # 2. Build system prompt (optimized for voice output)
    system_prompt = f"""你是一个有用的 AI 语音助手。用中文简洁口语化回答。
重要：回答会通过语音合成播放，所以：
- 不要使用 markdown 格式（不要用 **加粗**、*斜体*、`代码`、# 标题、- 列表）
- 用自然口语表达，像跟朋友聊天一样
- 列举时用"第一、第二"或"首先、其次"，不要用列表符号
- 保持简洁，每次回答控制在 100 字以内
记忆上下文:
{context_str if context_str else '无'}
"""

    agent.conversation.add_user_message(user_text)
    messages = agent.conversation.get_messages(system_prompt)

    # Start playback session early
    agent.start_playback_session()

    try:
        # Use deepseek-chat with native tool_choice for tool calls
        response = await deepseek_client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            tools=agent.tools.tool_definitions,
            tool_choice="auto",
            stream=True,
        )

        # Collect streaming response — handle both text and tool_calls
        full_response = ""
        sentence_buffer = ""
        tool_calls_acc = {}  # accumulate streamed tool call fragments

        async for chunk in response:
            delta = chunk.choices[0].delta

            # Accumulate tool call fragments
            if delta.tool_calls:
                for tc in delta.tool_calls:
                    idx = tc.index
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {"id": "", "name": "", "arguments": ""}
                    if tc.id:
                        tool_calls_acc[idx]["id"] = tc.id
                    if tc.function:
                        if tc.function.name:
                            tool_calls_acc[idx]["name"] = tc.function.name
                        if tc.function.arguments:
                            tool_calls_acc[idx]["arguments"] += tc.function.arguments

            # Normal text content
            if delta.content:
                content = delta.content
                sys.stdout.write(content)
                sys.stdout.flush()

                full_response += content
                sentence_buffer += content

                # Sentence splitting
                should_speak = False
                if any(p in content for p in ['.', '?', '!', '。', '？', '！', '\n']):
                    should_speak = True
                elif len(full_response) < 40 and (',' in content or '，' in content):
                    if len(sentence_buffer) > 5:
                        should_speak = True
                elif len(sentence_buffer) > 15 and (',' in content or '，' in content):
                    should_speak = True
                elif len(sentence_buffer) > 60:
                    should_speak = True

                if should_speak and sentence_buffer.strip():
                    agent.speak_stream(sentence_buffer.strip())
                    sentence_buffer = ""

        # Flush remaining text
        if sentence_buffer.strip():
            agent.speak_stream(sentence_buffer.strip())

        # Handle tool calls if any
        if tool_calls_acc:
            print(f"\n[Tool Calls] {len(tool_calls_acc)} detected")
            # Build assistant message with tool_calls for conversation
            tool_call_objs = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                print(f"  → {tc['name']}({tc['arguments']})")
                result = agent.tools.execute(tc["name"], tc["arguments"])
                print(f"  ← {result}")
                tool_call_objs.append({
                    "id": tc["id"],
                    "type": "function",
                    "function": {"name": tc["name"], "arguments": tc["arguments"]}
                })
                # Add tool result to conversation for follow-up
                agent.conversation.history.append({
                    "role": "assistant",
                    "content": None,
                    "tool_calls": tool_call_objs,
                })
                agent.conversation.history.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": str(result),
                })

            # Follow-up call with tool results to get natural language response
            follow_msgs = agent.conversation.get_messages(system_prompt)
            follow_response = await deepseek_client.chat.completions.create(
                model="deepseek-chat",
                messages=follow_msgs,
                stream=True,
            )
            full_response = await _stream_and_speak(agent, follow_response)

        print()

        # Update History
        agent.conversation.add_assistant_message(full_response)

        # Wait for all speech to finish
        await asyncio.to_thread(agent.wait_speaking_done)

    except Exception as e:
        print(f"DeepSeek Error: {e}")
    finally:
        agent.stop_playback_session()

async def main_voice():
    """Voice mode: microphone → ASR → Brain → TTS → speaker"""
    agent = AudioAgent(voice_mode=True)

    print("\n🎙️ Voice mode active. Say something! (Ctrl+C to quit)")
    consecutive_errors = 0
    while True:
        try:
            user_text = await asyncio.to_thread(agent.listen_loop)
            if user_text:
                consecutive_errors = 0
                await brain_process(user_text, agent)
        except KeyboardInterrupt:
            break
        except Exception as e:
            consecutive_errors += 1
            print(f"Error: {e}")
            if consecutive_errors >= 3:
                print("Too many consecutive errors, exiting.")
                break
            # Reset ASR stream on error
            try:
                agent.asr_stream = agent.asr_recognizer.create_stream()
            except Exception:
                pass
            await asyncio.sleep(1)

    agent.stop_event.set()
    agent.tts_text_queue.put(None)
    print("\nBye!")


async def main_text():
    """Text mode: keyboard input → Brain → TTS → speaker (no mic needed)"""
    agent = AudioAgent(voice_mode=False)

    print("\n⌨️ Text mode active. Commands: /clear (clear history), /quit (exit)")
    print(f"   Loaded {len(agent.conversation.history)} previous messages.\n")
    while True:
        try:
            user_text = await asyncio.to_thread(input, "[You]: ")
            user_text = user_text.strip()
            if not user_text:
                continue
            if user_text in ('/quit', '/exit', 'exit', 'quit'):
                break
            if user_text == '/clear':
                agent.conversation.clear()
                print("[System] Conversation history cleared.\n")
                continue
            if user_text == '/history':
                for msg in agent.conversation.history:
                    role = "You" if msg["role"] == "user" else "AI"
                    print(f"  [{role}]: {msg.get('content', '')[:80]}")
                print()
                continue
            await brain_process(user_text, agent)
            print()
        except (KeyboardInterrupt, EOFError):
            break
        except Exception as e:
            print(f"Error: {e}")

    agent.stop_event.set()
    agent.tts_text_queue.put(None)
    print("\nBye!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Voice Memory Agent")
    parser.add_argument("--text", action="store_true", help="Text input mode (no microphone)")
    args = parser.parse_args()

    if args.text:
        asyncio.run(main_text())
    else:
        asyncio.run(main_voice())
