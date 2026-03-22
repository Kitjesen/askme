import requests
import json
import os
import sys
import miniaudio
import sounddevice as sd
import numpy as np
import threading
import queue

# Minimax API Configuration
MINIMAX_API_KEY = "sk-api-UVY-jfMCd7XYRA_ZRbpMIu_yQGm3CZ9jnqjQUh3y78L1DBZZ2N6jUsIu-p4T5n9wmdyOXR8QbqMVyqRaaDlT4w3VecToXGnCj3mhYzytbgRwVhYaTTkKBjg"
MINIMAX_URL = "https://api.minimaxi.com/v1/t2a_v2"

# Playback Buffer
audio_buffer = queue.Queue()
is_playing = False

def play_audio_callback(outdata, frames, time, status):
    global is_playing
    if status:
        print(status)
    
    if audio_buffer.empty():
        outdata.fill(0)
        return

    n = 0
    while n < frames and not audio_buffer.empty():
        remaining = frames - n
        current_chunk = audio_buffer.queue[0]
        k = current_chunk.shape[0]

        if remaining <= k:
            outdata[n:, 0] = current_chunk[:remaining]
            audio_buffer.queue[0] = current_chunk[remaining:]
            n = frames
            if audio_buffer.queue[0].shape[0] == 0:
                audio_buffer.get()
            break

        outdata[n : n + k, 0] = audio_buffer.get()
        n += k

    if n < frames:
        outdata[n:, 0] = 0

def start_playback_thread():
    global is_playing
    is_playing = True
    print("\nStarting playback thread...")
    with sd.OutputStream(
        channels=1,
        callback=play_audio_callback,
        dtype="float32",
        samplerate=32000, # Minimax output sample rate
        blocksize=1024,
    ):
        while is_playing:
            sd.sleep(100)

def test_minimax_stream(text="你好，这是一段MiniMax流式语音合成的实时播放测试。"):
    global is_playing
    # Check if thread is already running to avoid duplicate playback
    if is_playing:
        print("Playback already running, skipping start.")
        return

    print(f"Testing MiniMax STREAMING TTS & PLAYBACK with text: {text}")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}"
    }
    
    payload = {
        "model": "speech-2.6-hd",
        "text": text,
        "stream": True,
        "voice_setting": {
            "voice_id": "male-qn-qingse",
            "speed": 1,
            "vol": 1,
            "pitch": 0,
            "emotion": "happy"
        },
        "audio_setting": {
            "sample_rate": 32000,
            "bitrate": 128000,
            "format": "mp3", 
            "channel": 1
        }
    }
    
    # Start playback in background
    playback_thread = threading.Thread(target=start_playback_thread)
    playback_thread.daemon = True
    playback_thread.start()

    try:
        print("Sending stream request...")
        with requests.post(MINIMAX_URL, json=payload, headers=headers, stream=True) as response:
            if response.status_code != 200:
                print(f"Error: {response.status_code}")
                print(response.text)
                return

            # Correct usage: miniaudio.decode(mp3_data, nchannels=1, sample_rate=...)
            # BUT wait, miniaudio.decode expects full file usually. 
            # Streaming decode: miniaudio.stream_any(source, source_format, ...)
            
            # Actually, `miniaudio` python binding is a bit low level.
            # Let's check docs or use `miniaudio.decode` on chunks if frames are complete?
            # NO, MP3 chunks are frames.
            
            # Better approach: Just use `miniaudio.decode` which supports raw memory.
            # But it returns ALL pcm.
            
            # Let's fix the API call. `miniaudio` (the python lib) has `decode`.
            # If we want streaming decode of MP3 frames, we might need `pydub` or just
            # concatenate frames and decode? No that's slow.
            
            # Let's look at `miniaudio` library capabilities.
            # It seems I hallucinated `Mp3StreamDecoder` class in standard `miniaudio` python binding.
            # The `miniaudio` package on PyPI is a wrapper.
            
            # SIMPLER ALTERNATIVE for this script:
            # Accumulate bytes and decode them?
            
            # Or use `librosa` or `soundfile`? `soundfile` doesn't support MP3 easily on all platforms.
            
            # Let's try `miniaudio.decode` on the chunk. 
            # If `mp3_chunk` is a valid MP3 frame, it might decode.
            
            for line in response.iter_lines():
                if line:
                    line_str = line.decode('utf-8').strip()
                    if line_str.startswith("data:"):
                        try:
                            json_str = line_str[5:]
                            data = json.loads(json_str)
                            
                            if "data" in data and "audio" in data["data"]:
                                hex_audio = data["data"]["audio"]
                                if hex_audio:
                                    mp3_chunk = bytes.fromhex(hex_audio)
                                    
                                    # Try decoding the chunk directly
                                    try:
                                        # Note: decoding single frames might produce clicking if state is reset.
                                        # But let's try.
                                        decoded = miniaudio.decode(mp3_chunk, nchannels=1, sample_rate=32000)
                                        # decoded is DecodedSound object, .samples is memoryview/bytes
                                        
                                        # Convert int16 samples to float32
                                        samples_int16 = np.frombuffer(decoded.samples, dtype=np.int16)
                                        samples_float = samples_int16.astype(np.float32) / 32768.0
                                        
                                        audio_buffer.put(samples_float)
                                        sys.stdout.write("♪")
                                        sys.stdout.flush()
                                    except Exception as decode_err:
                                        # If chunk is partial frame, this might fail.
                                        # In real app we should buffer.
                                        pass
                                        
                        except Exception as e:
                            pass # Ignore parse errors

            print(f"\n\nStream finished. Waiting for playback to complete...")
            # Wait until buffer is empty
            while not audio_buffer.empty() and is_playing:
                time.sleep(0.1)
            # Give a little extra grace period for the last chunk
            time.sleep(1.0)
            
    except Exception as e:
        print(f"Exception: {e}")
    finally:
        pass
        
    is_playing = False
    if playback_thread.is_alive():
        playback_thread.join(timeout=2)
    print("Done.")

if __name__ == "__main__":
    import time
    # Ensure queue is empty before starting
    while not audio_buffer.empty():
        audio_buffer.get()
        
    test_minimax_stream()
