import requests
import json
import os
import sys

# Minimax API Configuration
MINIMAX_API_KEY = "sk-api-UVY-jfMCd7XYRA_ZRbpMIu_yQGm3CZ9jnqjQUh3y78L1DBZZ2N6jUsIu-p4T5n9wmdyOXR8QbqMVyqRaaDlT4w3VecToXGnCj3mhYzytbgRwVhYaTTkKBjg"
MINIMAX_GROUP_ID = "1880126057781301267"
MINIMAX_URL = "https://api.minimaxi.com/v1/t2a_v2"

def test_minimax_tts(text="楚天罡真牛逼"):
    print(f"Testing MiniMax TTS with text: {text}")
    print("Sending request...")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {MINIMAX_API_KEY}"
    }
    
    # Payload for NON-STREAMING request first to verify basic connectivity and format
    # Using 'mp3' as format which is standard
    payload = {
        "model": "speech-2.6-hd",
        "text": text,
        "stream": False, 
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
    
    try:
        response = requests.post(MINIMAX_URL, json=payload, headers=headers)
        
        if response.status_code == 200:
            print("Request successful!")
            data = response.json()
            
            if "data" in data and "audio" in data["data"]:
                hex_audio = data["data"]["audio"]
                print(f"Received audio data length: {len(hex_audio)} chars (hex)")
                
                # Convert hex to bytes
                audio_bytes = bytes.fromhex(hex_audio)
                filename = "test_minimax_output.mp3"
                
                with open(filename, "wb") as f:
                    f.write(audio_bytes)
                print(f"Audio saved to {filename}")
                print(f"Please try playing '{filename}' to verify audio content.")
                
            else:
                print("Error: No audio data in response.")
                print(f"Full response: {data}")
        else:
            print(f"Request failed with status code: {response.status_code}")
            print(f"Response text: {response.text}")
            
    except Exception as e:
        print(f"Exception occurred: {e}")

if __name__ == "__main__":
    test_minimax_tts()
