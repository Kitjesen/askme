import urllib.request
import os
import tarfile

def download_and_extract(url, dest_dir):
    filename = url.split('/')[-1]
    filepath = os.path.join(dest_dir, filename)
    
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)
        
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, filepath)
    
    print(f"Extracting {filename}...")
    with tarfile.open(filepath, "r:bz2") as tar:
        tar.extractall(path=dest_dir)
    
    # Clean up archive
    os.remove(filepath)
    print("Done.")

if __name__ == "__main__":
    # ASR Model
    download_and_extract(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-streaming-zipformer-bilingual-zh-en-2023-02-20.tar.bz2",
        "models/asr"
    )
    
    # TTS Model (Better Quality & Bilingual)
    # Using 'vits-mely-zh-en' or similar if available, or 'sherpa-onnx-vits-zh-ll' for better Chinese
    # Let's switch to 'sherpa-onnx-vits-zh-ll' which is a large model with better quality
    # OR 'vits-piper-en_US-amy-low' for English if user speaks English primarily
    
    # Since we want Chinese/English support and better quality:
    # We will download a high-quality Bilingual model if possible, 
    # but 'vits-zh-aishell3' is known to be robotic.
    
    # Let's try 'sherpa-onnx-vits-zh-ll' (Better Chinese, maybe supports some English?)
    # URL: https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2
    
    download_and_extract(
        "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/sherpa-onnx-vits-zh-ll.tar.bz2",
        "models/tts"
    )
