"""Generate voice samples to demonstrate Askme TTS quality."""

import os
import sys
import time
import wave
import numpy as np

# Ensure project root on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_DIR = "models/tts/vits-zh-aishell3"
OUTPUT_DIR = "docs/voice_samples"

SAMPLES = [
    ("01_greeting", "你好，我是Thunder，穹沛的巡检机器人。等待指令。"),
    ("02_confirm_task", "收到，正在前往三号配电柜巡检。预计到达时间两分钟。"),
    ("03_estop", "已紧急停机。所有运动已暂停，等待下一步指令。"),
    ("04_safety_confirm", "这是高风险操作：移动机械臂到前方位置。请说确认执行继续，或取消放弃。"),
    ("05_patrol_report", "巡逻报告：三号区域检测到温度异常，柜体表面六十八度，超过阈值。已记录并上报工单。"),
    ("06_memory_demo", "根据之前的巡检记录，这个区域上周也出现过类似温度异常。建议排查散热系统。"),
    ("07_skill_response", "好的，正在为您搜索附近的充电桩。找到两个可用充电位，最近的在A区，距离约五十米。"),
]


def build_tts():
    import sherpa_onnx

    model_file = os.path.join(MODEL_DIR, "vits-aishell3.onnx")
    lexicon = os.path.join(MODEL_DIR, "lexicon.txt")
    tokens = os.path.join(MODEL_DIR, "tokens.txt")

    rule_fsts = []
    for name in ("date.fst", "number.fst", "phone.fst", "new_heteronym.fst"):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            rule_fsts.append(path)

    rule_fars = []
    for name in ("rule.far",):
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            rule_fars.append(path)

    config = sherpa_onnx.OfflineTtsConfig(
        model=sherpa_onnx.OfflineTtsModelConfig(
            vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                model=model_file,
                lexicon=lexicon,
                tokens=tokens,
            ),
            num_threads=4,
            provider="cpu",
        ),
        rule_fsts=",".join(rule_fsts),
        rule_fars=",".join(rule_fars),
        max_num_sentences=1,
    )
    return sherpa_onnx.OfflineTts(config)


def save_wav(filepath, samples, sample_rate):
    samples_int16 = (np.clip(samples, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(filepath, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(samples_int16.tobytes())


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("Loading VITS-Aishell3 model...")
    t0 = time.time()
    tts = build_tts()
    print(f"Model loaded in {time.time() - t0:.1f}s\n")

    # Try different speakers for variety
    speakers = [0, 10, 33, 66]

    for sid in speakers:
        sid_dir = os.path.join(OUTPUT_DIR, f"speaker_{sid}")
        os.makedirs(sid_dir, exist_ok=True)

        print(f"=== Speaker {sid} ===")
        for filename, text in SAMPLES:
            t1 = time.time()
            audio = tts.generate(text, sid=sid, speed=1.0)
            elapsed = time.time() - t1
            samples = np.array(audio.samples, dtype=np.float32)
            duration = len(samples) / audio.sample_rate

            out_path = os.path.join(sid_dir, f"{filename}.wav")
            save_wav(out_path, samples, audio.sample_rate)

            print(f"  {filename}: {duration:.1f}s audio, synth {elapsed:.2f}s, RTF={elapsed/duration:.2f}")

        print()

    print(f"Done! Samples saved to {OUTPUT_DIR}/")
    print(f"Total speakers: {len(speakers)}, samples per speaker: {len(SAMPLES)}")


if __name__ == "__main__":
    main()
