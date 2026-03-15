# Askme 模块化重构计划

> 目标：每个模块独立开关、独立测试、独立调试。改一个不影响整条链路。
> 教训：今天一次推了 15 个文件到 sunrise，VAD/ASR/TTS/LLM 全耦合在 listen_loop 里，一个参数改坏整条链路无法定位。

---

## 现状问题

### 1. audio_agent.py 是巨型上帝类（900+ 行）
- MIC 采集、VAD、ASR、KWS、TTS、chime、噪声门、回声门、降噪、云端 ASR、地址检测——**全塞在一个文件里**
- listen_loop() 单个方法 350+ 行，改任何一处都可能影响全部

### 2. 配置耦合
- `noise_gate_peak` 影响 ASR（阻止噪声帧进入 ASR）
- `echo_gate_peak` 影响 VAD（TTS 播放时屏蔽麦克风）
- `MIC 增益`（ALSA 硬件）影响所有阈值
- 改一个参数需要重调所有参数

### 3. 调试困难
- 只有 `MIC peak=XXX VAD=SPEECH/silent` 日志
- 不知道哪一层滤掉了语音
- 不知道 ASR 在处理什么
- 不知道为什么 endpoint 没触发

---

## 目标架构

```
┌─────────────────────────────────────────────────────────┐
│                    AudioPipeline (新)                      │
│  协调各模块，但不实现任何处理逻辑                            │
├─────────┬──────────┬──────────┬──────────┬───────────────┤
│ MicInput│AudioProc │   VAD    │   ASR    │  OutputMgr    │
│ (采集)  │ (滤波)   │ (检测)   │ (识别)   │ (TTS+chime)   │
│         │          │          │          │               │
│ • 设备  │ • HPF    │ • Silero │ • local  │ • TTS 引擎    │
│ • 增益  │ • 降噪   │ • 阈值   │ • cloud  │ • chime 合成  │
│ • 采样率│ • 回声消除│ • 状态机 │ • fallback│ • playback   │
└─────────┴──────────┴──────────┴──────────┴───────────────┘
         ↓             ↓           ↓           ↓
      可独立       可独立       可独立      可独立
      测试         开关         切换       测试
```

---

## 重构步骤（每步独立验证）

### Step 0: 保存当前能用的状态 [5 分钟]
```
目标: 确保有一个已知能用的基线
操作:
  - sunrise 上 commit 当前旧代码状态
  - 记录 ALSA 增益值 (111) 到 config.yaml 注释
  - 确认语音对话正常
验证: 说"你好" → 收到回复
```

### Step 1: 拆分 MicInput 模块 [30 分钟]
```
新文件: askme/voice/mic_input.py
职责: 麦克风设备管理 + 音频采集
接口:
  class MicInput:
      def __init__(self, device, sample_rate, channels)
      def read_chunk(self) -> np.ndarray  # 返回 float32 samples
      def get_peak(self, samples) -> int
      def is_open(self) -> bool

从 audio_agent.py 提取:
  - sd.InputStream 管理
  - input_device 配置
  - peak 计算
  - pre_roll buffer 管理

独立测试:
  mic = MicInput(device=0, sample_rate=16000)
  chunk = mic.read_chunk()
  print(f"peak={mic.get_peak(chunk)}")
```

### Step 2: 拆分 AudioProcessor 模块 [30 分钟]
```
新文件: askme/voice/audio_processor.py
职责: 音频预处理（滤波、降噪、回声消除）
接口:
  class AudioProcessor:
      def __init__(self, config)
      def process(self, samples: np.ndarray) -> np.ndarray
      def calibrate(self, noise_samples: np.ndarray)
      @property
      def is_calibrated(self) -> bool

整合现有模块:
  - audio_filter.py (HPF)
  - noise_reduction.py (频谱减法)
  - echo gate 逻辑
  - noise gate 逻辑

独立测试:
  proc = AudioProcessor({"hpf_enabled": True, "noise_reduction": False})
  clean = proc.process(raw_samples)
  print(f"peak before={peak(raw)}, after={peak(clean)}")
```

### Step 3: 拆分 VAD 状态机 [30 分钟]
```
新文件: askme/voice/vad_controller.py (包装现有 vad.py)
职责: 语音活动检测 + 状态管理 + barge-in 逻辑
接口:
  class VADController:
      def __init__(self, config)
      def feed(self, samples_int16) -> VADEvent
      @property
      def state(self) -> VADState  # SILENCE | SPEECH | BARGE_IN_PENDING | BARGE_IN_CONFIRMED

  class VADEvent(Enum):
      SILENCE = "silence"
      SPEECH_START = "speech_start"
      SPEECH_CONTINUE = "speech"
      SPEECH_END = "speech_end"
      BARGE_IN_START = "barge_in_start"
      BARGE_IN_CONFIRMED = "barge_in_confirmed"
      BARGE_IN_DISMISSED = "barge_in_dismissed"

从 audio_agent.py 提取:
  - speech_active 状态机
  - barge_in_pending / barge_in_hold 逻辑
  - _MAX_SPEECH_DURATION guard
  - noise_gate 过滤

独立测试:
  vad = VADController({"threshold": 0.5, "barge_in_hold": 0.15})
  event = vad.feed(samples)
  print(f"event={event}, state={vad.state}")
```

### Step 4: 拆分 ASR 管理器 [45 分钟]
```
新文件: askme/voice/asr_manager.py
职责: ASR 后端管理 + fallback + 结果过滤
接口:
  class ASRManager:
      def __init__(self, config)
      def start_recognition(self)      # 开始新一轮识别
      def feed_audio(self, samples)     # 喂音频
      def check_result(self) -> ASRResult | None  # 检查是否有结果
      def reset(self)

  @dataclass
  class ASRResult:
      text: str
      source: str  # "local" | "cloud"
      confidence: float
      latency_ms: float

整合:
  - 本地 sherpa-onnx ASR (asr.py)
  - 云端 Paraformer (cloud_asr.py)
  - fallback 逻辑
  - 噪声文本过滤 (_NOISE_UTTERANCES)
  - 标点恢复

独立测试:
  asr = ASRManager({"cloud_enabled": True, "local_fallback": True})
  asr.start_recognition()
  asr.feed_audio(speech_samples)
  result = asr.check_result()
  print(f"text='{result.text}' source={result.source}")
```

### Step 5: 重写 listen_loop [1 小时]
```
目标: listen_loop 变成纯协调器，只调用模块接口
伪代码:
  def listen_loop(self):
      mic = self._mic          # Step 1
      proc = self._processor   # Step 2
      vad = self._vad          # Step 3
      asr = self._asr          # Step 4

      asr.start_recognition()
      while not self.stop_event.is_set():
          raw = mic.read_chunk()
          clean = proc.process(raw)
          peak = mic.get_peak(clean)

          event = vad.feed(to_int16(clean))

          if event == VADEvent.SPEECH_START:
              asr.start_recognition()
              logger.info("Speech start (peak=%d)", peak)

          if event in (SPEECH_CONTINUE, SPEECH_START):
              asr.feed_audio(clean)

          if event == VADEvent.SPEECH_END:
              result = asr.check_result()
              if result and result.text:
                  return result.text

          if event == VADEvent.BARGE_IN_CONFIRMED:
              self._output.stop_tts()

验证: 完整语音对话（唤醒 → ASR → LLM → TTS → 连续对话）
```

### Step 6: 加回新功能（逐个，每个单独验证）[2 小时]
```
6a. Pipeline Trace (trace.py) → 验证 /trace 端点
6b. Address Detection → 验证闲聊过滤
6c. LLM 切 Qwen-turbo → 验证 TTFT 降低
6d. Cloud ASR (Paraformer) → 验证识别准确率
6e. Dashboard + Web Chat → 验证页面 + stop_playback
6f. 音频滤波 (HPF) → 在增益 111 下校准，验证 VAD 仍工作
```

---

## 每步验证清单

每完成一步，在 sunrise 上执行：

```bash
# 1. 部署
git push sunrise master

# 2. 重启
tmux send-keys -t askme C-c C-c
sleep 2
tmux send-keys -t askme 'python -m askme --legacy' Enter

# 3. 验证（必须全过）
□ /health 返回 status: ok
□ MIC peak 日志正常（~200-400 静音，1000+ 语音）
□ 说"你好" → VAD 检测到 → ASR 识别 → LLM 回复 → TTS 播放
□ 连续说第二句 → 不需要唤醒词 → 正常回复
```

---

## 风险和注意事项

1. **永远不改 ALSA 增益**——固定 111，写入 startup 脚本
2. **每步一个 commit**——出问题立刻 revert
3. **sunrise 上先跑旧代码确认能用，再逐步替换**
4. **新模块默认 disabled**——通过 config 开关，不影响基线
5. **独立测试脚本**——每个新模块都有 `test_xxx.py` 可以在 sunrise 上单独跑

---

## 时间估算

| Step | 内容 | 时间 |
|------|------|------|
| 0 | 保存基线 | 5 分钟 |
| 1 | MicInput | 30 分钟 |
| 2 | AudioProcessor | 30 分钟 |
| 3 | VADController | 30 分钟 |
| 4 | ASRManager | 45 分钟 |
| 5 | 重写 listen_loop | 1 小时 |
| 6 | 加回新功能 | 2 小时 |
| **总计** | | **~5 小时** |
