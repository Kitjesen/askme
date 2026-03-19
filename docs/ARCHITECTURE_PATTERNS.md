# 内部成熟产品的架构模式提取报告

> 分析日期：2026-03-13
> 审视范围：LingTu (v1.7.5)、OTA 系统 (v3.1.0+)、Brainstem (stable)
> 目标：为 askme 语音系统提炼可复用的设计模式

---

## 核心发现

askme 的当前实现（skill_dispatcher + brain_pipeline）已采用了内部产品的若干先进模式，但在以下 5 个维度仍有优化空间。

---

## 1. **任务中断恢复** — LingTu + OTA 的双层保护

### 模式来源

**LingTu (nav-gateway)**
- **contract**: `NavigationSession` 的 provider 追踪 (`provider`, `provider_task_id`, `provider_status`)
- **文件**: `D:\inovxio\products\nova-dog\runtime\contracts\CONTRACT_INVENTORY.md:125`
- **设计**: 每次派发导航都记录下游供应商身份，网络中断后可重连原任务

**OTA 系统 (v3.1.0+)**
- **断点续传**: Range 请求，任务状态持久化到 SQLite
- **文件**: `D:\inovxio\infra\ota\CHANGELOG.md:70`
- **验证**: 真机 E2E 测试（sunrise RDK S100P）全部通过，包括回滚场景

### askme 现状

```python
# askme/pipeline/skill_dispatcher.py:43-54
@dataclass
class MissionContext:
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = "voice"  # voice | text | runtime
    steps: list[MissionStep] = field(default_factory=list)
    shared_context: dict[str, str] = field(default_factory=dict)
    started_at: float = field(default_factory=time.time)
```

**缺陷**：
- `MissionContext` 仅在内存中，运行时宕机丢失所有步骤
- 无 provider 追踪 — 如果某个 skill 调用远程 runtime 服务（如 navigation），中断后无法判断是否需要重连或回滚
- 无显式的"任务持久化边界"定义（哪些步骤必须原子化）

### 具体建议

**建议 1: 任务持久化层**
- 在 `askme/data/missions/` 下维护 mission 的 markdown 日志
- 格式参考 `askme/data/sessions/` 的模式（已有 SessionMemory）
- 包含：mission_id, started_at, steps[], current_step_index, provider_context, last_checkpoint
- 实现：MissionPersistence 类在 dispatcher.py 中

**建议 2: Provider 追踪**
- 为 MissionContext 添加字段：
  ```python
  @dataclass
  class MissionContext:
      # 现有字段...
      provider_tasks: dict[str, ProviderTask] = field(default_factory=dict)
      # 格式: {skill_name: {provider: "runtime", task_id: "nav-123", status: "running"}}
  ```
- 当 skill 调用 runtime 时（navigate, follow_person 等），记录：
  - `provider` = "runtime"
  - `provider_task_id` = 来自 nav-gateway 的 session_id
  - `provider_status` = 当前下游状态
- 中断恢复时，先查询 provider 状态而非假设任务已完成

**建议 3: 错误恢复决策树**
参考 OTA 的 `errors.py` 模式（见下文）：
```python
# 在 skill_dispatcher.py 中添加
class MissionError(Exception):
    """Mission-level 错误，携带恢复建议。"""
    retryable: bool
    recovery_action: str  # "retry", "skip", "rollback"
    affected_step: int
```

---

## 2. **显式状态机** — Brainstem FSM 的模式

### 模式来源

**Brainstem (han_dog_brain/cms/)**
- FSM 实现：`Cms<S, A>` 其中 S=State, A=Action
- **State 枚举**: grounded, running, paused, failed, shutdown
- **Action 枚举**: begin, pause, resume, estop, recover
- **转换规则**: 每个状态仅允许特定的 action 子集（在状态机中硬编码）
- **文件**: `D:\inovxio\brain\brainstem\CLAUDE.md:92-93`
- **关键**: Dart 的严格类型系统 + Freezed 代码生成，确保状态转换代码不能"漏掉"任何分支

### askme 现状

```python
# askme/pipeline/skill_dispatcher.py（伪代码）
async def dispatch_skill(self, skill_name: str, ...):
    if self._mission is None:
        self._mission = MissionContext()
    # 执行 skill
    self._mission.add_step(...)
    # 但没有显式的"任务状态"枚举
```

**缺陷**：
- MissionContext 无显式状态字段，推理状态完全隐含在"steps 是否存在"
- 无法区分：待确认 vs 运行中 vs 暂停 vs 已完成 vs 失败（需要这些区分来支持"打断"）
- 多个输入源（voice/text/runtime）可能同时修改 mission，无状态锁

### 具体建议

**建议 4: 显式任务状态机**

定义文件：`askme/pipeline/mission_state.py`（新文件）

```python
from enum import Enum

class MissionState(Enum):
    """Mission 生命周期状态。"""
    IDLE = "idle"              # 未启动
    PENDING_CONFIRMATION = "pending_confirmation"  # 待用户确认
    RUNNING = "running"        # 任务运行中
    PAUSED = "paused"          # 暂停
    WAITING_INPUT = "waiting_input"  # 等待用户输入
    SUCCEEDED = "succeeded"    # 成功
    FAILED = "failed"          # 失败
    CANCELED = "canceled"      # 用户取消

class MissionAction(Enum):
    """允许的任务转换。"""
    START = "start"
    PAUSE = "pause"
    RESUME = "resume"
    CANCEL = "cancel"
    SUCCEED = "succeed"
    FAIL = "fail"
    REQUEST_INPUT = "request_input"
    PROVIDE_INPUT = "provide_input"

# 状态转换矩阵（参考 Brainstem 的做法）
_ALLOWED_TRANSITIONS = {
    MissionState.IDLE: {MissionAction.START},
    MissionState.PENDING_CONFIRMATION: {MissionAction.START, MissionAction.CANCEL},
    MissionState.RUNNING: {MissionAction.PAUSE, MissionAction.CANCEL, MissionAction.SUCCEED, MissionAction.FAIL},
    MissionState.PAUSED: {MissionAction.RESUME, MissionAction.CANCEL},
    MissionState.WAITING_INPUT: {MissionAction.PROVIDE_INPUT, MissionAction.CANCEL},
    MissionState.SUCCEEDED: set(),  # 终止态
    MissionState.FAILED: set(),     # 终止态
    MissionState.CANCELED: set(),   # 终止态
}
```

在 `MissionContext` 中：
```python
@dataclass
class MissionContext:
    mission_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source: str = "voice"
    state: MissionState = MissionState.IDLE  # 新增
    steps: list[MissionStep] = field(default_factory=list)
    # ...

    def transition(self, action: MissionAction) -> None:
        """状态转换，违反规则则抛异常。"""
        if action not in _ALLOWED_TRANSITIONS[self.state]:
            raise InvalidMissionTransition(
                f"Cannot {action.value} from {self.state.value}"
            )
        # 转换逻辑...
```

**优势**：
- 打断（interrupt）现在变成："如果 RUNNING，转为 PAUSED，然后 dispatch estop skill"
- "确认"流程清晰：IDLE → PENDING_CONFIRMATION → RUNNING
- 类型安全：不能从 SUCCEEDED 转回 RUNNING

---

## 3. **优先级与抢占** — dog-safety-service 的模式

### 模式来源

**Runtime Safety Contract**
- **端点**: `POST /api/v1/safety/modes/estop`（幂等 key 支持）
- **效果**: 设置 `estop_active=true`，所有派发决策查询 estop 状态
- **文件**: `D:\inovxio\products\nova-dog\runtime\contracts\CONTRACT_INVENTORY.md:59-76`
- **关键**: estop 不是"取消运行任务"，而是"阻止新任务派发"，已运行的任务自行决定是否响应

### askme 现状

```python
# askme/voice/audio_agent.py（伪代码）
async def run(self):
    while True:
        # ASR → LLM → TTS
        # 但没有优先级队列
        # 如果用户说"停止"，它会进入 LLM，可能延迟 5-10s 才响应
```

**缺陷**：
- 没有 out-of-band 打断机制（如按物理按钮时，能立即中断 ASR）
- 没有优先级：estop 和普通 skill 同等处理
- LLM 调用是同步的，无法被提前取消

### 具体建议

**建议 5: 优先级队列 + Interrupt Manager**

实现文件：`askme/pipeline/interrupt_manager.py`（新文件）

```python
from enum import IntEnum
import asyncio

class InterruptPriority(IntEnum):
    """中断优先级。"""
    ESTOP = 100        # 最高：物理按钮或语音"停止"
    PAUSE = 50         # 中等：暂停当前任务
    GENERAL = 10       # 低：普通对话

@dataclass
class Interrupt:
    priority: InterruptPriority
    action: str  # "estop" | "pause" | "cancel_current_skill"
    reason: str
    timestamp: float = field(default_factory=time.time)

class InterruptManager:
    """管理中断队列，确保高优先级请求优先处理。"""

    def __init__(self):
        self._queue: asyncio.PriorityQueue[tuple[int, Interrupt]] = asyncio.PriorityQueue()

    async def request_interrupt(self, interrupt: Interrupt) -> None:
        """请求中断。优先级高的会被先处理。"""
        # 负数使得高优先级先弹出
        priority_key = -int(interrupt.priority)
        await self._queue.put((priority_key, interrupt))

    async def pop_if_available(self) -> Interrupt | None:
        """非阻塞检查是否有待处理的中断。"""
        try:
            _, interrupt = self._queue.get_nowait()
            return interrupt
        except asyncio.QueueEmpty:
            return None

    async def wait_for_interrupt(self, timeout: float | None = None) -> Interrupt | None:
        """阻塞等待中断。"""
        try:
            _, interrupt = await asyncio.wait_for(
                self._queue.get(),
                timeout=timeout
            )
            return interrupt
        except asyncio.TimeoutError:
            return None
```

在 `BrainPipeline` 中集成：
```python
# 每个 LLM 流式调用前，检查是否有高优先级中断
async def chat_stream_with_interrupt(self, messages, ...):
    stream = self.llm.chat_stream(messages)
    async for chunk in stream:
        # 检查是否有 estop
        interrupt = await self._interrupt_mgr.pop_if_available()
        if interrupt and interrupt.priority == InterruptPriority.ESTOP:
            # 立即停止流，释放 LLM token
            break
        yield chunk
```

---

## 4. **错误分类与恢复** — OTA errors.py 模式

### 模式来源

**OTA 系统错误设计**
- 每个业务异常都定义：code, status_code, message
- 所有异常继承 OTAError 基类
- FastAPI 统一异常处理器将异常转换为标准 JSON 响应
- **关键**: `retryable` 字段让 caller 决定是否重试
- **文件**: `D:\inovxio\infra\ota\server\ota_server\errors.py:1-75`

### askme 现状

```python
# askme/brain/llm_client.py:34-60
_RETRYABLE_STATUS = {500, 502, 503, 504, 529}

class LLMClient:
    async def _stream_with_retry(self, ...):
        for attempt in range(self._max_retries + 1):
            try:
                return await self._client.chat.completions.create(...)
            except APIStatusError as e:
                if e.status_code not in _RETRYABLE_STATUS:
                    raise
                # 重试相同模型
            except APITimeoutError:
                # 重试
```

**缺陷**：
- 只有 LLM 层做了重试，skill 层没有
- 错误信息不结构化：caller 无法区分"网络超时"vs"模型宕机"vs"用户错误"
- skill 执行失败时无回滚策略

### 具体建议

**建议 6: 结构化错误体系**

实现文件：`askme/errors.py`（扩展当前的 errors.py）

参考 OTA 的设计：

```python
from enum import Enum
from typing import Optional, Dict, Any

class ErrorCode(Enum):
    """askme 统一错误码。"""
    # LLM 相关
    LLM_TIMEOUT = "llm_timeout"
    LLM_RATE_LIMIT = "llm_rate_limit"
    LLM_MODEL_UNAVAILABLE = "llm_model_unavailable"
    LLM_CONTEXT_OVERFLOW = "llm_context_overflow"

    # Skill 相关
    SKILL_NOT_FOUND = "skill_not_found"
    SKILL_EXECUTION_FAILED = "skill_execution_failed"
    SKILL_TIMEOUT = "skill_timeout"
    SKILL_INVALID_PARAMS = "skill_invalid_params"

    # Runtime 相关
    RUNTIME_UNREACHABLE = "runtime_unreachable"
    RUNTIME_ESTOP_ACTIVE = "runtime_estop_active"

    # 任务相关
    MISSION_INVALID_TRANSITION = "mission_invalid_transition"
    MISSION_INTERRUPTED = "mission_interrupted"

class AskmeError(Exception):
    """askme 统一业务异常基类。"""

    code: ErrorCode = ErrorCode.LLM_TIMEOUT
    retryable: bool = False
    recovery_action: str = "fail"  # "retry" | "skip" | "rollback" | "fail"

    def __init__(
        self,
        message: str,
        detail: Optional[Dict[str, Any]] = None,
        retryable: bool | None = None,
        recovery_action: str | None = None,
    ):
        self.message = message
        self.detail = detail or {}
        if retryable is not None:
            self.retryable = retryable
        if recovery_action is not None:
            self.recovery_action = recovery_action
        super().__init__(self.message)

class LLMTimeoutError(AskmeError):
    code = ErrorCode.LLM_TIMEOUT
    retryable = True
    recovery_action = "retry"

class SkillExecutionFailed(AskmeError):
    code = ErrorCode.SKILL_EXECUTION_FAILED
    retryable = False
    recovery_action = "rollback"  # 默认回滚，除非 skill 定义了不同的恢复策略

class RuntimeEstopActive(AskmeError):
    code = ErrorCode.RUNTIME_ESTOP_ACTIVE
    retryable = False
    recovery_action = "fail"
```

在 skill_dispatcher 中使用：

```python
async def dispatch_skill(self, skill_name: str, ...):
    try:
        result = await executor.execute_skill(skill_name, ...)
    except AskmeError as e:
        if e.retryable and self._mission.step_count < 3:
            # 自动重试最多 3 次
            await self.dispatch_skill(skill_name, ...)
        elif e.recovery_action == "rollback":
            # 回滚：取消已完成的步骤
            await self._rollback_mission()
        elif e.recovery_action == "skip":
            # 跳过：继续下一步
            self._mission.add_step(skill_name, user_text, f"[跳过] {e.message}")
        else:
            # fail：中止任务
            raise
```

---

## 5. **契约优先设计** — API_STANDARDS 模式

### 模式来源

**Runtime API Standards**
- 所有服务必须暴露 OpenAPI 契约（`contracts/openapi/`）
- gRPC 服务定义 `.proto` 文件（`shared/proto/runtime/`）
- 契约先于实现：先改 OpenAPI，再改代码
- **幂等性**: 所有创建/触发操作支持 `Idempotency-Key`
- **标准头**: `X-Request-Id`, `X-Correlation-Id`, `X-Operator-Id`, `X-Service-Name`
- **文件**: `D:\inovxio\products\nova-dog\runtime\contracts\API_STANDARDS.md:1-62`

### askme 现状

```python
# askme/skills/skill_model.py（伪代码）
@dataclass
class Skill:
    name: str
    description: str
    parameters: dict  # 自由形式，无 OpenAPI 定义
```

**缺陷**：
- skill 接口无显式合约定义
- 无 OpenAPI spec，caller 必须读代码才知道参数
- 无版本控制（如果修改 skill 参数，旧 caller 会破裂）
- 无幂等键支持

### 具体建议

**建议 7: Skill 契约定义**

创建文件：`askme/contracts/skills-v1.yaml`（新 OpenAPI 文件）

```yaml
openapi: 3.0.0
info:
  title: Askme Skills API
  version: 1.0.0

paths:
  /api/v1/skills:
    get:
      summary: List available skills
      responses:
        200:
          content:
            application/json:
              schema:
                type: object
                properties:
                  items:
                    type: array
                    items:
                      $ref: '#/components/schemas/SkillMetadata'

  /api/v1/skills/{skill_name}/execute:
    post:
      summary: Execute a skill
      parameters:
        - name: skill_name
          in: path
          required: true
          schema:
            type: string
      requestBody:
        required: true
        content:
          application/json:
            schema:
              $ref: '#/components/schemas/SkillExecutionRequest'
      responses:
        200:
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/SkillExecutionResponse'
        400:
          $ref: '#/components/responses/BadRequest'
        429:
          $ref: '#/components/responses/TooManyRequests'

components:
  schemas:
    SkillMetadata:
      type: object
      required:
        - name
        - description
        - version
        - parameters_schema
      properties:
        name:
          type: string
          example: "navigate"
        description:
          type: string
        version:
          type: string
          example: "1.0.0"
        parameters_schema:
          type: object
          additionalProperties: true
          description: "JSON Schema for parameters"

    SkillExecutionRequest:
      type: object
      required:
        - mission_id
        - parameters
      properties:
        mission_id:
          type: string
          format: uuid
        parameters:
          type: object
          additionalProperties: true
        idempotency_key:
          type: string
          description: "Optional idempotency key for safe retry"

    SkillExecutionResponse:
      type: object
      required:
        - mission_id
        - step_id
        - status
        - result
      properties:
        mission_id:
          type: string
        step_id:
          type: string
        status:
          type: string
          enum: [succeeded, failed, timeout]
        result:
          type: object
          additionalProperties: true
        executed_at:
          type: string
          format: date-time
```

在代码中实现版本化接口：

```python
# askme/skills/skill_executor.py
class SkillExecutor:
    async def execute_skill_v1(
        self,
        skill_name: str,
        mission_id: str,
        parameters: dict,
        idempotency_key: str | None = None,
    ) -> SkillExecutionResponse:
        """Execute skill with version 1.0.0 contract."""
        # 幂等性检查
        if idempotency_key:
            cached = await self._idempotency_cache.get(idempotency_key)
            if cached:
                return cached

        result = await self._execute(skill_name, parameters)

        response = SkillExecutionResponse(
            mission_id=mission_id,
            step_id=uuid.uuid4().hex[:12],
            status="succeeded",
            result=result,
        )

        if idempotency_key:
            await self._idempotency_cache.set(idempotency_key, response, ttl=3600)

        return response
```

---

## 总结：优先级与实施路线图

### 立即可行（P0 — 本周）

1. **建议 4**：定义 MissionState 枚举 + 状态转换
   - 文件：`askme/pipeline/mission_state.py`
   - 改动：`MissionContext` 添加 state 字段，dispatch_skill 检查状态合法性
   - 工作量：2-3h
   - 收益：解锁"打断"功能，支持"确认"流程

2. **建议 6**：结构化错误体系
   - 文件：扩展 `askme/errors.py`
   - 改动：定义 ErrorCode + AskmeError 体系，skill_dispatcher 添加 recovery_action 处理
   - 工作量：3-4h
   - 收益：更清晰的错误诊断，支持自动重试与回滚

### 短期可行（P1 — 2周内）

3. **建议 5**：InterruptManager
   - 文件：`askme/pipeline/interrupt_manager.py`
   - 集成：BrainPipeline + VoiceLoop
   - 工作量：4-5h
   - 收益：物理按钮或"停止"语音可立即中断 LLM 流

4. **建议 1**：MissionPersistence
   - 文件：`askme/pipeline/mission_persistence.py`
   - 扩展 SessionMemory 的思路
   - 工作量：5-6h
   - 收益：宕机恢复，任务可审计

5. **建议 2**：Provider 追踪
   - 改动：MissionContext + skill_executor
   - 工作量：3-4h
   - 收益：支持中断后重连原任务

### 长期优化（P2 — 1个月）

6. **建议 7**：契约定义
   - 文件：`askme/contracts/skills-v1.yaml`
   - 改动：skill_executor 实现版本化接口
   - 工作量：6-8h
   - 收益：SDK 生成、自动化测试、客户集成友好

---

## 代码片段速查

| 内容 | 源文件 | 行号 |
|------|--------|------|
| Mission 生命周期 | askme/pipeline/skill_dispatcher.py | 43-84 |
| LLM 重试策略 | askme/brain/llm_client.py | 34-60 |
| 错误分类基类 | infra/ota/server/ota_server/errors.py | 46-73 |
| FSM 状态转换 | brain/brainstem/CLAUDE.md | 81-94 |
| 契约清单 | products/nova-dog/runtime/contracts/CONTRACT_INVENTORY.md | 1-50 |

---

## 参考文献

1. **LingTu 导航系统** — 成熟的任务生命周期管理（provider 追踪、会话管理）
2. **OTA 升级系统** — 工业级错误处理、断点续传、真机验证
3. **Brainstem 控制栈** — 严格的 FSM 设计、状态转换矩阵、类型安全
4. **NOVA Dog Runtime** — 企业合约设计（OpenAPI + Protobuf）、幂等性、审计日志

---

## 附录：对标检查清单

- [ ] MissionContext 有显式 state 字段
- [ ] skill_dispatcher 在状态转换前校验合法性
- [ ] 所有业务异常继承 AskmeError 并定义 ErrorCode
- [ ] 高优先级中断（estop）能立即中断 LLM 流
- [ ] mission 支持持久化到磁盘（宕机恢复）
- [ ] skill 调用记录 provider 信息（runtime 服务追踪）
- [ ] skill 接口有 OpenAPI 定义
- [ ] 创建/触发操作支持 Idempotency-Key

