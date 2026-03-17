# inovxio_memory v1.0 — 世界级工业机器人记忆系统

> RALPLAN-DR Deliberate Mode | Revision 2 (post-Architect review)

---

## RALPLAN-DR Summary

### Principles (5)

1. **Extract-Store-Retrieve-Evolve**: 4 阶段管道，但 Evolve（链接+冲突）延迟到 Phase 1（需要嵌入/LLM 支持）
2. **Markdown 是长期记忆的真源**: 人类可读、git 追踪、与 MemU category.md 共用目录
3. **零依赖独立包**: 通过 EpisodicConfig dataclass 打断 askme.config 耦合
4. **离线优先、崩溃安全**: 原子写入（temp+rename），所有写操作
5. **MemU 是向量引擎，inovxio_memory 是领域层**: Phase 1 定义 MemU memory_type 映射

### Decision Drivers (3)

1. **实际记忆代码只有 ~760 行**（不含 MemU/LingTu），facade 和轻量重写工作量接近
2. **S100P 资源受限**: 不能引入重型服务
3. **3 个 facade 必须收敛为 1**: MemoryService(shared) + MemorySystem(askme) + 新 Memory → 统一为 Memory

### Viable Options（修订后）

#### Option A-revised: Facade + 各 Store 保留原生数据模型（推荐）

**Architect 综合方案**: 保留 Facade 架构的单入口优势，但放弃统一 MemoryRecord。各 Store 保留 Episode/Location/Procedure 原生模型。跨 Store 查询用轻量 SearchResult。

```python
# 各 Store 保留原生模型
mem.episodic.log("anomaly", "温度过高")  → 内部用 Episode
mem.spatial.record_visit("仓库A", ...)   → 内部用 Location
mem.procedural.record_outcome(...)       → 内部用 Procedure
mem.markdown.write("equipment", "...")   → 内部用 plain text

# 跨 Store 查询返回轻量 SearchResult
results = mem.search("温度")  → list[SearchResult(store, id, content, score)]

# 统一上下文聚合
context = mem.get_context(query="仓库A")  → 合并所有 Store 的上下文
```

**优点**: 保留 Episode Ebbinghaus 衰减、Location 空间查询、Procedure Beta 分布 — 零类型安全损失
**缺点**: API 不完全统一（store-specific 方法 + 跨 store search）

#### Option B: 统一 MemoryRecord（否决）
**否决理由（Architect 论证）**: Episode 的 stability/retrievability、Location 的 2D coords、Procedure 的 alpha/beta 都会被塞进 metadata dict，丢失类型安全。实际是 God-Object 反模式。

### Pre-mortem（5 场景，含 Architect 补充）

1. **EpisodicStore 的 askme.config 依赖**: 13 个配置参数 + project_root — 解法：EpisodicConfig dataclass，askme 侧做适配
2. **MarkdownStore 和 MemU category.md 冲突**: 解法：共用同一目录，MarkdownStore 默认 data_dir 指向 MemU 的 category 目录
3. **SQLite FTS5 中文分词**: 解法：Phase 0 用 JSON，Phase 1 加 jieba
4. **askme 30+ 测试 monkeypatch 断裂**: 解法：askme/brain/ 保留 re-export 兼容层，测试逐步迁移
5. **双 Episode 类分叉**: 解法：inovxio_memory 的 Episode 是唯一真源，askme/brain/episode.py 变为 re-export

### Expanded Test Plan

| Level | 内容 | 验收标准 |
|-------|------|---------|
| **Unit** | 每个 Store 独立测试 | >= 10 tests/store, Episode 保留 Ebbinghaus 测试 |
| **Integration** | Memory facade 协调多 Store | log 50 episodes → reflect → verify digest.md → query by location → verify decay |
| **E2E** | Sunrise 真机巡检 | 3 位置巡检 + 2 异常 + Markdown 生成 + voice latency P95 增加 < 50ms |
| **Observability** | /health 暴露记忆指标 | item_count, buffer_occupancy%, disk_usage_mb, reflection_count, admission_rate |

---

## Architecture（修订后）

```
inovxio_memory/
├── __init__.py              — Memory, SearchResult, create_memory
├── _search.py               — SearchResult 轻量跨 Store 查询结果
│
├── stores/
│   ├── episodic.py          — EpisodicStore (保留 Episode 原生模型)
│   │                          EpisodicConfig dataclass 打断 askme.config 依赖
│   │                          ReflectionCallback protocol 解耦 LLM
│   ├── spatial.py           — SpatialStore (保留 Location/SpatialEvent 原生模型)
│   ├── procedural.py        — ProceduralStore (保留 Procedure 原生模型)
│   └── markdown.py          — MarkdownStore (新，与 MemU category.md 共用目录)
│
├── admission.py             — MemoryAdmissionControl (现有)
├── episode.py               — Episode + recency_boost (唯一真源)
├── map_adapter.py           — LingTu topo_memory 适配器 (合并两版)
├── memory.py                — Memory facade (替代 MemoryService + MemorySystem)
└── py.typed
```

**Phase 0 不包含（延迟到 Phase 1）**:
- ~~linking.py~~ — 需要嵌入才有价值
- ~~conflict.py~~ — 需要 LLM 做实体识别
- ~~_backend.py / backends/~~ — Phase 0 各 Store 自己管持久化
- ~~stores/semantic.py~~ — 需要 MemU 向量接入

### Facade 收敛计划

| 现有 | 归宿 | 时机 |
|------|------|------|
| `MemoryService` (shared/service.py) | 被 `Memory` 替代 | Phase 0 完成后废弃 |
| `MemorySystem` (askme/memory_system.py) | askme 侧的 `Memory` adapter | Phase 0 完成后废弃 |
| `Memory` (新) | **唯一入口** | Phase 0 交付 |

### EpisodicConfig 设计

```python
@dataclass
class EpisodicConfig:
    """从 askme.config 中解耦的配置。"""
    data_dir: str = "data/memory"
    flush_threshold: int = 100
    episode_retention_hours: float = 24.0
    reflect_min_events: int = 5
    reflect_cooldown_seconds: float = 300.0
    importance_threshold: float = 5.0
    knowledge_context_chars: int = 1000
    digest_context_chars: int = 600
    relevant_context_chars: int = 600
    relevant_top_k: int = 5
    max_buffer_size: int = 200

# askme 侧适配:
from askme.config import get_config
cfg = get_config()
episodic_cfg = EpisodicConfig(
    data_dir=str(project_root() / cfg["app"]["data_dir"]),
    **cfg.get("memory", {}).get("episodic", {}),
)
```

### SearchResult 设计

```python
@dataclass
class SearchResult:
    """跨 Store 查询的轻量结果。"""
    store: str          # "episodic" | "spatial" | "procedural" | "markdown"
    content: str        # 人类可读描述
    score: float        # 相关性评分 0-1
    timestamp: float    # 创建时间
    metadata: dict      # Store 特定数据（Location coords、Procedure success_rate 等）
```

### Memory Facade API

```python
class Memory:
    """统一记忆入口 — 替代 MemoryService + MemorySystem。"""

    def __init__(self, data_dir="data/memory", site_id="default", robot_id="default"):
        self.episodic = EpisodicStore(EpisodicConfig(data_dir=...))
        self.spatial = SpatialStore(data_dir=...)
        self.procedural = ProceduralStore(data_dir=...)
        self.markdown = MarkdownStore(data_dir=...)
        self.admission = MemoryAdmissionControl()

    # ── 跨 Store 搜索 ──
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """搜索所有 Store，按 score 排序返回。"""

    # ── 统一上下文 ──
    def get_context(self, query: str = "", max_chars: int = 2000) -> str:
        """聚合所有 Store 的上下文，注入 LLM 系统提示。"""

    # ── 班次交接 ──
    def get_shift_summary(self, hours: int = 8) -> str:
        """过去 N 小时的巡检总结。"""

    # ── 地图同步 ──
    def sync_map(self, topo_path: str, kg_path: str = None) -> int: ...

    # ── 准入 ──
    def should_remember(self, kind: str, text: str, importance: float = 0.5) -> bool: ...

    # ── 持久化 ──
    def save(self): ...
    def load(self): ...
```

## Implementation Steps（Phase 0, 修订后）

| Step | 文件 | 内容 | 验收 |
|------|------|------|------|
| 0.1 | `episode.py` | 确认为唯一真源，askme re-export | import 测试 |
| 0.2 | `admission.py` | 迁移现有，无变化 | 17 tests |
| 0.3 | `_search.py` | SearchResult dataclass | 序列化测试 |
| 0.4 | `stores/spatial.py` | 包装 SiteKnowledge，保留 Location 模型 | 10 tests |
| 0.5 | `stores/procedural.py` | 包装 ProceduralMemory，保留 Procedure 模型 | 10 tests |
| 0.6 | `stores/episodic.py` | 从 askme 抽取，EpisodicConfig + ReflectionCallback | 15 tests |
| 0.7 | `stores/markdown.py` | **新**：读写 knowledge/*.md，与 MemU 共用目录 | 10 tests |
| 0.8 | `map_adapter.py` | 合并 askme + shared 两版 | 13 tests |
| 0.9 | `memory.py` | Memory facade + search + get_context | 15 tests |
| 0.10 | askme 迁移 | re-export 兼容层 + 废弃旧 facade | 1208 askme tests 通过 |

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| EpisodicStore 的 13 个配置参数 | EpisodicConfig dataclass，askme 做适配 |
| 30+ 测试 monkeypatch 断裂 | re-export 兼容层，测试逐步迁移 |
| MarkdownStore 和 MemU 冲突 | 共用同一目录 |
| Phase 0 无嵌入 = 搜索只是关键词 | 接受限制，Phase 1 接入 MemU 向量 |
| 3 个 facade 共存过渡期 | 明确废弃时间表 |

## ADR

**Decision**: Option A-revised — Facade + 各 Store 保留原生数据模型 + 轻量 SearchResult 跨 Store 查询

**Drivers**: 760 行实际记忆代码，facade 和重写工作量接近；保留 Ebbinghaus/Beta posterior 等领域智能

**Alternatives**:
- 统一 MemoryRecord: 否决 — God-Object，丢失类型安全
- Mem0 SDK: 否决 — 无离线、无空间、无 Markdown
- 完全重写: 否决 — 不需要，760 行代码 facade 即可

**Why chosen**: 保留每个 Store 的领域丰富性（Episode 衰减、Location 空间、Procedure Beta），同时提供统一入口和跨 Store 搜索

**Consequences**: API 不完全统一（store-specific + 跨 store search），Phase 1 才有嵌入检索

**Follow-ups**:
- Phase 1: SQLite + 嵌入检索 + linking + conflict resolution + MemU type mapping
- Phase 2: 多机共享 + 工业合规 + gRPC 服务

## Changelog (Architect Review)
- **Dropped**: MemoryRecord 统一数据模型 → 各 Store 保留原生模型
- **Dropped**: linking.py + conflict.py 从 Phase 0 → 延迟到 Phase 1
- **Added**: EpisodicConfig dataclass 解决 askme.config 耦合
- **Added**: 3 facade 收敛计划（MemoryService + MemorySystem → Memory）
- **Added**: Pre-mortem #4（测试断裂）+ #5（双 Episode 分叉）
- **Added**: 观测指标增加 buffer_occupancy%, disk_usage_mb
- **Clarified**: "30k 行代码" → 实际记忆代码 760 行
