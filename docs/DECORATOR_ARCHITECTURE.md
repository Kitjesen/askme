# Askme 装饰器架构设计稿

**主题：从"手动注册+硬编码路由"升级为"声明式自注册模块系统"**
**日期：2026-03-21**
**状态：设计中**

---

## 一、当前问题

### 加一个工具需要改 4 个地方

```
1. 写 class XxxTool(BaseTool) — 50-80 行模板代码
2. 写 register_xxx_tools() 函数
3. 改 app.py — import + 调用 register
4. 如果是 Agent 工具 — 改 thunder_agent_shell.py _AGENT_ALLOWED_TOOLS
5. 如果是 Agent 工具 — 改 _TOOL_VOICE_LABELS
```

### 加一个技能需要改 3 个地方

```
1. 写 SKILL.md
2. 如果走 Agent Shell — 改 brain_pipeline.py _AGENT_SHELL_SKILLS
3. 如果有特殊路由 — 改 intent_router.py
```

### 数字

- 31 个 BaseTool 子类
- 41 个 SKILL.md 技能
- app.py 54 处手动 wiring
- 7 个 register_xxx 函数分散各处

---

## 二、设计目标

1. **加工具 = 写一个函数 + 一个装饰器** — 不改 app.py
2. **加技能 = 写一个函数 + 一个装饰器** — 不改 brain_pipeline.py
3. **模块自描述** — 名字、参数、安全等级、依赖全在装饰器里
4. **启动时自动发现 + 自动注册** — Convention over Configuration
5. **向后兼容** — 现有 BaseTool 和 SKILL.md 继续工作

---

## 三、两个核心装饰器

### 1. `@tool` — 函数变工具

```python
from askme.core.decorators import tool

@tool(
    name="look_around",
    description="观察周围环境",
    safety="normal",
    agent_allowed=True,        # 自动加入 _AGENT_ALLOWED_TOOLS
    voice_label="观察环境",     # 自动加入 _TOOL_VOICE_LABELS
)
def look_around(question: str = "") -> str:
    """可选 question 参数让视觉模型重点关注特定物体。"""
    ...
```

**自动完成的事：**
- 从函数签名提取 parameters schema（类型注解 → JSON Schema）
- 创建 BaseTool 子类实例并注册到 ToolRegistry
- 加入 Agent Shell 白名单 + 语音标签
- **不需要改 app.py**

**需要依赖注入的工具：**
```python
@tool(
    name="look_around",
    requires=["vision_bridge"],  # 声明依赖
)
def look_around(question: str = "", *, _vision=None) -> str:
    # _vision 由框架在注册时自动注入
    ...
```

### 2. `@skill` — 函数变技能

```python
from askme.core.decorators import skill

@skill(
    name="find_object",
    triggers=["帮我找", "找一下", "哪里有"],
    tools=["look_around", "scan_around", "move_robot"],
    agent_shell=True,      # 自动加入 _AGENT_SHELL_SKILLS
    timeout=90,
)
async def find_object(user_input: str, context: dict) -> str:
    """搜索指定物体——观察环境、导航到可能位置、视觉确认。"""
    # prompt 模板可以写在 docstring 或单独的 .md 里
    ...
```

**自动完成的事：**
- 注册到 SkillManager
- voice_trigger 注册到 IntentRouter
- 加入 _AGENT_SHELL_SKILLS（如果 agent_shell=True）
- **不需要改 brain_pipeline.py**

---

## 四、自动发现机制

### 方案 A：模块扫描（推荐）

```python
# askme/core/registry.py

def auto_discover():
    """扫描 askme/tools/ 和 askme/skills/ 下所有模块，
    找到 @tool 和 @skill 标记的函数，自动注册。"""

    for module_path in discover_modules("askme.tools"):
        import_module(module_path)  # 导入触发装饰器执行

    for module_path in discover_modules("askme.skills"):
        import_module(module_path)
```

**启动时：**
```python
# app.py — 从 54 行 wiring 变成 1 行
from askme.core.registry import auto_discover
auto_discover(self.tools, self.skill_manager)
```

### 向后兼容

- 现有 `BaseTool` 子类继续工作（auto_discover 同时扫描 class 和 decorator）
- 现有 `SKILL.md` 继续工作（SkillManager.load() 不变）
- `@tool` 和 `@skill` 是可选的——新代码用装饰器，旧代码不用改

---

## 五、依赖注入

工具经常需要 VisionBridge、AudioAgent 等运行时对象。两种方案：

### 方案 A：requires 声明式（推荐）

```python
@tool(name="look_around", requires=["vision_bridge"])
def look_around(question="", *, _vision=None):
    ...

# auto_discover 时：
# 1. 读 requires 列表
# 2. 从 app 的已注册组件中查找 "vision_bridge"
# 3. 绑定到 _vision 参数
```

### 方案 B：全局 ServiceLocator

```python
from askme.core.services import get_service

@tool(name="look_around")
def look_around(question=""):
    vision = get_service("vision_bridge")
    ...
```

方案 A 更显式、可测试。方案 B 更简单但隐式依赖。

---

## 六、@rpc 装饰器（Phase 2）

学 DimOS 的 `@rpc`，让模块方法可以跨模块调用：

```python
class ProactiveAgent:
    @rpc
    def get_risk_level(self) -> float:
        """当前风险等级，供其他模块查询。"""
        return self._current_risk

# 其他模块：
risk = await rpc_call("proactive_agent.get_risk_level")
```

Phase 2 再做——当前模块间通信还不够复杂。

---

## 七、实施计划

### Phase 1：装饰器定义 + 自动发现（不改现有代码）

```
新建:
  askme/core/__init__.py
  askme/core/decorators.py      — @tool, @skill 装饰器实现
  askme/core/registry.py        — auto_discover 扫描 + 注册
  askme/core/services.py        — 组件注册表（依赖注入）
  tests/test_decorators.py      — 装饰器单元测试

改动:
  无 — Phase 1 只定义装饰器，不改现有代码
```

### Phase 2：新工具/技能用装饰器

```
新工具用 @tool 写（不再写 BaseTool 子类）
新技能用 @skill 写（不再写 SKILL.md）
app.py 加 auto_discover() 调用
```

### Phase 3：迁移现有工具

```
逐步把 31 个 BaseTool 子类迁移为 @tool 函数
app.py 的 54 行 wiring 逐步删除
最终 app.py 只需要 ~10 行初始化代码
```

---

## 八、效果对比

### 加一个新工具

| 步骤 | 现在 | 装饰器后 |
|------|------|----------|
| 写实现 | 写 class (50行) | 写函数 (20行) |
| 注册 | 改 app.py | 不改 |
| Agent 白名单 | 改 thunder_agent_shell.py | 不改 |
| 语音标签 | 改 thunder_agent_shell.py | 不改 |
| 总文件改动 | 3-4 个文件 | **1 个文件** |

### 加一个新技能

| 步骤 | 现在 | 装饰器后 |
|------|------|----------|
| 写定义 | 写 SKILL.md | 写函数 + @skill |
| Agent Shell 路由 | 改 brain_pipeline.py | 不改 |
| 触发词 | SKILL.md 里 | @skill 参数里 |
| 总文件改动 | 2-3 个文件 | **1 个文件** |

---

## 九、不做的事

- 不引入 RxPY / Observable
- 不做 LCM Transport（文件通信够用）
- 不做 Blueprint 自动连线（模块还不够多）
- 不改现有 BaseTool / SKILL.md 的运行机制（向后兼容）
- 不做 @rpc（Phase 2）

---

## 十、核心原则

1. **装饰器只做标记，不做包装** — 学 DimOS 的 `@rpc`（3 行代码）
2. **发现在启动时，不在运行时** — 导入模块触发注册，一次性完成
3. **声明式 > 命令式** — 工具自己说"我是什么"，不是 app.py 说"你是什么"
4. **向后兼容** — 旧代码不动，新代码用新方式
5. **简单优先** — 能用装饰器解决的不引入框架
