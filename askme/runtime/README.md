# Runtime Package

`askme.runtime` is the application runtime engine. It does not decide which
product to run; `askme.blueprints` does that. Runtime only knows how to build,
wire, start, stop, and inspect modules.

Start here when you need to understand or change runtime behavior:

```text
runtime/
  core/module.py     runtime engine: Module, In/Out ports, auto-wire, Runtime, RuntimeApp
  modules/           concrete modules: LLMModule, VoiceModule, PipelineModule, ControlModule
  task/              mission handoff, runtime arbiter, callbacks, audit
  examples.py        small runnable examples for learning the module system
```

Subpackage ownership:

- `core/`: runtime engine and wiring primitives.
- `modules/`: concrete runtime modules used by blueprints.
- `task/`: mission handoff, arbiters, callbacks, and audit helpers.

## Responsibilities

Runtime owns the mechanics of a running app:

- Declare modules, dependencies, and typed ports.
- Instantiate modules from a `Runtime` graph.
- Auto-wire matching `In[T]` and `Out[T]` ports.
- Build modules in dependency order.
- Start, stop, inspect health, and expose flow statistics through `RuntimeApp`.

Runtime does not own product selection, customer readiness metadata, or delivery
packaging. Those belong to `askme.blueprints`.

## Public Entrypoints

Use these imports for new runtime code:

```python
from askme.runtime.core import (
    Alias,
    In,
    Module,
    ModuleRegistry,
    Out,
    Required,
    Runtime,
    RuntimeApp,
)
```

Common public objects:

- `Module`: base class for one runtime part.
- `Runtime`: declarative graph builder, usually via `Runtime.use(...)`.
- `RuntimeApp`: built app returned by `await runtime.build(cfg)`.
- `ModuleRegistry`: build-time lookup passed to module `build(...)`.
- `In[T]` / `Out[T]`: typed wiring markers.
- `Required[T]`: marks a dependency as build-required.
- `Alias[T, "module_name"]`: disambiguates a provider when several modules
  expose the same type.

Compatibility imports such as `askme.runtime.module` still work, but new code
should import from `askme.runtime.core`.

## Mental Model

Runtime has four concepts:

```text
Module      one runtime part, such as VoiceModule or PipelineModule
In/Out      typed ports used to connect modules
Runtime     declarative module list built with Runtime.use(...)
RuntimeApp  built runtime instance that can start(), stop(), health()
```

The call chain is:

```text
blueprints/presets/*.py
  -> Runtime.use(ModuleA) + Runtime.use(ModuleB)
  -> runner.py calls await blueprint.build(cfg)
  -> Runtime.build(cfg)
       instantiate modules
       auto-wire In/Out ports
       sort by depends_on
       call each module.build(cfg, registry)
  -> RuntimeApp.start()
       call each module.start()
```

## Smallest Example

Run the self-contained demo:

```powershell
python -m askme.runtime.examples
```

The example defines one provider module and one consumer module:

```python
from askme.runtime.core import In, Module, ModuleRegistry, Out, Runtime


class ClockService:
    def now(self) -> str:
        return "09:30"


class ClockModule(Module):
    name = "clock"
    provides = ("clock",)

    clock: Out[ClockService]

    def build(self, cfg: dict, registry: ModuleRegistry) -> None:
        self.service = ClockService()


class GreeterModule(Module):
    name = "greeter"
    depends_on = ("clock",)

    clock: In[ClockService]

    def build(self, cfg: dict, registry: ModuleRegistry) -> None:
        self.message = f"ready at {self.clock.service.now()}"


runtime = Runtime.use(ClockModule) + Runtime.use(GreeterModule)
app = await runtime.build({})
await app.start()
print(app.greeter.message)
await app.stop()
```

Important detail: `clock: In[ClockService]` receives the provider module
instance, not the raw `ClockService`. That is why the consumer uses
`self.clock.service.now()`.

## How To Write A Module

Use this shape:

```python
class MyModule(Module):
    name = "my_module"
    depends_on = ("other_module",)
    provides = ("something",)

    dependency: In[SomePortType]
    output: Out[MyOutputType]

    def build(self, cfg: dict, registry: ModuleRegistry) -> None:
        self.client = ...

    async def start(self) -> None:
        ...

    async def stop(self) -> None:
        ...

    def health(self) -> dict:
        return {"status": "ok"}
```

Rules:

- Put product composition in `askme.blueprints.presets`, not in `runtime`.
- Put reusable runtime wiring in `runtime/core`.
- Put concrete runtime parts in `runtime/modules`.
- `depends_on` controls build/start order.
- `In[T]` and `Out[T]` control automatic wiring.
- Use `Required[T]` only when missing dependency must fail build.
- Use `Alias[T, "module_name"]` when multiple modules provide the same type.

## Existing Product Compositions

Look at these for real project wiring:

- `askme/blueprints/presets/text.py`
- `askme/blueprints/presets/voice.py`
- `askme/blueprints/presets/voice_perception.py`
- `askme/blueprints/presets/edge_robot.py`

For example, `voice.py` is only a module list:

```python
voice = (
    Runtime.use(LLMModule)
    + Runtime.use(ToolsModule)
    + Runtime.use(MemoryModule)
    + Runtime.use(PipelineModule)
    + Runtime.use(SkillModule)
    + Runtime.use(VoiceModule)
    + Runtime.use(TextModule)
    + Runtime.use(HealthModule)
)
```

The startup helper in `blueprints/runner/runner.py` later calls:

```python
app = await blueprint.build(cfg)
await app.start()
```

## How To Assemble From A Blueprint

New product runtimes are usually assembled in a blueprint preset, not in
`runtime`:

```python
from askme.runtime.core import Runtime
from askme.runtime.modules import HealthModule, LLMModule, TextModule

text_like = (
    Runtime.use(LLMModule)
    + Runtime.use(TextModule)
    + Runtime.use(HealthModule)
)
```

Then the preset can expose the runtime object:

```python
__all__ = ["text_like"]
```

Callers build and run it the same way as existing presets:

```python
app = await text_like.build(cfg)
await app.start()
try:
    print(app.health())
finally:
    await app.stop()
```

For command-line startup, presets delegate to:

```python
from askme.blueprints.runner.runner import run_blueprint

run_blueprint(text_like, "Text-like runtime")
```

## What To Edit

```text
Need to change how modules connect?       runtime/core/module.py
Need to change one concrete module?       runtime/modules/<name>_module.py
Need to change which modules run?         blueprints/presets/*.py
Need customer readiness metadata?         blueprints/catalog/catalog.py
Need mission handoff behavior?            runtime/task/
```

## Verification

For runtime documentation and examples:

```powershell
python -m compileall askme/runtime/examples.py
python -m askme.runtime.examples
```

For a blueprint composition smoke check without opening runtime IO:

```powershell
python -m askme.blueprints.presets.text --preflight --json
```

Legacy imports remain available while callers are migrated.
