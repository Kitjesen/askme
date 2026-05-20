# Blueprint Catalog

`catalog` 是 blueprint 的产品目录层，用来回答“有哪些蓝图、客户能看到什么、需要哪些配置、能不能交付、怎么验证”。

## 文件职责

- `models.py`: 数据结构。当前只有 `BlueprintSpec`，定义每个 blueprint 需要声明哪些字段。
- `data.py`: 静态数据。放 `BLUEPRINTS`、`ALIASES` 和模块组合常量，不做运行时检查。
- `catalog.py`: 逻辑。负责查询、加载 Runtime、检查模块组成、生成 readiness / delivery / API payload。
- `__init__.py`: 对外兼容入口。旧代码可以继续从 `askme.blueprints.catalog` 导入。

每个 blueprint 的模块列表要完整写在自己的常量里，不要用一个 blueprint 的模块列表去拼另一个 blueprint。

## 修改入口

- 新增或修改客户可见能力、配置项、验证命令：改 `data.py`。
- 改“是否可交付”的判定、返回给 Dashboard/API 的形状：改 `catalog.py`。
- 改 blueprint 元数据字段结构：先改 `models.py`，再同步 `data.py` 和 `catalog.py`。

不要把业务判断写进 `data.py`，也不要把大段静态产品说明塞回 `catalog.py`。
