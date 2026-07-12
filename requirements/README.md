# pip-tools 依赖管理

本目录使用 [pip-tools](https://github.com/jazzband/pip-tools) 管理依赖锁定，
确保可复现的构建。

## 文件结构

| 文件 | 用途 |
|------|----------|
| `main.in` | 核心运行时依赖源（对应 `pyproject.toml [project.dependencies]`） |
| `main.txt` | 核心依赖 lock 文件（由 `main.in` 编译生成） |
| `dev.in` | 开发依赖源（对应 `pyproject.toml [project.optional-dependencies] dev`） |
| `dev.txt` | 开发依赖 lock 文件（由 `dev.in` 编译生成，受 `main.txt` 约束） |

## 工作流

### 安装依赖

```bash
# 仅运行时依赖
pip install -e .

# 运行时 + 开发依赖
pip install -e ".[dev]"

# 全部可选依赖
pip install -e ".[dev,robot,memory,robotmem,mempalace,vision,embed]"
```

> 推荐通过 `pyproject.toml` 的 extras 安装，而非直接使用 `requirements/*.txt`，
> 这样 `pip` 会自动处理可编辑安装的路径别名。锁文件用于 CI/CD 和部署时的可复现性。

### 更新依赖锁

```bash
# 安装或更新 pip-tools
pip install pip-tools

# 更新核心依赖锁
pip-compile --output-file requirements/main.txt requirements/main.in

# 更新开发依赖锁（受 main.txt 版本约束）
pip-compile --constraint requirements/main.txt --output-file requirements/dev.txt requirements/dev.in
```

### 添加新运行时依赖

1. 将包名（不加版本号）添加到 `requirements/main.in`
2. 同时将其添加到 `pyproject.toml` 的 `[project.dependencies]`（带最低版本约束）
3. 重新编译：`pip-compile --output-file requirements/main.txt requirements/main.in`
4. 重新编译 dev：`pip-compile --constraint requirements/main.txt --output-file requirements/dev.txt requirements/dev.in`

### 添加新开发依赖

1. 将包名添加到 `requirements/dev.in`
2. 同时将其添加到 `pyproject.toml` 的 `[project.optional-dependencies] dev`（带最低版本约束）
3. 重新编译：`pip-compile --constraint requirements/main.txt --output-file requirements/dev.txt requirements/dev.in`

## 最佳实践

- `.in` 文件只列包名，不指定版本——版本锁定在 `.txt` 文件中
- 不要手动编辑 `.txt` 文件——它们由 `pip-compile` 自动生成
- 版本冲突时，在 `.in` 文件中添加版本约束（如 `package>=X,<Y`）
- 提交 `.in`、`.txt` 和 `README.md` 到版本控制
