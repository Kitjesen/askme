# Askme 生产化差距分析

> 2026-03-21 — 今天建了完整的视觉→推理→行动链路，但以下问题需要在上真机前解决。

## P0 — ~~必须修复~~ ✅ 已修复 (2026-03-21)

### 1. ~~move_robot 绕过安全层~~ ✅
**修复**: 改走 `dog-control-service :5080` API (dispatch_capability) + `nav-gateway :5090` API (POST /api/v1/nav/tasks)。不再直接 pub cmd_vel。服务不可用时返回明确错误。

### 2. ~~scan_around 直接旋转~~ ✅
**修复**: 改成读 daemon 预计算检测结果（即时）。360° 旋转通过 control service 请求，不直接 pub cmd_vel。

### 3. ~~语义导航绕过 nav-gateway~~ ✅
**修复**: `move_robot(go_to)` 改走 `nav-gateway :5090` API，有 task_id 和状态跟踪。

## P1 — ~~需要限制~~ ✅ 已修复 (2026-03-21)

### 4. ~~solve_problem bash 权限太大~~ ✅
**修复**: SandboxedBashTool 加命令黑名单：
```
禁止: rm -rf /, reboot, shutdown, kill -9, apt remove, systemctl stop, pip uninstall
允许: systemctl status, journalctl, df, free, top, cat /var/log, ping, grep, ls
```

### 5. web_search 结果未过滤
**现状**: Agent 直接采纳网上搜索结果执行
**风险**: 搜到错误/恶意命令并执行
**修复**: solve_problem prompt 加 "搜索结果仅作参考，执行前必须判断安全性"

## P2 — 生产就绪（功能完善）

### 6. ProactiveAgent + solve_problem 未真机测试
**现状**: auto_solve 回调写好了但没有真机验证过
**测试**: 需要实际触发一次异常 → 验证整个 OTREV 链路

### 7. 音频设备竞争
**现状**: voice_loop 和 HTTP chat 可能同时用 aplay
**修复**: 音频路由器加互斥锁（已有 audio_router 但需要验证）

### 8. Cloud ASR 未启用
**现状**: `cloud_asr.enabled: false`，只用本地 sherpa-onnx
**收益**: DashScope Paraformer 精度更高（尤其噪声环境）
**修复**: `cloud_asr.enabled: true`（已有代码，只需开 config）

## 功能归属

| 功能 | 当前位置 | 正确归属 | 状态 |
|------|----------|----------|------|
| 运动控制 | askme move_tool | runtime control-service | ❌ 需迁移 |
| 语义导航 | askme move_tool | runtime nav-gateway | ❌ 需迁移 |
| 帧抓取 + YOLO | askme frame_daemon | askme | ✅ 正确 |
| VLM 理解 | askme vision_bridge | askme | ✅ 正确 |
| 异常检测 | askme proactive_agent | askme | ✅ 正确 |
| 巡逻决策 | askme proactive_agent | askme | ✅ 正确 |
| 实际巡逻路径 | 未实现 | LingTu patrol 功能 | ⬜ 待接入 |
