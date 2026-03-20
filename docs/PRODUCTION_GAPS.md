# Askme 生产化差距分析

> 2026-03-21 — 今天建了完整的视觉→推理→行动链路，但以下问题需要在上真机前解决。

## P0 — 必须修复（会撞墙/危险）

### 1. move_robot 绕过安全层
**现状**: `move_tool.py` 直接 `ros2 topic pub /nav/cmd_vel`
**风险**: 无碰撞检测、无急停保护、跟 LingTu 规划器冲突
**修复**: 改走 `dog-control-service :5080` API → `dispatch_capability("rotate", {angle: 90})`
**依赖**: dog-control-service 需要支持 rotate/forward capability

### 2. scan_around 直接旋转
**现状**: subprocess 发 cmd_vel 旋转 + 拍照
**风险**: 同上，且旋转过程中无反馈
**修复**: 通过 control service 请求旋转，或做成 LingTu 的 "scan" 任务类型

### 3. 语义导航绕过 nav-gateway
**现状**: `move_robot(go_to)` 直接 pub `/nav/semantic/instruction`
**风险**: 无任务 ID、无状态跟踪、无超时、不知道到没到
**修复**: 改走 `nav-gateway :5090` API → `POST /api/v1/nav/tasks`

## P1 — 需要限制（安全边界）

### 4. solve_problem bash 权限太大
**现状**: Agent 可执行任意 bash 命令
**风险**: 误删文件、改错配置、重启关键服务
**修复**: bash 白名单 — 只允许诊断类命令：
```
允许: systemctl status *, journalctl, df, free, top, cat /var/log/*, ping
禁止: rm, kill, reboot, systemctl stop/restart (非 askme), apt, pip
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
