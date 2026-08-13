# Mouth → Full 两阶段优化

入口分别是 `train_mouth.py` 和 `train_full.py`。两阶段都有两个互相
独立的消融开关：

```text
第一阶段：--guidance-mode ism | uvd-sfd
第二阶段：--sdedit-mode independent | flame-surface
```

- `ism`：保留 AnimPortrait3D 的原始 null-prompt DDIM inversion ISM。
- `uvd-sfd`：为兼容旧实验目录保留的参数名，实际方法已改为
  UVD-consistent ISM；它只改变 ISM 的噪声联合分布。
- `independent`：原始逐视角 SDEdit。
- `flame-surface`：基于冻结第一阶段表面的多视角一致 SDEdit。

算法细节见 [uvd_consistent_ism.md](uvd_consistent_ism.md) 和
[flame_surface_consistent_sdedit.md](flame_surface_consistent_sdedit.md)。

## 默认优化范围

Mouth 阶段从 Stage-1 的 `model/uvd.ply` 和
`model/reconstruction_params.npz` 初始化，前 500 个 optimizer steps 为
ISM，后 500 步为 SDEdit。当前配置只更新 dental points；非 dental
Gaussian 的梯度和 Adam 动量都被清零。

Full 阶段从完整 mouth 输出继续，前 1000 个 optimizer steps 为 ISM，
后 750 步为 SDEdit。当前 `full_protection` 配置冻结 eyes、mouth 和
dental points；区域 guidance/SDEdit 权重只开启 full 与 face。Full 阶段
在 optimizer step 50/100 执行 densification，raw ISM 和 UVD-consistent
ISM 使用完全相同的 topology、学习率与保护规则。

## 运行

原始基线：

```powershell
F:\Anaconda3\envs\headstudio\python.exe train_mouth.py `
  --reconstruction outputs\reconstruction\00000001 `
  --guidance-mode ism `
  --sdedit-mode independent `
  --gpu 0

F:\Anaconda3\envs\headstudio\python.exe train_full.py `
  --reconstruction outputs\reconstruction\00000001 `
  --guidance-mode ism `
  --sdedit-mode independent `
  --prompt "<当前 identity 的完整外观描述>" `
  --gpu 0
```

完整新方法：

```powershell
F:\Anaconda3\envs\headstudio\python.exe train_mouth.py `
  --reconstruction outputs\reconstruction\00000001 `
  --guidance-mode uvd-sfd `
  --sdedit-mode flame-surface `
  --surface-views 4 `
  --gpu 0

F:\Anaconda3\envs\headstudio\python.exe train_full.py `
  --reconstruction outputs\reconstruction\00000001 `
  --guidance-mode uvd-sfd `
  --sdedit-mode flame-surface `
  --surface-views 4 `
  --prompt "<当前 identity 的完整外观描述>" `
  --gpu 0
```

先做第一阶段 smoke test：

```powershell
F:\Anaconda3\envs\headstudio\python.exe train_mouth.py `
  --reconstruction outputs\reconstruction\00000001 `
  --guidance-mode uvd-sfd `
  --sdedit-mode independent `
  --max-steps 10 `
  --output outputs\smoke\mouth_uvd_ism `
  --gpu 0
```

`--max-steps <= 500`（mouth）或 `<= 1000`（full）时不会进入 SDEdit。
`--dry-run` 只检查入口、资产和最终命令。每次正式运行必须使用新的
输出目录；旧 probability-flow UVD-SFD checkpoint 不允许直接恢复。
ISM/UVD-ISM 的 checkpoint 不能互相续训。SDEdit 则允许从尚未执行任何
SDEdit 更新的同一个阶段边界 checkpoint 分叉；一旦第二阶段已有更新，
再切换 `independent`/`flame-surface` 会被拒绝。

## 消融注意事项

四种组合默认输出目录不会冲突。做“只比较 full 阶段”的消融时，必须
给所有 full 命令显式传入同一组 `--mouth-ply` 和 `--mouth-params`；否则
默认会读取与当前两个 mode 相匹配的 mouth 输出，初始化也随之改变。

Surface SDEdit 默认每步联合 4 个视角，而原始 SDEdit 保持 AnimPortrait3D
的单视角行为。论文应同时报告 GPU memory/耗时，并增加等视角数的
no-memory control，避免把更多 teacher samples 的收益归因于 surface
memory。
