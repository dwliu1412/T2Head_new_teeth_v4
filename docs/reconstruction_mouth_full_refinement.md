# Mouth → Full 两阶段优化

入口分别是 `train_mouth.py` 和 `train_full.py`。两阶段都有两个互相
独立的消融开关：

```text
第一阶段：--guidance-mode ism | uvd-sfd
第二阶段：--sdedit-mode independent | flame-surface
```

- `ism`：保留 AnimPortrait3D 的原始 null-prompt DDIM inversion ISM。
- `uvd-sfd`：使用同一份 CFD/UVD 一致噪声直接构造 `x_t` 与 `x_s`，
  优化文本条件的 `t` 时刻预测与空提示的 `s` 时刻预测之差；不再执行
  null-prompt DDIM inversion。
- `independent`：原始逐视角 SDEdit。
- `flame-surface`：基于冻结第一阶段表面的多视角一致 SDEdit。

算法细节见 [uvd_consistent_ism.md](uvd_consistent_ism.md) 和
[flame_surface_consistent_sdedit.md](flame_surface_consistent_sdedit.md)。

## 默认优化范围

Mouth 阶段从 Stage-1 的 `model/uvd.ply` 和
`model/reconstruction_params.npz` 初始化，前 500 个 optimizer steps 为
所选 guidance（ISM 或 UVD-SFD），后 500 步为 SDEdit。当前配置只更新 dental points；非 dental
Gaussian 的梯度和 Adam 动量都被清零。

Full 阶段从完整 mouth 输出继续，前 1000 个 optimizer steps 为所选 guidance，
后 750 步为 SDEdit。当前 `full_protection` 配置冻结 eyes、mouth 和
dental points；区域 guidance/SDEdit 权重只开启 full 与 face。默认保持 mouth
输出的 topology，不执行 densification；只有显式传入 `--densification-steps`
才会开启该实验。尺度约束仅使用 aligned world-space scale，不再对随父面大小
变化的 face-local scale 使用固定阈值。当前 full 将 world 三轴逐轴硬限制在
`0.05`，并约束每点相对输入 mouth 状态的逐轴增长和轴比增长不超过 `1.5×`；
scale 学习率为原 GSAvatar full 设置的十分之一，以避免 ISM 在最初几步生成针状
高斯。

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
  --output outputs\smoke\mouth_uvd_sfd `
  --gpu 0
```

`--max-steps <= 500`（mouth）或 `<= 1000`（full）时不会进入 SDEdit。
`--dry-run` 只检查入口、资产和最终命令。每次正式运行必须使用新的
输出目录；旧版 UVD 目标的 checkpoint 不允许直接恢复。
ISM/UVD-SFD（以及旧版 UVD 目标）的 checkpoint 不能互相续训。SDEdit
则允许从尚未执行任何
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
