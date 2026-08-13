# Mouth / Full refinement 无变化：根因与重跑说明

## 已确认的根因

`outputs/refinement/00000001/mouth` 不是“优化很弱”，而是没有发生一次
有效的 Adam 更新：

- `epoch=0-step=100.ckpt` 的 AMP scale 已降到 `5.17e-26`；
- step 200 后 scale 为 `0.0`；
- mouth 所有 checkpoint 的 optimizer state 数量均为 0；
- step 100 到 1000 之间，feature / opacity / UVD / rotation 逐位不变；
- 验证图 step 33 与 step 1000 只有 18/786432 个通道值相差 1。

原因是参考 `GSAvatar/train_mouth.py` 用 FP32 对 Gaussian 做反向传播，迁移版却把
419745 个 UVD Gaussian 的 mouth ISM 放进了 `16-mixed`。每次反向传播都溢出，
GradScaler 因而跳过 optimizer；Lightning 的 global step 仍继续增长，最终静默输出
了无变化的 PLY。

## 同时修复的迁移偏差

- mouth Gaussian 优化改为 `32-true`，扩散模型权重仍可保持 fp16；
- 恢复参考 prompt：`mouth region, <abstract prompt>`；
- mouth/full 的 pitch=70..110 正确映射为 elevation=-20..20；
- full 恢复每个 ISM step 的 3 次独立 pose/view 梯度累积；
- full 恢复参考的 feature/opacity 学习率、CFG=100 和相同的眼部区域权重；
- full 改用与 mouth 相同的 AnimPortrait3D 四通道 ControlNet、显式 VAE 和
  `animportrait3d` ISM；全局分支不使用 ControlNet，face/eye/mouth 使用固定
  landmark crop 放大到 512×512；
- 恢复原版 full 的 UVD/scale/rotation 学习率、50/100 步屏幕梯度 densification
  和 70% 正面半球采样；移除会饱和并拉回旧 artifacts 的自适应 replay/proximal；
- 连续 8 次 optimizer 未执行会立即终止；导出前也会拒绝零更新结果；
- mouth sidecar 记录真实 optimizer step，full 会拒绝旧的 no-op 输出和短 smoke 输出。

## 必须从 Stage-1 重新跑

旧 mouth checkpoint 的 Adam state 为空，不能 `--resume`。使用新目录：

```powershell
F:\Anaconda3\envs\headstudio\python.exe train_mouth.py `
  --reconstruction outputs\reconstruction\00000001 `
  --prompt "A man" `
  --output outputs\refinement\00000001\mouth_fixed `
  --gpu 0

F:\Anaconda3\envs\headstudio\python.exe train_full.py `
  --reconstruction outputs\reconstruction\00000001 `
  --mouth-ply outputs\refinement\00000001\mouth_fixed\save\mouth.ply `
  --mouth-params outputs\refinement\00000001\mouth_fixed\save\mouth_params.npz `
  --prompt "a photorealistic portrait of a young man with dark brown pompadour hair, natural skin, realistic ears, wearing a clean white dress shirt and a dark navy suit jacket, natural lips and clean well-separated teeth" `
  --abstract-prompt "a photorealistic young man with natural skin, realistic ears, eyes, natural lips and clean well-separated teeth" `
  --output outputs\refinement\00000001\full_repaired `
  --gpu 0
```

full 必须使用每个 identity 自己的完整外观 prompt；`A man`、`A woman` 等泛化
prompt 会被入口拒绝，因为它们没有提供头发、耳朵和服装的修复目标。

`outputs/refinement/00000001/mouth_fp32_smoke` 是本次单步验证产物，不可作为
full 的输入。
