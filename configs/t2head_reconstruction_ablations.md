# 第二阶段配置与消融

主实验使用 `configs/t2head_reconstruction_vsd.yaml`，公平的 SDS 对照使用
`configs/t2head_reconstruction_sds.yaml`。切换 identity 时必须同步修改：

- `data.reconstruction_dir`
- `system.prompt`

shape 固定、无 `neck_pose`、动态参数仅来自 `chemistry_exp.npy`、复用第一阶段
OpenCV 相机与 FaceLift 对齐，这些是任务定义和坐标正确性，不是可关闭的模块。
原始从零路径仍由 `configs/t2head.yaml` 和原 system registry 提供，未被修改。

## 基线命令

```powershell
$env:THREESTUDIO_LAZY_IMPORT="1"

# 原始从零优化
F:\Anaconda3\envs\headstudio\python.exe launch.py --config configs/t2head.yaml --train --gpu 0

# A0：第一阶段重建直接评估；不补口腔点，也不做参数更新
F:\Anaconda3\envs\headstudio\python.exe launch.py --config configs/t2head_reconstruction_vsd.yaml --validate --gpu 0 system.mouth.points=0

# 主实验：reconstruction + constrained VSD repair
F:\Anaconda3\envs\headstudio\python.exe launch.py --config configs/t2head_reconstruction_vsd.yaml --train --gpu 0

# 公平对照：除 score objective 外相同
F:\Anaconda3\envs\headstudio\python.exe launch.py --config configs/t2head_reconstruction_sds.yaml --train --gpu 0

# SDS 裁剪消融：同时关闭 latent 逐元素裁剪和 Gaussian 参数 global-norm 裁剪
F:\Anaconda3\envs\headstudio\python.exe launch.py --config configs/t2head_reconstruction_sds_no_clip.yaml --train --gpu 0
```

## 单模块消融

以下每一行都单独从 VSD 主配置开始，只追加该行的 override。不要把多项消融
混在一次实验里；固定初始化、相机、chemistry 数据、随机种子和训练步数。

| 模块 | 对照 override | 验证的问题 |
|---|---|---|
| diffusion repair | `system.diffusion_weight=0` | 高频伪影与自然度改善是否来自扩散先验 |
| VSD / SDS | 改用 `configs/t2head_reconstruction_sds.yaml` | 在线分布适配是否比固定 SDS 更少身份漂移和饱和 |
| SDS 梯度裁剪 | 改用 `configs/t2head_reconstruction_sds_no_clip.yaml` | 两层裁剪是否压制了 SDS 的实际参数更新 |
| Gaussian warmup | `system.guidance_warmup_steps=0 system.guidance.vsd_start_step=0 system.guidance.vsd_warmup_sds_steps=0` | 先让 LoRA 认识当前渲染分布是否更稳定 |
| 低频 reference 约束 | `system.reference_weight=0 system.reference_dual_lr=0` | 静态身份、发型、服装和轮廓是否漂移 |
| 低频而非逐像素 replay | `system.reference_resolution=512 system.reference_kernel=1` | 锁死重建伪影是否妨碍修复 |
| adaptive dual | `system.reference_dual_lr=0` | 自动收紧保真约束是否优于固定权重 |
| 参数 trust region | `system.proximal.feature=0 system.proximal.opacity=0 system.proximal.d=0 system.proximal.scale=0 system.proximal.rotation=0` | 参数漂移与驱动不一致是否增大 |
| appearance→geometry | `system.optimization.geometry_start=0` | 从第一步更新几何是否破坏 FLAME 绑定 |
| appearance-only | `system.optimization.d_lr=0 system.optimization.scale_lr=0 system.optimization.rotation_lr=0` | 局部几何自由度是否确有必要 |
| 前景 diffusion mask | `system.diffusion_background_weight=1` | 背景 score 是否污染轮廓、颜色和 opacity |
| 动态 ControlNet | `data.use_mediapipe_condition=false system.guidance.condition_scale=0` | pose-following 是否来自同坐标 FLAME 条件 |
| negative prompt | `system.guidance.cfg_unconditional_source=null` | 饱和、蜡质皮肤、牙齿伪影是否回升 |
| 保守 CFG | `system.guidance.guidance_scale=7.5` | 从零阶段常用的高 CFG 是否造成身份和颜色漂移 |
| 共享 timestep | `system.guidance.coupled_share_t=false` | 同 pose 多视角的梯度方差是否增大 |
| 独立 noise | `system.guidance.coupled_share_noise=true` | 错误共享二维噪声是否损害多视角结构 |
| chemistry 动态训练 | `data.use_dynamic_expression=false` | 动态覆盖对驱动一致性的贡献 |
| 开嘴过采样 | `data.chemistry_open_mouth_oversample=false` | 大开口成功率以及闭嘴泄漏的变化 |
| jaw 异常帧过滤 | `data.chemistry_jaw_outlier_quantile=1` | 极端拟合帧是否导致爆点或嘴部崩坏 |
| expression/eye 过滤 | `data.chemistry_expression_max_norm=0 data.chemistry_eye_max_norm=0` | 失败拟合是否造成眼球爆转、表情闪烁 |
| 口腔表示支撑 | `system.mouth.points=0` | 闭嘴初始化缺少内壁容量是否是无法张嘴的根因 |
| 颜色屏障 | `system.chroma_weight=0` | 显式色度屏障对过饱和的独立贡献 |
| 世界尺度屏障 | `system.world_scale_weight=0 system.mouth.barrier_weight=0` | 极少数 scale/d 爆点是否受控 |

`coupled_mean_grad` 始终保持 `false`：不同视角相同 latent 像素没有三维对应关系，
直接平均会把眼睛、嘴、头发等不同空间位置混在一起，不能作为合法的一致性损失。

固定 topology 也不是运行时开关：训练前可一次性补充口腔支撑，开始训练后不再
densify/prune。这样每个参数都能与初始化一一对应，trust region 和消融结论才可解释。

## 建议报告

- 第一阶段所有静态相机上的 PSNR/SSIM、alpha IoU、identity 相似度；
- 前景 HSV 饱和度、高光截断率和跨视角颜色方差；
- 固定 chemistry 序列在正面/侧前/侧面的 landmark 跟随误差与时序闪烁；
- 按 jaw 开度分桶的开嘴成功、闭嘴泄漏、浮牙/双牙和口腔空洞率；
- 各参数相对初始化的漂移、reference constraint violation 和 dual weight 曲线；
- 最终 Gaussian 数量（默认仅比重建多固定的 1000 个口腔点）。
