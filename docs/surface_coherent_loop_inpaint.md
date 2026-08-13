# Surface-coherent `loop_inpaint` v3：设计、训练逻辑与验证产物

本文只描述当前 `configs/loop_inpaint.yaml` 与 `surface_inpaint/` 中已经实现的
v3 行为。旧的稀疏静态 teacher bank、Detail UV
伪真值等描述均不再适用。

## 1. v3 要解决什么

旧逻辑把少量 pose/view 的 diffusion 图当成独立伪真值反复拟合，存在三个直接
问题：

1. 视角和表情稀疏，训练图之外的驱动 pose 很容易暴露长条 Gaussian 或口腔错误；
2. 单一 UV 对应会把嘴唇、上下牙和口腔混为同一表面，即使它们在 UV 图中位于
   不同区域，也无法处理屏幕空间遮挡、相同/重叠 UV 坐标及半透明混合；
3. Detail 阶段如果把增强图再次投到 UV，会丢失 teacher 已经生成好的像素级牙齿
   细节，并把直接监督退化为又一次平均。

v3 的核心不是“多生成几张图”，而是把监督拆成两类：

- Base：使用 canonical surface 构造跨 pose/view 的一致监督；
- Detail：严格使用同 pose、同标定相机的 teacher RGB 与 edit mask 直接监督。

此外，所有几何、correspondence、teacher、atlas 和训练边界都输出可检查的中间
结果，避免某个模块没有实际生效却仍然完成训练。

## 2. 不可破坏的约束

### 2.1 不使用 global pose、neck pose 和 translation

配置必须保持：

```yaml
pose_control:
  use_global_orient: false
  use_neck_pose: false
  use_translation: false
```

训练、teacher、诊断和最终 driving 均只使用：

- expression；
- jaw pose；
- left-eye pose；
- right-eye pose。

`pose[:, 0:3]` 的 global orientation 和 `pose[:, 3:6]` 的 neck pose 不进入
训练 batch；translation 始终为零。`UVDAvatar.set_pose()` 会再次强制置零并做
断言，因此这不是仅靠配置约定的软限制。

相机始终使用 Stage-1 的标定坐标，不通过头部刚体运动伪造新视角。

### 2.2 Stage 0 后冻结几何

Stage 0 完成后永久冻结：

- Gaussian `_uv`、`_face_idx`、`_d`；
- `_scaling`、`_rotation`；
- FLAME shape 和 topology；
- Gaussian 数量与绑定关系。

训练只更新 SH0 appearance/color 与 opacity，不执行 densify、prune 或重新绑定。

### 2.3 新旧运行不可混用

当前默认输出名为：

```yaml
output:
  root: outputs/inpaint
  name: 00000001_surface_coherent_v3
```

v3 checkpoint schema 为 version 10，并校验 config、实现代码、相机、pose 数据、
edit mask 和冻结的 Stage-1 tensor digest。旧 checkpoint 不能静默加载为 v3。

## 3. 完整训练流程

```text
Stage-1 reconstructed UVD avatar
        |
        v
00_stage1_input
        |
        v
Stage 0: FLAME pose-envelope covariance stabilization
        |
        v
01_geometry_stabilized
        |
        v
Stage 1 / Base: steps 0 ... 9999
  - face absolute UV atlas
  - 4 independent oral intrinsic-residual atlases
  - 50% arbitrary canonical supervision
  - 50% exact same-pose/same-view direct teacher supervision
        |
        v
02_coherent_base
        |
        v
Stage 2 / Detail: steps 10000 ... 12999
  - Base atlases frozen
  - 100% exact same-pose/same-view teacher RGB/mask
  - no UV pseudo target and no UV confidence gate
        |
        v
03_detail_refinement
        |
        v
final UVD/world PLY + 72-frame driving render
```

## 4. Stage 0：长条 Gaussian 稳定

UVD Gaussian 的 canonical covariance 会经过每个 FLAME pose 的局部 Jacobian
变换到世界空间：

```text
C_world = J_pose C_uvd J_pose^T
```

只在 reference pose 检查 scale 不够；某个 Gaussian 可能在 reference 中正常，
却被开嘴 pose 的 Jacobian 拉成长条。稳定器扫描：

- reconstruction reference pose；
- chemistry jaw-x 的 `0/25/50/75/90/98/99.5%` 分位 pose；
- validation open-mouth pose。

世界协方差主尺度记为 `s0 >= s1 >= s2`，默认违规条件为：

```text
absolute violation: s0 > 0.080
planar streak:       s0 > 0.035 and s0/s1 > 8
```

这里使用 `s0/s1`，而不是 `s0/s2`。正常 surface disk 的法向厚度 `s2` 本来就应
很小，不能因此被误判成长条。

违规 Gaussian 只缩短世界空间最大主轴，再通过逆 Jacobian 拉回 UVD covariance。
默认做 3 个 pass，并用 `repair_margin: 0.90` 留出跨 pose 数值余量。该阶段不删除
Gaussian，也不改变 UV、face binding、normal offset 或 topology。

## 5. 五层 semantic surface correspondence

### 5.1 固定语义层

v3 不再把整个头部当成一个 UV 表面。每个 Gaussian 根据 FLAME face binding
被划入且只划入以下一层：

| ID | 名称 | 作用 |
|---:|---|---|
| 0 | `face` | 排除四个口腔层后的面部表面 |
| 1 | `lips` | 嘴唇 |
| 2 | `teeth_upper` | 上牙 |
| 3 | `teeth_lower` | 下牙 |
| 4 | `oral_cavity` | 口腔内部 |
| -1 | invalid | 无可靠 correspondence |

上下牙即使在 UV 中处于不同区域，也仍必须保留语义层。原因是问题不只来自 UV
坐标位置，还来自：

- 屏幕像素中 lips/teeth/cavity 的前后遮挡；
- 不同拓扑区域可能存在相同或重叠数值 UV；
- alpha compositing 会让一个像素同时含多个表面；
- 对 UV/语义 ID 做普通双线性缩放会重新把边界平均。

所有 correspondence key 都是：

```text
(semantic_layer_id, canonical_uv_texel)
```

因此 `teeth_upper` 与 `teeth_lower` 永远不会因为数值 UV 接近而共享 attention
memory、noise 或 atlas。

### 5.2 每层独立 raster buffer

每层输出：

- UV 一阶/二阶矩；
- layer-only alpha；
- UV variance；
- expected depth；
- occlusion-aware correspondence contribution；
- actual appearance contribution。

`correspondence contribution` 使用
`max(current_opacity, initial_opacity)`，并给 oral Gaussian 设置 `0.03` 的
opacity floor。它只用于建立稳定对应，不参与 RGB 训练，避免当前牙齿 opacity
很低时 correspondence 本身消失。

`appearance contribution` 则使用 Gaussian 当前真实 opacity，表示该层当前对
屏幕 RGB 的实际贡献。Base 口腔 residual 的反合成必须使用这个值，不能使用为
对应关系人为抬高过的 opacity。

### 5.3 主导层与歧义剔除

每个屏幕像素按 occlusion-aware contribution 选择主导层。默认条件为：

```yaml
fusion:
  layered_surface:
    contribution_threshold: 0.005
    dominance_ratio: 1.20
```

第一、第二候选贡献过于接近时，像素被标为 ambiguous，不进入跨表面融合。UV ID
下采样使用 nearest，避免在嘴唇/牙齿边界插值出不存在的语义层。

## 6. Teacher：24 视角、五层 noise 与 correspondence attention

### 6.1 视角覆盖

当前 teacher 配置为：

```yaml
teacher:
  view_sampling: stratified_all_rings
  views_per_pose: 24
  batch_size: 2
```

24 个相机从全部 5 个 calibrated elevation ring 分层选取，而不是只使用一条
水平相机环。`save_all_teacher_observations: true` 会持久化每个 pose 的全部
24 个 observation；`teacher_previews_per_pose` 只有关闭全量保存时才控制预览
抽样，不会改变 teacher 实际视角数。

### 6.2 五个 canonical noise atlas

`SurfaceNoiseAtlas` 的状态为：

```text
[5 semantic layers, 4 latent channels, 1024, 1024]
```

同一个 `(layer_id, UV)` 在不同 pose/view 中采到相关噪声；不同 layer 即使 UV
相同也采到不同噪声。背景使用独立随机噪声，并在轮廓处以保持单位方差的方式与
surface noise 混合。

### 6.3 去噪 self-attention 中传播 surface feature

U-Net self-attention processor 在原 processor 之后执行 correspondence
propagation。每个 attention token 被量化到：

```text
layer_id * 64^2 + uv_texel_id
```

同一 CFG branch 内、至少被 2 个 joint views 看到的 texel 才建立 feature
memory；conditional 与 unconditional branch 不互相泄漏。传播 gate 随 5 个
DDIM transition 的 denoise progress 从 0 增长到最多 `0.65`。

当前 teacher micro-batch 为 2，因此一次 attention 调用在两个 joint views
之间传播；完整 24-view 的全局一致性由跨 observation 的 canonical fusion
承担。配置 `require_runtime_activity: true` 会检查 context、self-attention
调用和 joint-view 数，attention 未实际执行时本次 refresh 直接失败。

口腔层 contribution 可能很弱，所以 attention 与 layered surface 均使用
`0.005` 阈值。配置校验会拒绝
`surface_attention.alpha_threshold > layered_surface.contribution_threshold`，
避免牙齿 correspondence 先被 renderer 接受、后在 attention 中又被丢弃。

## 7. Base：face absolute atlas + 四个 oral intrinsic-residual atlas

### 7.1 刷新计划

Base 范围是 step `0 ... 9999`，refresh step 为：

```text
0, 1000, 2000, ..., 9000
```

每次 5 poses × 24 views，共：

```text
10 refreshes × 5 poses × 24 views = 1,200 teacher observations
```

step 0 使用 reference、jaw 分位和 validation open-mouth 的确定性 envelope；
后续 refresh 使用 reference 加 jaw-stratified chemistry poses，并要求数据存在时
包含 open-mouth 样本。这里没有 global/neck/translation。

### 7.2 face atlas 存绝对 RGB

只有主导层为 `face` 的像素进入共享 face atlas。它存：

- absolute RGB；
- confidence；
- edit probability；
- independent camera support；
- cross-view variance。

融合使用 per-camera 去重、two-pass Huber consensus、最少 2 个独立相机 support
和 confidence-aware history merge。训练时，face atlas 可以由任意动态 FLAME
pose 和标定视角通过 surface UV 查询。

### 7.3 oral atlas 不存 composited teacher RGB

四个独立 atlas 分别为：

- `lips`；
- `teeth_upper`；
- `teeth_lower`；
- `oral_cavity`。

如果直接把 teacher 屏幕 RGB 写入牙齿 atlas，即使牙齿 UV 位于图集下方，该 RGB
仍然已经混入前景嘴唇、其他口腔层和背景。这是 3DGS alpha compositing 的结果，
不是 UV 区域划分错误。

v3 对每个口腔层执行 actual-appearance 反合成。记：

```text
T         = diffusion teacher RGB
R         = 当前 avatar composited render
c_actual  = 该语义层用当前真实 opacity 得到的屏幕贡献
f         = residual_decomposition_floor = 0.01
```

编码到 atlas 的不是 `T`，而是：

```text
delta_screen  = T - R
delta_surface = delta_screen / max(c_actual, f)
E             = clamp(0.5 + 0.5 * delta_surface, 0, 1)
```

这样 teacher 未修改的 lips/background 会在 `T - R` 中抵消；再除以牙层的真实
appearance contribution，得到该表面需要承担的近似 intrinsic RGB 改变量。

每次 Base refresh 同时保存当前每个 Gaussian 的 RGB snapshot `C_ref`。训练前按
该层 Gaussian 的 UV 查询 encoded residual，并解码为：

```text
C_target = clamp(C_ref + 2 * (E - 0.5), 0, 1)
weight   = confidence^confidence_power * edit * semantic_region
```

口腔 atlas residual 相对于本次 refresh 的 `C_ref` 定义，因此每次 refresh
替换四个 oral residual atlas，而不是把不同 reference frame 的 residual 做 EMA。
前一次优化结果已包含在新的 `C_ref` 中。face absolute atlas 仍可做 history merge。

### 7.4 牙齿监督 fail-fast

默认配置要求：

```yaml
fusion:
  layered_surface:
    required_effective_layers: [teeth_upper, teeth_lower]
    minimum_effective_gaussians: 1
```

Base refresh 后会把每层 atlas 解码并采样回绑定 Gaussian，计算
`effective_gaussians`、`effective_weight_sum` 和 target delta。任一要求的牙层
没有至少 1 个有效 Gaussian 时立即报错，不能在“teacher 图里有牙，但实际牙层
loss 为零”的状态下继续训练。

### 7.5 Base 每一步如何采样

```yaml
detail_supervision:
  base_direct_probability: 0.5
```

每个 Base optimizer step：

- 50%：从当前 direct bank 选一个 pose，再在该 pose 内采相机，严格重建同
  FLAME pose、同 calibrated camera，直接拟合 teacher RGB/edit mask；
- 50%：从动态训练 loader 取任意 pose/view，face 使用 absolute atlas 查询；
  四个 oral layer 使用 per-Gaussian decoded residual loss。

canonical face 分支会从 identity/alpha preservation mask 中扩张保护 oral 区，
避免 face identity loss 把牙齿和口腔更新回滚。口腔 opacity 最大允许相对初始值
增加 `0.35`；非口腔仍限制为 `0.10`。

## 8. Detail：只使用 exact same-pose/same-view direct supervision

Detail 范围是 step `10000 ... 12999`，refresh step 为：

```text
10000, 10500, 11000, 11500, 12000, 12500
```

每次 4 poses × 24 views，共：

```text
6 refreshes × 4 poses × 24 views = 576 teacher observations
```

每次包含 reference 加 jaw-rank-stratified chemistry poses；数据提供 open-mouth
样本时，最高开嘴层强制包含它。teacher timestep 随 refresh 线性变化：

```text
400, 324, 248, 172, 96, 20
```

每个 observation 把以下内容无损持久化：

- teacher RGB：RGB uint8 PNG；
- edit mask：L uint8 PNG；
- 完整 expression/jaw/eyes；
- camera data index 与 camera frame index；
- 文件 SHA-256 与 config/implementation provenance。

训练采样时先选择一个 pose group，再从该组取相机。程序用保存的 pose 重建
batch，并断言实际 frame ID 与 manifest 完全一致。

Detail 的关键边界是：

```text
target = saved teacher RGB at the exact saved pose and camera
mask   = saved edit mask at the exact saved pose and camera
```

Detail 不执行以下操作：

- 不把增强图投影成 UV atlas；
- 不用 UV atlas 作为伪真值；
- 不使用 UV confidence 对 edit mask 再做 gate；
- 不从 arbitrary pose/view 查询 Detail target；
- 不继续更新 Base atlas；
- 不使用 Base oral atlas loss。

`03_detail_refinement/atlas/` 仍会导出，作用只是检查冻结的 Base reference，不能
据此理解为 Detail 使用了 UV 监督。

## 9. 损失与参数更新

直接分支和 face-atlas 分支共享：

```text
edit:
  masked L1
  + SSIM
  + high-pass residual L1

non-edit identity:
  full-resolution identity L1
  + low-pass identity L1

regularization:
  alpha preservation
  + feature proximal
  + opacity proximal
  + out-of-range chroma penalty
```

Base canonical 分支额外加入四层 oral decoded target loss：

```text
layered_oral_weight = 2.0
```

只更新：

- `_features_dc` / SH0 color；
- `_opacity`。

每个 optimizer step 后执行 gradient clipping、非口腔/口腔分开的 opacity 上界，
并断言固定几何 tensor 未改变。

## 10. 四阶段和 teacher 的完整验证输出

默认根目录：

```text
outputs/inpaint/00000001_surface_coherent_v3/
```

顶层主要产物：

```text
resolved_config.yaml
train.log
metrics.jsonl
stage_index.json
geometry_stability.json
geometry_stability_comparison.jpg
driving_comparison.jpg
diagnostic_comparison.jpg
metrics_summary.json
_RUN_SUCCESS.json

00_stage1_input/
01_geometry_stabilized/
02_coherent_base/
03_detail_refinement/

teacher/
atlas/
previews/
checkpoints/
model/
test_render/
```

四个编号 stage 是 run root 的直接子目录，不存在额外的 `stages/` 容器。每个
完成的 stage 至少包含：

```text
model/
diagnostics/
driving/
metrics.json
_SUCCESS.json
```

`_SUCCESS.json` 只有在该阶段所有启用产物写完后生成。顶层
`_RUN_SUCCESS.json` 还会验证四个 stage、全部计划内 teacher refresh 和 comparison
产物，不能只因训练循环结束就生成。

### 10.1 每阶段 diagnostics

四个 stage 使用相同 pose/view 诊断网格和同一条 72-frame driver。每个
`diagnostics/` 包含：

```text
render_grid.jpg
surface_layer_grid.jpg
oral_correspondence_grid.jpg
oral_appearance_contribution_grid.jpg
metrics.json

renders/
alpha/
surface_validity/
surface_layer_id/
surface_layers/
  face/
  lips/
  teeth_upper/
  teeth_lower/
  oral_cavity/
```

每个 `surface_layers/<layer>/` 又分别保存：

```text
uv/
alpha/
variance/
depth/                  # PNG + NPY
visibility/             # correspondence contribution
appearance_contribution/
```

这里必须同时看 `visibility` 与 `appearance_contribution`：前者回答“这个表面是否
有稳定 correspondence”，后者回答“它当前对 RGB 实际贡献了多少”。牙齿只在
前者非零、后者接近零时，说明对应关系存在但当前牙齿仍太透明。

### 10.2 每次 teacher refresh

每个 `teacher/step_xxxxxx/` 包含：

```text
00_current_render_grid.jpg
01_flame_condition_grid.jpg
02_surface_teacher_grid.jpg
03_edit_mask_grid.jpg
04_surface_layer_grid.jpg
05_oral_correspondence_grid.jpg
06_oral_appearance_contribution_grid.jpg
pose_XX_*/view_*.jpg
direct_bank/
manifest.json
_SUCCESS.json
```

单 view strip 同时展示 current、condition、teacher、edit mask、semantic layer、
oral correspondence 和 oral actual RGB contribution。`manifest.json` 逐
observation 记录 pose、jaw、相机方位、各层 correspondence/appearance 贡献、
teacher delta、高频 delta 和 attention runtime 增量。

`direct_bank/` 包含：

```text
targets/observation_*.png
edit_masks/observation_*.png
manifest.json
_SUCCESS.json
```

refresh 崩溃重试时，不覆盖已完成 bank，而是写
`direct_bank_retry_XX/` 并归档旧 outer success marker。

### 10.3 Base atlas 的可视化

`02_coherent_base/atlas/face/`：

```text
rgb.png
confidence.png
edit.png
support.png
variance.png
state.pt
```

四个 oral layer 各自输出：

```text
residual_encoded.png
reference_rgb.png
target_delta_encoded.png
decoded_target.png
effective_weight.png
confidence.png
edit.png
support.png
variance.png
state.pt
```

同时保存：

```text
atlas/semantic_layer_reference_rgb.pt
atlas/encoding.json
```

判断牙层监督是否真的进入训练时，优先看：

1. `teeth_upper|teeth_lower/decoded_target.png` 是否包含合理牙齿目标；
2. `effective_weight.png` 是否在牙齿 Gaussian 区域非零；
3. `metrics.json` 中 `effective_gaussians` 与
   `effective_weight_sum` 是否非零；
4. teacher 的 `05_oral_correspondence_grid.jpg` 和
   `06_oral_appearance_contribution_grid.jpg` 是否符合预期。

只看 `residual_encoded.png` 容易误判，因为中性 residual 的编码值就是 0.5。

### 10.4 Base 与 Detail 的直接监督证据

`02_coherent_base/` 同时保存：

```text
supervision_grid.jpg          # canonical atlas supervision
direct_supervision_grid.jpg   # Base 的 exact direct 分支
```

`03_detail_refinement/` 保存：

```text
supervision_grid.jpg
direct_supervision.json
direct_targets/
atlas/                        # 仅冻结 Base reference
```

`direct_supervision.json` 明确记录：

```text
mode: direct_teacher_same_pose_same_view
uv_atlas_used_as_pseudo_ground_truth: false
edit_mask_uv_gated: false
base_atlas: frozen reference only
```

## 11. 运行、校验与恢复

只校验配置、输入文件和刷新预算，不加载 CUDA：

```powershell
F:\Anaconda3\envs\headstudio\python.exe loop_inpaint.py `
  --config configs/loop_inpaint.yaml `
  --mode train `
  --validate-only
```

当前配置应报告：

```text
10 base refreshes / 1200 observations
6 detail refreshes / 576 observations
```

正式训练：

```powershell
F:\Anaconda3\envs\headstudio\python.exe loop_inpaint.py `
  --config configs/loop_inpaint.yaml `
  --mode train
```

精确恢复：

```powershell
F:\Anaconda3\envs\headstudio\python.exe loop_inpaint.py `
  --config configs/loop_inpaint.yaml `
  --mode train `
  --resume outputs/inpaint/00000001_surface_coherent_v3/checkpoints/latest.pt
```

不带 `--resume` 时，如果输出目录已非空，程序会拒绝混入新运行。恢复时 active
direct bank 必须对应已完成 step 的最近计划 refresh；Base atlas 与四个 oral
atlas 必须对应最近 Base refresh，并且在 Detail 中保持冻结。

## 12. 常用调参

### 12.1 Base/Detail 刷新太频繁

```yaml
teacher:
  coarse_refresh_interval: 1000  # Base；增大则更少刷新
  refresh_interval: 500          # Detail；增大则更少刷新
```

刷新预算公式：

```text
Base   = ceil(coarse_iterations / coarse_refresh_interval)
         * coarse_poses_per_refresh * views_per_pose

Detail = ceil((iterations - coarse_iterations) / refresh_interval)
         * poses_per_refresh * views_per_pose
```

这些 interval 也决定 checkpoint 中期望的 active bank/atlas step。修改后不能继续
使用旧 config digest 的 checkpoint。

### 12.2 视角或 pose 数量

```yaml
teacher:
  views_per_pose: 24
  coarse_poses_per_refresh: 5
  poses_per_refresh: 4
```

减少这些值会直接降低 surface coverage 和 teacher 成本；`teacher.batch_size`
只影响联合生成显存和 attention 同时传播的 view 数，不改变 observation 总数。

### 12.3 Base direct/canonical 比例

```yaml
detail_supervision:
  base_direct_probability: 0.5
  open_mouth_probability: 0.5
```

前者控制 Base optimizer step 中 exact direct 的概率；后者控制 direct bank
采样 open-mouth pose 的概率。Detail 始终为 direct，不受
`base_direct_probability` 控制。

### 12.4 牙齿弱可见与 residual 反合成

```yaml
surface_attention:
  alpha_threshold: 0.005

fusion:
  layered_surface:
    contribution_threshold: 0.005
    residual_decomposition_floor: 0.01
    opacity_floor: 0.03
    minimum_effective_gaussians: 1
```

- threshold 太高会把弱可见下牙从 correspondence/atlas 中滤掉；
- `residual_decomposition_floor` 太小会放大极弱 contribution 的噪声，太大则
  补偿不足；
- `opacity_floor` 只影响 correspondence，不代表真实牙齿已经可见；
- 不建议关闭 dental fail-fast，否则会重新出现“teacher 有牙但训练无牙层
  梯度”的静默失败。

### 12.5 牙齿仍然无法显现

检查 `appearance_contribution` 与 `effective_weight` 后再调整：

```yaml
optimization:
  oral_max_opacity_increase: 0.35

loss:
  layered_oral_weight: 2.0
```

不要只提高 opacity 上限。若 `effective_weight` 为零，应先修 correspondence、
dominance 或 teacher coverage；若它非零而 actual appearance contribution 始终
接近零，再评估 opacity 与 oral loss 权重。

## 13. 与 AvatarMakeup 论文的关系

[AvatarMakeup（arXiv:2507.02419）](https://arxiv.org/abs/2507.02419) 的主线是：
先通过 FLAME/mesh 映射构建 global UV guidance，获得跨视角/表情的一致 Base；
再用小 timestep diffusion guidance 做 Detail Refinement；优化时冻结
position/rotation/scale，只更新 feature 与 opacity。

v3 保留并适配了这条 coarse-to-fine 思路，但任务从 makeup transfer 改成
reconstructed head 的 mouth/detail repair。以下是本工程扩展，不应声称为论文
原有模块：

- FLAME/UVD 五语义层 correspondence；
- `(layer_id, UV)` self-attention feature propagation；
- 五层 canonical surface-correlated noise；
- oral actual-appearance residual 反合成；
- teeth effective-supervision fail-fast；
- pose-envelope covariance repair；
- Base 50% exact direct 混合监督；
- Detail direct-bank provenance、断言和全阶段产物。

这也解释了它与简单 AnimPortrait3D 风格“render → diffusion → 拟合若干 2D
target”的差别：v3 的 Base 监督由持久化且分层的 canonical surface field 约束，
Detail 才有意回到同 pose、同 view 的像素级直接拟合，并依赖 Base 已建立的一致
结构作为起点。

## 14. 当前验证边界

`--validate-only` 只验证配置、Stage-1 输入和刷新预算；单元测试与 CUDA smoke
可以验证 renderer、五层 correspondence、attention/noise、atlas/direct loss
和 backward 链路。只有完成新的 13,000-step v3 正式训练，并人工检查四阶段
diagnostics 与最终 72-frame driving video，才能判断口腔和长条伪影的最终视觉
质量。不能仅凭 teacher contact sheet 宣称 3D avatar 已经学到对应细节。
