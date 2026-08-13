# FaceLift 多视角 + LivePortrait 动态帧联合重建

## 1. 为什么需要新的输入格式

旧版 `train_reconstruction.py` 从 `optim.pkl` 只读取一组 FLAME 参数，并把这一组
参数用于所有相机。因此，仅把 `frame_xxxxx.png` 追加到旧 `cameras.json` 会让图像
和 FLAME expression/pose 错配。

联合数据把“观测”和“FLAME 状态”显式分开：

- `cameras_joint.json`：每张训练图及其相机，并用 `flame_index` 指向 FLAME 状态；
- `flame_params_joint.npz`：一组全局共享 shape，以及逐状态的 expression、
  global、neck、jaw、eyes；
- `use_for_alignment=true`：只标在 120 张静态多视角图上。72 张动态图不参与
  多视角 landmark 三角化。

当前数据会生成 192 条观测和 73 个 FLAME 状态：

- 120 张 `elev_*.png` -> `flame_index=0`，共享 `optim.pkl` 的静态状态；
- 72 张 `frame_*.png` -> `flame_index=1..72`，分别使用
  `flame_results.npy` 的 expression/pose；
- 72 张动态图的相机完整复制 `elev_0_azim_270.png`；
- shape 只保存一次，默认选择 `optim.pkl`。

`flame_results.npy` 的 9 维 pose 在当前 tracker 输出中按
`jaw(3) + left_eye(3) + right_eye(3)` 解码；动态 global/neck 置零。这一点已在统一
文件中拆成具名数组，不再让训练代码猜测 pose layout。

## 2. 生成统一文件

在项目根目录执行：

```powershell
F:\Anaconda3\envs\headstudio\python.exe tools\prepare_facelift_joint_dataset.py `
  --input-dir outputs\facelift_multiview\00000002
```

当前两个文件已经生成。需要重新生成时加 `--overwrite`。

如果要改用 LivePortrait 的 identity shape，可在重新生成时明确指定：

```powershell
# 时序均值比任取单帧更稳
F:\Anaconda3\envs\headstudio\python.exe tools\prepare_facelift_joint_dataset.py `
  --input-dir outputs\facelift_multiview\00000002 `
  --shape-source liveportrait-mean `
  --overwrite
```

改变 shape 后必须重新执行对齐，并从头开始重建训练；旧 alignment 不能复用。

## 3. 对齐共享 FLAME

```powershell
F:\Anaconda3\envs\headstudio\python.exe tools\align_facelift_flame_joint.py `
  --input-dir outputs\facelift_multiview\00000002
```

输出目录为 `outputs/facelift_multiview/00000002/flame_alignment_joint/`。
`alignment.npz` 是训练需要的全局 FLAME-to-FaceLift 变换；默认还会输出静态
多视角 overlay/contact sheet。只想快速求变换时可加 `--no-render`。

对齐只读取 `use_for_alignment=true` 的静态图，但使用统一文件中最终选定的共享
shape 和静态状态 0。所得一个 Sim(3) 变换供全部动态状态共享。

## 4. 联合训练

```powershell
F:\Anaconda3\envs\headstudio\python.exe train_reconstruction_joint.py `
  --config configs\reconstruction_joint.yaml
```

默认输出到 `outputs/reconstruction_joint/00000002/`。每次抽到观测后，程序会在
计算 FLAME 顶点、UVD 坐标和协方差之前切换到该观测的 expression/jaw/eyes，
因此动态图的监督与其几何状态严格对应。shape、UVD 参数和 alignment 始终共享。

配置中的采样默认按观测均匀，因此比例为 120:72（62.5% 静态多视角、37.5%
LivePortrait）。若希望动态观测约占一半，可改成：

```yaml
sampling:
  source_repeats:
    multiview: 1
    liveportrait: 2
```

这里的 repeat 只是采样权重，不会复制图片或参数。

## 5. 结果检查

优先检查：

1. `flame_alignment_joint/key_views.png` 中轮廓、眼角、嘴角是否对齐；
2. `training_renders/iteration_*.jpg` 中静态视角与动态张嘴帧是否同时收敛；
3. `final_views/metrics.json` 中 `by_source.multiview` 与
   `by_source.liveportrait` 的指标，避免总均值掩盖某一来源退化；
4. 动态帧若出现嘴型整体错位，先核对 `live_pose_layout`，不要通过调学习率掩盖
   pose 解码错误。

原有 `tools/align_facelift_flame.py`、`train_reconstruction.py` 和旧输入文件均未修改，
仍可用于原来的纯静态流程。
