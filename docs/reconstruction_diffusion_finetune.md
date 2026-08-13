# 基于重建初始化的可驱动头部高斯扩散微调：设计与消融

## 1. 目标与不可变约束

本阶段不是重新生成一个“看起来像提示词”的静态头部，而是在第一阶段多视角重建所得的 UVD-FLAME 高斯上，修复单图重建带来的跨视角伪影，并保留可驱动性、身份与外观。训练结束后，模型应同时满足：

1. 静态参考视角中不丢失第一阶段已经恢复的身份、发型、服饰和整体色调；
2. 在未见过的 FLAME expression、jaw 和 eye 参数下仍保持几何与纹理一致；
3. 嘴巴能够真实张开，口腔内部不出现空洞、漂浮高斯或粘连牙齿；
4. 扩散先验只负责修正不可靠区域，不能把一个已经收敛的重建重新拉回高方差的“从零生成”过程。

以下约束属于数据定义和坐标正确性，**不作为消融项**：

- shape 固定为第一阶段重建保存的个体 shape，训练全程不更新、不置零；
- 不读取、不采样、不传递 `neck_pose`，颈部姿态恒为零；
- 动态 expression、jaw 和 eye 参数只从 `chemistry_exp.npy` 采样；
- 相机严格复用第一阶段静态重建的 OpenCV 内外参、图像尺寸和裁剪约定；
- FLAME 到 FaceLift/重建世界坐标的均值与协方差变换必须正确；
- RGB、alpha、FLAME 条件图必须由同一组几何、对齐矩阵和相机投影得到。

## 2. 为什么不能照搬从零 SDS

从零文本到 3D 的 SDS 需要用强语义梯度，把随机几何和随机颜色快速推向一个高概率文本概念。这种做法的合理前提是当前参数几乎没有可信信息，因此“大步探索”比“局部保持”更重要。[DreamFusion](https://arxiv.org/abs/2209.14988) 和头部专用的 [HeadStudio](https://arxiv.org/abs/2402.06149) 都主要面对这一类欠约束生成问题。

强初始化后的问题正好相反。第一阶段重建已经给出了一个高质量但有局部伪影的后验中心，扩散模型却不知道该实例的精确身份、跨视角对应关系和 FLAME 驱动规律。若继续使用从零阶段的高 CFG、大时间步、联合几何纹理更新和频繁增删点，会产生四类典型失效：

- **身份漂移**：文本只描述类别属性，不能唯一确定当前人物；扩散梯度会把已恢复的实例拉向数据集平均脸。
- **纹理过饱和**：高 CFG 和反复的二维蒸馏会放大条件分支与负向分支的差值，高频颜色比真实多视角一致性更容易降低单视图噪声残差。
- **驱动不一致**：每个视角若使用不同表情、错误的相机语义或不一致的 ControlNet 投影，优化器会把姿态变化烘焙进静态高斯属性。
- **嘴部塌陷**：二维扩散先验倾向用暗纹理“画出”张嘴，而不是构造随下颌运动的唇缘、牙齿和口腔遮挡关系。

因此第二阶段不再写成一个任意加权的“大杂烩 loss”，而定义为**受约束的后验局部修复**：

\[
\begin{aligned}
\min_\theta\quad&
\mathbb E_{c,\xi}\!\left[\mathcal L_{\mathrm{diff}}
+ \mathcal L_{\mathrm{prox}}
+ \mathcal L_{\mathrm{barrier}}
+ \mathcal L_{\mathrm{chroma}}\right],\\
\text{s.t.}\quad&
\mathbb E_{c}\!\left[\mathcal L_{\mathrm{ref}}(\theta;c)\right]
\le
\underbrace{\mathbb E_c[
\mathcal L_{\mathrm{ref}}(\theta_0;c)]}_{B_0}
+\varepsilon,\\
&u_i=u_i^0,\ v_i=v_i^0,\ f_i=f_i^0,\ 
\beta=\beta^\star,\ N=N_0+N_{\mathrm{mouth}} .
\end{aligned}
\]

\(\theta_0\) 是第一阶段重建加一次性口腔初始化后的参数；\(c\) 是第一阶段标定相机，
\(\xi\) 是 chemistry pose。第二行不是希望 reference loss “越小越好”，而是要求
扩散修复不能让可靠的低频实例证据比初始化恶化超过容差 \(\varepsilon\)。
第三行把 UV、face binding、shape 与训练期 topology 变成硬约束，避免用软权重
假装固定。

实现使用投影 primal-dual 更新。忽略与 \(\theta\) 无关的常数后，Gaussian 的目标为

\[
\mathcal L_\theta =
\mathcal L_{\mathrm{diff}}+\mathcal L_{\mathrm{prox}}
+\mathcal L_{\mathrm{barrier}}+\mathcal L_{\mathrm{chroma}}
+\lambda\,\mathcal L_{\mathrm{ref}},
\]

dual 更新为

\[
\lambda_{k+1}=\Pi_{[\lambda_{\min},\lambda_{\max}]}
\left[
\lambda_k+\eta_\lambda
\left(\mathcal L_{\mathrm{ref},k}-B_0-\varepsilon\right)
\right].
\]

这样 reference 没有恶化时不会无止境压制扩散；一旦身份/轮廓开始漂移，约束权重
会自动升高。\(B_0\) 不能取第一批随机视角的误差：实现先冻结全部 Gaussian，
在 warmup 的多相机采样上估计运行均值，再固定该基线。这避免把“侧脸本来更难”
误判成训练导致的身份漂移。

## 3. UVD-FLAME 动态几何

### 3.1 均值

对第 \(i\) 个高斯，保存 UVD 参数

\[
q_i=(u_i,v_i,d_i,f_i),
\]

其中 \(f_i\) 是 FLAME 三角面索引，\((u_i,v_i)\) 确定面内位置，\(d_i\) 是沿该处法线的偏移。给定固定个体 shape \(\beta^\star\) 和动态参数

\[
\xi=(\psi,\theta_{\mathrm{jaw}},\theta_{\mathrm{eye}}),
\]

FLAME 变形后的面内点与法线分别记作
\(S_{f_i}(u_i,v_i;\beta^\star,\xi)\) 和
\(n_{f_i}(u_i,v_i;\beta^\star,\xi)\)。训练坐标中的高斯均值为

\[
\mu_i^{T}(\xi)
=S_{f_i}(u_i,v_i;\beta^\star,\xi)
+d_i\,n_{f_i}(u_i,v_i;\beta^\star,\xi).
\]

这里没有 neck 项；全局平移和全局头部旋转也不应被 chemistry 数据隐式替代。一个多视角训练组内必须共享同一个 \(\xi\)，否则所谓“跨视角监督”实际上观察的是不同三维状态。

### 3.2 协方差

仅变换均值而不变换协方差，会让高斯在反射、缩放或表情形变后具有错误的朝向和屏幕覆盖范围。设从 UVD 局部坐标到当前 FLAME 三维空间的雅可比为

\[
J_i(\xi)=
\frac{\partial \mu_i^T}{\partial(u_i,v_i,d_i)},
\]

UVD 局部旋转和尺度为 \(R_i,s_i\)，则一个通用写法是

\[
\Sigma_i^T(\xi)
=J_i(\xi)R_i\operatorname{diag}(s_i^2)R_i^\top J_i(\xi)^\top.
\]

具体实现可以等价地通过局部 frame 或 deformation gradient 构造，但必须同时保持切向和法向尺度的语义。

### 3.3 FaceLift 对齐

第一阶段保存的 FaceLift 对齐写为仿射矩阵

\[
A=
\begin{bmatrix}
L&t\\0&1
\end{bmatrix}.
\]

渲染坐标中的均值和协方差必须分别为

\[
\mu_i^F=L\mu_i^T+t,\qquad
\Sigma_i^F=L\Sigma_i^T L^\top.
\]

\(L\) 可能包含统一尺度和反射，不能默认它属于 \(SO(3)\)，也不能简单把 \(A\) 乘进相机位姿后仍按刚体相机解释。尤其是反射会改变手性；只旋转 quaternion 而忽略尺度和反射并不等价。正确做法是先在几何侧应用 \(A\)，再用第一阶段的原始相机投影。

## 4. OpenCV 相机必须严格复用

设第一阶段保存的 OpenCV world-to-camera 矩阵为

\[
W_{\mathrm{cv}}=
\begin{bmatrix}R_{\mathrm{cv}}&t_{\mathrm{cv}}\\0&1\end{bmatrix},
\]

内参为

\[
K=
\begin{bmatrix}
f_x&0&c_x\\
0&f_y&c_y\\
0&0&1
\end{bmatrix}.
\]

对 FaceLift 坐标点 \(X^F\)，投影必须遵循

\[
\bar X^C=W_{\mathrm{cv}}\bar X^F,\qquad
p\sim K
\begin{bmatrix}
X^C/Z^C\\Y^C/Z^C\\1
\end{bmatrix}.
\]

若底层 rasterizer 使用 OpenGL 轴向约定，只能在相机适配层做一次明确的轴翻转；不能重新生成“数值看起来接近”的球面相机。以下量必须逐项复用：`w2c/c2w` 的定义、焦距、主点、图像分辨率、像素中心约定、近远裁剪面、RGB/alpha 的 resize 与 crop。相机中心、提示词视角标签和 elevation/azimuth 若需要计算，应把相机中心通过 \(A^{-1}\) 映射回 FLAME 训练坐标后再定义，不能沿用从零阶段假定的“正面方位角”。

建议在训练前做两个不可省略的几何检查：

1. 使用保存的中性 FLAME 状态重渲染全部第一阶段相机，确认轮廓与参考 alpha 对齐；
2. 将同一批 FLAME 顶点投影为条件图，并叠加到 RGB 上检查眼角、唇角、下颌线，误差应只来自 rasterization，而非坐标系。

## 5. SDS、VSD 与强初始化下的用法

### 5.1 SDS

令渲染器输出 \(x=g_\theta(c,\xi)\)，VAE 编码为 \(z_0=E(x)\)。正向加噪为

\[
z_t=\alpha_tz_0+\sigma_t\epsilon,\qquad
\epsilon\sim\mathcal N(0,I).
\]

SDS 使用冻结扩散模型的噪声残差近似参数梯度：

\[
\nabla_\theta\mathcal L_{\mathrm{SDS}}
\approx
\mathbb E_{t,\epsilon,c,\xi}
\left[
w(t)
\left(\hat\epsilon_{\phi}^{\mathrm{cfg}}(z_t,t,y,C)-\epsilon\right)
\frac{\partial z_0}{\partial x}
\frac{\partial x}{\partial\theta}
\right].
\]

\(y\) 是文本，\(C\) 是与当前相机和 FLAME 状态一致的条件图。强初始化阶段应降低大时间步和极高 CFG 的占比，因为大时间步更强调类别语义而非当前实例的局部修复。

### 5.2 CFG 与 negative prompt

以 negative prompt 分支代替无条件分支时，

\[
\hat\epsilon^{\mathrm{cfg}}
=\hat\epsilon_{\mathrm{neg}}
+s\left(\hat\epsilon_{\mathrm{pos}}-\hat\epsilon_{\mathrm{neg}}\right).
\]

negative prompt 应描述可观察的失败模式，例如过曝、过饱和、蜡质皮肤、双牙、漂浮牙齿、变形嘴唇和模糊纹理；它不是几何约束，不能代替相机、FLAME 绑定或口腔建模。CFG 需要比从零阶段保守，并通过消融验证，而不是默认越大越好。

### 5.3 VSD

[ProlificDreamer](https://arxiv.org/abs/2305.16213) 的 VSD 用一个随当前三维分布适配的 LoRA score，减去当前粒子分布已经解释的部分。简化梯度可写为

\[
\nabla_\theta\mathcal L_{\mathrm{VSD}}
\approx
\mathbb E
\left[
w(t)
\left(
\hat\epsilon_{\mathrm{pre}}^{\mathrm{cfg}}
-\hat\epsilon_{\mathrm{LoRA}}
\right)
\frac{\partial z_0}{\partial x}
\frac{\partial x}{\partial\theta}
\right].
\]

与 SDS 相比，VSD 更适合作为强初始化后的默认候选：预训练 score 提供文本和自然图像先验，LoRA score 估计当前渲染分布，二者之差更接近“当前模型还缺什么”，而不是持续把已正确部分推向扩散模型的平均模式。

默认前 300 步固定全部 Gaussian，只用当前初始化的动态渲染训练在线 LoRA，同时
估计 \(B_0\)。第 300 步后才释放颜色与 opacity，并在接下来的 300 步把 score
从 SDS 连续过渡到 VSD；这比在 LoRA 尚未认识当前渲染域时立刻使用二者差值更稳定。
SDS 对照使用相同的 Gaussian 冻结窗口，从而不会把“优化步数更多”混入 VSD/SDS
比较。

同一多视角组可以共享 timestep，以减少不同视角所处噪声难度的方差；噪声本身保持独立即可。不能直接平均不同视角中未对齐的像素梯度，否则会把眼、嘴和发丝等空间位置混合。

## 6. 三类保真机制

### 6.1 静态 reference replay

每次训练迭代使用同一组标定相机做两次有不同目的的渲染：动态 chemistry pose
进入 diffusion；第一阶段保存的 reference pose 进入静态 replay。reference replay
使用对应相机的 RGB 与 alpha。推荐优先约束：

- 低通 RGB 重建，用于保持身份、服饰、发型和全局色调；
- silhouette/alpha，用于保持头部与肩颈外轮廓；

不宜仅使用全分辨率逐像素 L1/L2，因为第一阶段伪影也会被完整锁死。实现用
Smooth-L1、低通卷积和 128×128 重采样保护身份、发型、服装、色调与 silhouette；
前景核心权重大，背景保留弱权重。低频 replay 把高频修复空间留给扩散先验，
其权重再由上述 dual constraint 自适应控制。

### 6.2 参数 trust region

对第一阶段参数快照 \(\theta_0\) 建立显式锚点。由于 \(u,v\) 已是硬冻结，
proximal 只作用于实际允许训练的参数：

\[
\mathcal L_{\mathrm{prox}}
=\sum_i
\lambda_d\rho(d_i-d_i^0)
+\lambda_s\rho(\log s_i-\log s_i^0)
+\lambda_r d_R(R_i,R_i^0)
+\lambda_o\rho(o_i-o_i^0)
+\lambda_h\rho(h_i-h_i^0),
\]

其中实现使用 Smooth-L1，\(d_R=1-|\langle q_i,q_i^0\rangle|\) 消除 quaternion
的正负二义性。原重建点使用完整权重，一次性新增的口腔点使用 0.1 倍权重，
因为静态闭嘴参考对其约束较弱。全局把 \(d\) 拉到零是错误的，因为头发、服饰、
牙齿和非贴面结构本来就需要较大的法向偏移。

### 6.3 屏幕空间前景 mask

扩散梯度在 latent 分辨率上按当前 alpha 限制：

\[
g_{\mathrm{masked}}
=m_{\mathrm{latent}}\odot g
+\lambda_{\mathrm{bg}}(1-m_{\mathrm{latent}})\odot g,
\qquad 0\le\lambda_{\mathrm{bg}}\ll1.
\]

\(m_{\mathrm{latent}}\) 应由当前渲染 alpha 下采样并停止梯度。这样可避免纯背景的大面积 score 主导颜色和 opacity；边缘可保留软 alpha，防止轮廓处出现硬切割。mask 只限制梯度空间范围，不替代 alpha replay。

## 7. appearance → geometry 分阶段优化

强初始化下默认采用逐步释放自由度，而不是从第一步联合更新全部参数。

### 阶段 A：冻结初始化并估计分布

- 固定全部 Gaussian 参数；
- 动态 chemistry 渲染只用于训练在线 VSD LoRA；
- 静态多视角 replay 用于估计初始化基线 \(B_0\)；
- 不让尚未适配当前渲染域的扩散梯度改写 avatar。

### 阶段 B：外观修复

- 固定 \(u,v,d\)、scale、rotation 和拓扑；
- 仅允许 SH/颜色及极小幅 opacity 更新；
- 保持较强低通 reference replay 和外观锚点；
- 使用较低噪声区间和保守 CFG。

### 阶段 C：局部几何释放

- 保持 shape、neck 约束不变；
- 释放 \(d\)、scale、rotation；\(u,v\) 继续硬冻结；
- 几何学习率比颜色低一个到两个数量级；
- dual reference constraint、参数 proximal 与尺度 barrier 始终保留。

warmup 结束时才启用 Gaussian 优化；之后学习率采用 cosine decay，SDS→VSD
权重连续 ramp。geometry 解锁点固定为 600，且所有 residual 始终相对同一个
\(\theta_0\)，不在中途重置锚点。

## 8. chemistry-only 动态采样

### 8.1 数据字段

每个动态样本只读取：

- expression；
- jaw pose；
- left/right eye pose。

`neck_pose` 即使存在于文件中也必须忽略，不能作为默认值、增强项或控制条件。shape 始终使用第一阶段保存的 \(\beta^\star\)，不能从 chemistry 文件读取，也不能在 batch 间变化。

### 8.2 鲁棒 jaw 过滤

说话数据可能包含拟合失败产生的极端轴角。预处理应先去除 NaN/Inf，再以 jaw 轴角范数或可解释的嘴部开合量建立鲁棒统计。可采用分位数截断，或 median/MAD 规则：

\[
r_j=\lVert\theta_{\mathrm{jaw},j}\rVert_2,\qquad
\lvert r_j-\operatorname{median}(r)\rvert
\le k\cdot1.4826\operatorname{MAD}(r).
\]

同类拟合失败也会出现在 expression 与左右眼轴角中。因此实现还分别提供
`chemistry_expression_max_norm` 和 `chemistry_eye_max_norm`；默认仅拒绝极端尾部，
设为非正数即可关闭，用于受控消融。过滤只决定 `chemistry_exp.npy` 中哪些帧可被
采样，不会用裁剪后的参数或其他数据源替换它们。

过滤是排除跟踪异常，不应把所有大开口样本都删除。阈值必须仅由训练划分估计，并固定用于验证和复现实验。

### 8.3 开嘴过采样

均匀抽帧通常被闭嘴和小运动主导，因此应在通过鲁棒过滤的集合中，再按 jaw 开合量划分 neutral/small/large-mouth 子集，对 large-mouth 进行受控过采样。每个优化批次中的多视角必须使用同一组 expression/jaw/eye 参数。验证集需固定一组从闭嘴到大开口的 chemistry 序列，不能只报告随机帧。

## 9. ControlNet 条件必须与 RGB 共投影

FLAME 条件图应由当前 \((\beta^\star,\xi)\) 下的三维 landmarks 或 mesh 生成，处理顺序必须与 RGB 一致：

\[
V^T(\beta^\star,\xi)
\xrightarrow{A}
V^F
\xrightarrow{W_{\mathrm{cv}},K}
p.
\]

条件图、RGB 与 alpha 必须共享同一相机索引、分辨率、crop、水平翻转状态和 batch 顺序。若条件图在 FLAME 训练坐标投影、RGB 却在 FaceLift 坐标渲染，扩散模型会把这种错位解释为需要修正的结构，从而产生双眼、双唇或偏移下颌。

动态条件应显式反映 jaw、expression 和 eye 变化。静态 reference replay 则使用第一阶段保存的静态 FLAME 状态，不能用随机 chemistry 条件配静态 RGB。类似“单图或单目证据 + 可动画先验”的核心困难也出现在 [Zero-1-to-A](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Zero-1-to-A_Zero-Shot_One_Image_to_Animatable_Head_Avatars_Using_Video_CVPR_2025_paper.html) 与 [GAF](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_GAF_Gaussian_Avatar_Reconstruction_from_Monocular_Videos_via_Multi-view_Diffusion_CVPR_2025_paper.html) 中：外观证据与动态条件必须在共同的三维表示和投影下建立对应。
[GeoDiff4D](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_GeoDiff4D_Geometry-Aware_Diffusion_for_4D_Head_Avatar_Reconstruction_CVPR_2026_paper.html)
也直接指出仅靠二维扩散先验难以得到一致三维几何；因此本实现不把二维 score
当作几何真值，而只允许它在显式 FLAME 绑定和重建约束内做 residual repair。

## 10. 口腔区域处理

“嘴能张开”不是单一二维损失可以解决的问题，而是表示、采样和约束共同决定的：

1. **表示支持**：检查嘴唇、牙齿和口腔内壁是否有明确的 FLAME/UVD 区域及合理法向偏移。若初始化缺少口腔内壁，可在训练前一次性、区域限定地补充少量口腔高斯；补点后仍保持第二阶段默认固定拓扑。
2. **动态覆盖**：使用经过过滤的大开口 chemistry 样本过采样，让唇缘、牙齿和内壁在多视角下被真实观察。
3. **区域 trust region**：皮肤和外轮廓保持强锚点，唇缘适度释放；牙齿与口腔颜色、opacity 和尺度使用独立权重，避免被皮肤平均颜色污染。
4. **可见性与尺度**：限制口腔高斯的世界尺度和 opacity，防止闭嘴时穿透嘴唇、张嘴时形成漂浮暗片。
5. **评估**：按 jaw 开合量分桶，从正面、侧前方和侧面检查唇间距、牙齿连通性、内壁覆盖和闭嘴泄漏。

可用唇部 landmark 的目标开合与渲染轮廓建立弱约束，但不能强制“图像中间变黑”来代替三维口腔。显式绑定表面高斯并随参数模型驱动，是 [GaussianAvatars](https://arxiv.org/abs/2312.02069) 保持动画一致性的关键思路之一。

## 11. 拓扑策略

默认策略是**固定 topology**：

- 不执行全局 densification；
- 不按短期梯度或 opacity 全局 prune；
- 不改变原有 face/UVD 绑定；
- 仅更新当前阶段明确开放的连续参数。

原因是第二阶段的扩散梯度是视角相关且噪声较大的，增删点会同时改变容量、遮挡关系和绑定分布，使 reference replay 与参数锚点失去一一对应。若第一阶段确定缺少口腔内壁，可把“训练前区域种子”视为初始化修复；训练开始后仍冻结拓扑。

当前第二阶段代码不再保留 densify/prune/repair 的运行时分支。需要研究可变拓扑时，
应另建明确的实验 system；不能让一个默认关闭但仍贯穿 checkpoint、anchor 和 optimizer
的分支把主实现重新膨胀，也不能将“对齐错误或投影错误导致的高梯度”当作
densification 信号。

## 12. 模块级消融矩阵

所有实验使用同一初始化、固定 chemistry 训练/验证划分、固定相机列表、随机种子和训练预算。shape 固定、无 neck、chemistry-only、OpenCV 相机严格复用、FaceLift 均值/协方差正确变换等硬约束在所有实验中保持不变。

| 编号 | 被验证模块 | 对照设置 | 实验设置 | 主要判据 |
|---|---|---|---|---|
| A0 | 强初始化本身 | 第一阶段重建直接评估 | 不做扩散更新 | 后续所有增益的基线 |
| A1 | 扩散目标 | SDS | VSD | 身份保持、饱和度、多视角一致性、动态稳定性 |
| A2 | 扩散必要性 | 仅 reference + anchors | 加入默认扩散目标 | 伪影修复与文本一致性是否来自扩散 |
| A3 | reference constraint | 关闭 | 低通 RGB + alpha + adaptive dual | 静态身份漂移、轮廓漂移、伪影残留 |
| A4 | replay 频率设计 | 全分辨率逐像素 replay | 低通/多尺度 replay | 是否既保身份又允许高频修复 |
| A5 | 参数 trust region | 关闭 | 分参数、分区域 anchors | UVD 漂移、scale 爆炸、opacity/颜色漂移 |
| A6 | 分阶段优化 | appearance/geometry 联合开放 | appearance → geometry | 收敛稳定性、几何伪影、最终驱动一致性 |
| A7 | 前景 latent mask | 全图扩散梯度 | alpha 软 mask + 弱背景权重 | 背景梯度占比、过饱和、轮廓质量 |
| A8 | ControlNet 动态条件 | 仅文本 | 文本 + 同坐标 FLAME 条件 | 表情跟随、跨视角唇眼位置一致性 |
| A9 | negative prompt | 关闭 | 针对过饱和/牙齿/变形的负向提示 | 色彩统计、牙齿伪影；同时监控身份损失 |
| A10 | CFG 强度 | 从零阶段的高 CFG | 保守 CFG | 饱和度、身份漂移、修复幅度 |
| A11 | jaw 鲁棒过滤 | 仅去除非有限值 | 分位数或 MAD 过滤 | 极端帧爆炸率、正常大开口召回 |
| A12 | 开嘴过采样 | chemistry 内均匀采样 | chemistry 内分桶过采样 | 大开口成功率、闭嘴泄漏 |
| A13 | 多视角状态共享 | 每视角独立 chemistry 帧 | 同一姿态多视角 | 跨视角驱动一致性 |
| A14 | 口腔区域处理 | 无区域特化 | 口腔表示检查/种子 + 区域约束 | 内壁覆盖、漂浮牙齿、穿透与闭嘴泄漏 |
| A15 | 口腔 topology 支撑 | 不补点 | 训练前一次性区域补点、之后固定 | 缺失表示容量是否是张不开嘴的根因 |
| A16 | timestep 策略 | 宽范围且大噪声占比高 | 偏低噪声并逐步退火 | 局部修复、身份保持、梯度方差 |
| A17 | reference 区域权重 | 全区域同权 | 皮肤/轮廓强、嘴部/伪影弱 | 可靠区域保持与可修复区域自由度 |

建议最小论文表格先报告 A0、A1、A3、A5、A6、A7、A8、A12、A14、A15；其余作为机制分析。A11 的“无过滤”对照只用于离线、受控训练，不应作为生产默认设置。

## 13. 评估协议

单个正面截图不足以验证可动画头部。至少从以下四个维度报告：

- **静态保真**：第一阶段固定相机上的 PSNR/SSIM/感知距离、alpha IoU，以及身份特征相似度；
- **颜色健康度**：前景 HSV 饱和度分布、高光截断比例、跨视角同一区域的颜色方差；
- **动态一致性**：固定 chemistry 验证序列在多视角下的 landmark 重投影一致性、轮廓稳定性和时序闪烁；
- **口腔质量**：按 jaw 开合量分桶统计开口成功率、闭嘴泄漏、牙齿重复/漂浮、内壁空洞，并保留正面与侧面可视化。

还应记录每类参数相对 \(\theta_0\) 的漂移、最终高斯点数、不同语义区域的梯度范数。若视觉质量提升伴随大规模 UVD 漂移或点数膨胀，应视为表示被扩散先验重写，而非可靠修复。

## 14. 推荐默认决策

在没有消融结果前，第二阶段的保守默认应是：

1. 以 VSD 为主要扩散目标，SDS 保留为可切换对照；
2. 以低通静态 replay 定义约束，并用 dual update 根据实际漂移自适应调权；
3. 先冻结估计初始化分布，再外观、后局部几何，固定 \(u,v\) 与训练期 topology；
4. 动态状态仅来自过滤后的 chemistry expression/jaw/eyes，同一状态渲染多个第一阶段相机；
5. ControlNet 条件与 RGB 使用完全相同的 FLAME 状态、FaceLift 对齐和 OpenCV 投影；
6. 使用保守 CFG、偏低噪声 timestep 和针对实际失败模式的 negative prompt；
7. 对大开口样本过采样，并对口腔区域使用独立表示检查和约束。

最终配置字段、权重和训练命令应以实现代码的真实接口为准，并在完成 dry-run、坐标叠加检查和单批梯度检查后另行补充，避免文档与代码接口脱节。

## 15. 实现入口与运行

实现采用独立 registry，不会进入原始 `create_from_flame` 从零初始化路径：

- DataModule：`reconstruction-finetune-datamodule`
- System：`head-3dgs-reconstruction-finetune-system`
- 推荐配置：`configs/t2head_reconstruction_vsd.yaml`
- 公平 SDS 对照：`configs/t2head_reconstruction_sds.yaml`
- 逐模块消融命令：`configs/t2head_reconstruction_ablations.md`

训练命令：

```powershell
$env:THREESTUDIO_LAZY_IMPORT="1"
F:\Anaconda3\envs\headstudio\python.exe launch.py --config configs/t2head_reconstruction_vsd.yaml --train --gpu 0
```

懒加载开关会跳过本路径不使用的 `tinycudann` geometry/material 模块；它只改变
registry 的导入方式，不改变训练实现或配置。未设置时仍保留项目原有的 eager 行为。

当前实现显式限定单 GPU；Gaussian 与在线 VSD LoRA 使用各自的优化器状态，
尚未实现 DDP 参数与 optimizer-moment 同步，不能通过增加 `--gpu` 编号直接扩成多卡。

切换 identity 时，只需要同步修改配置中的
`data.reconstruction_dir` 与 `system.prompt`。动态训练的
`reference_pose_probability` 默认为零；因此每个随机动态 pose 都严格来自
`chemistry_exp.npy`，静态身份保持则由独立 reference replay 完成。

## 参考工作

- [DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988)
- [ProlificDreamer: High-Fidelity and Diverse Text-to-3D Generation with Variational Score Distillation](https://arxiv.org/abs/2305.16213)
- [HeadStudio: Text to Animatable Head Avatars with 3D Gaussian Splatting](https://arxiv.org/abs/2402.06149)
- [GaussianDreamer: Fast Generation from Text to 3D Gaussians by Bridging 2D and 3D Diffusion Models](https://arxiv.org/abs/2310.08529)
- [DreamGaussian: Generative Gaussian Splatting for Efficient 3D Content Creation](https://arxiv.org/abs/2309.16653)
- [Zero-1-to-A: Zero-Shot One Image to Animatable Head Avatars Using Video Diffusion](https://openaccess.thecvf.com/content/CVPR2025/html/Zhou_Zero-1-to-A_Zero-Shot_One_Image_to_Animatable_Head_Avatars_Using_Video_CVPR_2025_paper.html)
- [GAF: Gaussian Avatar Reconstruction from Monocular Videos via Multi-view Diffusion](https://openaccess.thecvf.com/content/CVPR2025/html/Tang_GAF_Gaussian_Avatar_Reconstruction_from_Monocular_Videos_via_Multi-view_Diffusion_CVPR_2025_paper.html)
- [GeoDiff4D: Geometry-Aware Diffusion for 4D Head Avatar Reconstruction](https://openaccess.thecvf.com/content/CVPR2026/html/Xu_GeoDiff4D_Geometry-Aware_Diffusion_for_4D_Head_Avatar_Reconstruction_CVPR_2026_paper.html)
- [GaussianAvatars: Photorealistic Head Avatars with Rigged 3D Gaussians](https://arxiv.org/abs/2312.02069)
