# t-i curve via piecewise Kumaraswamy inverse-CDF
# ------------------------------------------------
# pip: matplotlib, numpy, pandas (可选)
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# ========== 参数区（可改） ==========
# 扩散离散上界（如 SD1.5 常见 0..800）
T = 800
# 阶段边界（真实 t）
t3, t2 = 420, 340  # => 归一化边界: [t3/T, 1], [t2/T, t3/T], [0, t2/T]
# 总步数 N=10,000，三段占比（几何/过渡/细节）
n1, n2, n3 = 3500, 1500, 5000
# Kumaraswamy 形状（人头域推荐）
a1, b1 = 0.7, 2.0  # Phase 1: 右偏长尾（快离开超大 t）
a2, b2 = 1.2, 1.2  # Phase 2: 近似对称，快进快出
a3, b3 = 2.5, 0.8  # Phase 3: 左偏厚尾（小 t 停留更久）
# 训练时可用的阶段下限（可用于实际 t 抖动采样）
LB1, LB2, LB3 = 450, 150, 20

# 选择区间化分位方式： "affine" 或 "trunc"
QUANTILE_MODE = "affine"
SAVE_TO = False  # 是否保存曲线数据


# ========== 核心函数 ==========

def inv_kuma(u, a, b):
    """Kumaraswamy(a,b) 的逆 CDF：F^{-1}(u)"""
    return (1.0 - (1.0 - u) ** (1.0 / b)) ** (1.0 / a)


def psi_affine(u, L, U, a, b):
    """方案A：仿射映射（默认推荐）"""
    return L + (U - L) * inv_kuma(u, a, b)


def psi_trunc(u, L, U, a, b):
    """方案B：严格截断（conditioning）"""
    # F(x) = 1 - (1 - x^a)^b
    FL = 1.0 - (1.0 - L ** a) ** b
    FU = 1.0 - (1.0 - U ** a) ** b
    y = FL + u * (FU - FL)  # 截断后的目标分位
    return (1.0 - (1.0 - y) ** (1.0 / b)) ** (1.0 / a)


def make_t_i(T, t2, t3, n1, n2, n3, a1, b1, a2, b2, a3, b3, mode="affine"):
    """生成整数 t-i 曲线（单调非增修正）"""
    # 归一化区间
    L1, U1 = t3 / T, 1.0
    L2, U2 = t2 / T, t3 / T
    L3, U3 = 0.0, t2 / T

    psi = psi_affine if mode == "affine" else psi_trunc

    ts = []

    # Phase 1 几何（大 t）（降序分位：1 - (i+0.5)/n）
    for i in range(n1):
        u = 1.0 - (i + 0.5) / max(1, n1)
        t_hat = psi(u, L1, U1, a1, b1)
        ts.append(T * t_hat)

    # Phase 2
    for i in range(n2):
        u = 1.0 - (i + 0.5) / max(1, n2)
        t_hat = psi(u, L2, U2, a2, b2)
        ts.append(T * t_hat)

    # Phase 3
    for i in range(n3):
        u = 1.0 - (i + 0.5) / max(1, n3)
        t_hat = psi(u, L3, U3, a3, b3)
        ts.append(T * t_hat)

    ts = np.array(ts, dtype=float)

    # 取整并保证单调非增（避免离散取整偶发上窜）
    ti = np.floor(ts).astype(int)
    for i in range(1, len(ti)):
        if ti[i] > ti[i - 1]:
            ti[i] = ti[i - 1]
    return ti


# ========== 生成 & 绘图 ==========
if __name__ == '__main__':
    ti = make_t_i(T, t2, t3, n1, n2, n3, a1, b1, a2, b2, a3, b3, mode=QUANTILE_MODE)

    plt.figure(figsize=(9, 4.5))
    plt.plot(np.arange(len(ti)), ti)
    plt.xlabel("Training step i (N = %d)" % (len(ti)))
    plt.ylabel("t (discrete diffusion step)")
    plt.title("t–i schedule (piecewise Kumaraswamy inverse-CDF, mode=%s)" % QUANTILE_MODE)
    plt.tight_layout()
    plt.show()

    # （可选）保存为 .npy / .csv，方便训练脚本载入
    if SAVE_TO:
        np.save("t_i_curve_10k.npy", ti)
        pd.DataFrame({"i": np.arange(len(ti)), "t": ti}).to_csv("t_i_curve_10k.csv", index=False)


# ========== 额外：实际训练时的 t 抖动采样示例 ==========
def sample_training_t(step, ti, n1, n2, n3, LB1, LB2, LB3):
    """给定 step，按阶段在 [LB_k, ti[step]] 内均匀采样一个 t"""
    N = len(ti)
    step = min(step, N - 1)
    cur_t = int(ti[step])

    if step < n1:
        left = max(LB1, t3)  # 阶段1建议不低于 t3
        right = max(cur_t, left + 1)
    elif step < n1 + n2:
        left = max(LB2, t2)
        right = max(cur_t, left + 1)
    else:
        left = max(LB3, 0)
        right = max(cur_t, left + 1)

    return np.random.randint(left, right)

# 示例：第 1234 步的一个实际 t
# print(sample_training_t(1234, ti, n1, n2, n3, LB1, LB2, LB3))
