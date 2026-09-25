# Two-dimensional hydrogen atom (DFT potential, half-shifted grid)

这个目录用 Fourier-QTT、DFT MPO 与 ITensor DMRG 求解二维氢原子

```math
H=-\frac12\left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)
-\frac{Z}{r},\qquad E_0^{\text{isolated}}=-2\quad(Z=1).
```

## 半移（cell-centred）网格

势能在实空间网格上取样后以 `V_k = U · diag(V) · U†` 转到 Fourier basis。
未偏移的网格 `x_j=-L+j·2L/M` 含有原点，`-Z/r` 在那里是无穷大，只能靠软化
`-Z/sqrt(r²+ε)` 才能使用。本目录改用**半移网格**（`grid_offset_list=[0.5, 0.5]`）：

```math
x_j=-L+\left(j+\tfrac12\right)\frac{2L}{M},
```

原点落在四个 cell 的共同角上，永远不会被取样，因此可以直接使用裸 Coulomb 势（`epsilon=0`）。
半移只改变 DFT MPO 的一个 bond-1 相位 `Φ(n)`（见 `../Juliatools/dft_mpo.jl`），不增加任何成本。

## 两种取样方式

- `potential_sampling = :point`：取 cell 中心的值 `-Z/r_j`。
- `potential_sampling = :cell_average`（默认）：取每个 cell 上 `-Z/r` 的**精确平均值**。
  `coulomb2d_cell_average` 使用矩形积分的闭式

```math
\iint\frac{dx\,dy}{r}=G(x_2,y_2)-G(x_1,y_2)-G(x_2,y_1)+G(x_1,y_1),\qquad
G(x,y)=x\,\operatorname{asinh}\frac yx+y\,\operatorname{asinh}\frac xy .
```

  原点旁边的 cell 平均值有限（`-2 asinh(1)/h`），但保留了奇点附近的积分权重；点取样则完全漏掉这部分。

TCI 以原点为 pivot（`pivot_points=[(0.0, 0.0)]`）拟合 `V`。

## DMRG 初始态

`initial_guess = :exponential`（默认）在实空间用 TCI 拟合 `exp(-2Z r)`，以 DFT MPO 转到
Fourier register 后作为 DMRG 起点（`realspace_state_mps_nd`）。随机初始态在 `N_list=[10,10]`
会卡在 `E=-0.673`、bond dim 5 的局部极小（跑 60 个 sweep 也不动），而势能 MPO 本身是正确的：
随机抽取的矩阵元与网格上的精确 DFT 和相差 `1.7e-11`。`initial_guess = :random` 仍可使用。

## 与原本的解析方法比较

`L=16`、`TCI tolerance=1e-10`、`DMRG cutoff=1e-8`、20 sweeps，三种方法都用同一个
`exp(-2r)` 初始态。解析方法是原套件的 `coulomb2d_fourier_coeff`（有限矩形上的精确 Fourier
积分）加 `coefficient_mpo_tci_nd`（在 4^N 的 paired index 上直接 TCI 拟合 MPO）。

| N（每维） | h | DFT 点取样 E0 | DFT cell average E0 | 解析 E0 |
| --- | --- | --- | --- | --- |
| 8  | 0.125   | −1.62740 | −1.89132 | −1.98082 |
| 9  | 0.0625  | −1.78736 | −1.96341 | −1.99498 |
| 10 | 0.03125 | −1.88464 | −1.98865 | −1.99867 |

与 `−2` 的误差：

| N | DFT 点取样 | DFT cell average | 解析 |
| --- | --- | --- | --- |
| 8  | 3.7e-1 | 1.1e-1 | 1.9e-2 |
| 9  | 2.1e-1 | 3.7e-2 | 5.0e-3 |
| 10 | 1.2e-1 | 1.1e-2 | 1.3e-3 |
| 每次 h 减半的误差比 | ~1.8 | ~3.1 | ~3.8 |

成本：

| N | 方法 | TCI evaluations | 建构时间 | 势能 MPO bond | H MPO bond | 基态 bond |
| --- | --- | --- | --- | --- | --- | --- |
| 8  | DFT cell average | 17 927 | 3.7 s | 72 | 76 | 11 |
| 8  | 解析 | 329 506 | 26.9 s | 57 | 61 | 12 |
| 9  | DFT cell average | 33 889 | 8.5 s | 96 | 100 | 12 |
| 9  | 解析 | 635 930 | 71.6 s | 65 | 69 | 12 |
| 10 | DFT cell average | 63 527 | 22.8 s | 134 | 138 | 12 |
| 10 | 解析 | 1 030 168 | 199.0 s | 71 | 75 | 12 |

（建构时间为 Julia 编译之后；N=8 的第一次执行另含约 50 s 编译。）

结论：

- **精度**：对 Coulomb 这种原点奇异的势能，原本的解析有限盒 Fourier 系数在同样的 N 下明显更准，
  而且约以 `h²` 收敛。DFT 点取样只以约 `h^0.85` 收敛；换成 cell average 后提升到约 `h^1.6`，
  N=10 时误差 1.1e-2，但仍比解析方法大约 8 倍。
- **成本**：DFT 版的 TCI 求值次数少约 16–18 倍、建构快约 7–9 倍，因为它只在 2^(2N) 的实空间网格上
  拟合 rank ~40 的函数，而不是在 4^(2N) 的 paired index 上拟合算符；代价是 `U V U†` 之后的
  势能 MPO bond dim 较大（N=10 时 134 vs 71），DMRG 每个 sweep 较慢。
- **适用性**：DFT 版对任意可取样的 `V(x,y)` 都能用，不需要推导解析 Fourier 系数。对平滑势能
  （如 SHO）它是谱精度的；对奇异势能，cell average 或更细的网格是必要的。

## 执行

从 `QTT_DFT/` 目录：

```bash
julia --project=. 2dH_atom/hatom2d_model.jl                # cell average, exp(-2r) initial guess
julia --project=. 2dH_atom/hatom2d_model.jl 1e-4           # soft Coulomb epsilon=1e-4, point sampling
julia --project=. 2dH_atom/hatom2d_model.jl 161 161        # plot points
```

默认 `N_list=[8,8]`、`half_width_list=[16,16]`。默认执行的结果（`hatom2d_results.txt`）：
E0 = −1.891316，`<T>+<V>` 自洽误差 9e-14，virial ratio 2.09，native-grid density normalization
1.0，x↔y 转置不对称 1.2e-4，validation PASS。

## 输出

- `hatom2d_results.txt`：取样方式、网格间距、实空间 V 的 rank、DFT MPO 与势能 MPO bond dim、TCI 误差与 global error estimate、DMRG 结果、能量分量、virial ratio、实空间诊断；
- `hatom2d_mps_dimensions.csv`：每个 sweep 结束后的 MPS bond dimensions；
- `hatom2d_tensor_shapes.txt`：逐-site 的 MPS / MPO shapes；
- `hatom2d_ground_state.csv`、`hatom2d_ground_state.svg`（`Re ψ`）、`hatom2d_density.svg`（`|ψ|²`）。
