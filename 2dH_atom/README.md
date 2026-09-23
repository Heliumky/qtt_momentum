# Two-dimensional soft-Coulomb atom (analytic Fourier-QTT MPO)

这个目录用 Fourier-QTT 与 ITensor DMRG 求解带可控软化的二维氢原子

```math
H=-\frac12\left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)
-\frac{Z}{\sqrt{r^2+\epsilon}}.
```

## TCI 直接拟合指定的分段解析系数

`L_x`、`L_y` 是矩形的 half-width，`epsilon` 由使用者直接指定。对于
`(m,n) != (0,0)`，程序严格使用指定的 large-box 闭式：

```math
C_{mn}=-Z\,
\frac{\exp\!\left[-\pi\sqrt{\epsilon}
\sqrt{m^2/L_x^2+n^2/L_y^2}\right]}
{2\sqrt{m^2L_y^2+n^2L_x^2}}.
```

这里负号来自吸引势 `-Z/sqrt(r^2+epsilon)`。TCI 直接按需调用这个常数时间
目标函数，不经过数值积分、实空间网格或 FFT。

上式在 `(m,n)=(0,0)` 分母为零，零模单独使用有限矩形的解析式：

```math
C_{00}=-\frac{Z}{L_xL_y}\left[
L_x\operatorname{asinh}\!\frac{L_y}{\sqrt{L_x^2+\epsilon}}
+L_y\operatorname{asinh}\!\frac{L_x}{\sqrt{L_y^2+\epsilon}}
-\sqrt{\epsilon}\tan^{-1}\!\frac{L_xL_y}
{\sqrt{\epsilon}\sqrt{L_x^2+L_y^2+\epsilon}}
\right].
```

当 `epsilon→0`，零模连续回到裸 Coulomb 的有限矩形平均值。当前默认
`epsilon=1e-8` 只是可运行的默认配置；可以在 `HAtom2DConfig` 或命令行中直接
改成需要的值，程序不会根据 `q_max` 自动重写它。

## 构造

`coefficient_mpo_tci_nd` 在联合 `Nx+Ny`-site register 上用配对
`(output_bit,input_bit)` 直接拟合
`W[n_out,n_in]=coefficient(n_in-n_out)`。两个维度的动能项仍是精确的
Fourier-space 对角 MPO（`kinetic_mpo` + `embed_mpo`）。

## 收敛到精确二维氢原子

未正则化、无限空间的二维氢原子基态能量是 `E_0=-2`。本模型要靠近它，需要
联合做三重收敛：`epsilon→0`、Fourier cutoff 增大、`L_x,L_y→∞`。增加盒子时必须
同时增加 qubit 数来维持或提高动量 cutoff；固定 `N_list` 只增大 `half_width`
会降低 `q_max`。有限 cutoff 下不要求能量逐点单调。当前默认值是
`N_list=[12,12]`、`half_width_list=[10,10]`、`epsilon=1e-8`。

## 为什么验证方式和 `1dsho`/`2dsho` 不同

`1dsho`、`2dsho` 的基态有精确解析解可以比对。二维氢原子**在有限的周期盒子
里**没有这么简单的闭式基态（`E_0=-2` 只是 `half_width\to\infty` 的极限），
所以 `hatom2d_model.jl` 改用几个不依赖闭式解、但仍然诚实且有意义的检查：

- `energy_self_consistency`：DMRG 报告的能量应等于分别用 `expect_mpo` 算出
  的 `<Tx>+<Ty>+<V>`；
- `bound_state`：`E_0<0`；
- `virial_ratio_sane`：`-<V>/<T>` 落在一个很宽松的量级范围内。只有孤立、
  未正则化极限的 Coulomb 势才严格给出 `2`；softening、有限 cutoff、有限盒和
  large-box coefficient approximation 都会改变它；
- `density_normalization`：重建的 `|ψ(x,y)|^2` 积分应为 `1`——这一项特意用
  state 自己原生的 `2^N_list` 网格计算（而不是用来画图/存 CSV 的、通常更粗
  的 `realspace_points_list`），因为接近裸 Coulomb 极限时基态在原点有 cusp，简单
  Riemann-sum 积分只有在采样率匹配（或超过）这个带限周期函数自身的原生频率
  时才精确；
- `transpose_symmetry`：当 `Nx=Ny`、`Lx=Ly` 时，Hamiltonian 在 `x<->y` 下
  严格对称，正确的基态数值解也应该满足 `ψ(x,y)≈ψ(y,x)`——同样用原生网格
  计算，不依赖画图分辨率。

无论最终 PASS 或 FAIL，完整 diagnostics 都会先写入 `hatom2d_results.txt`，
其中同时打印孤立二维氢原子的参考值 `E_0=-2` 作为物理背景（不作为验证目标）。

## 运行

从专案根目录运行：

```bash
julia 2dH_atom/hatom2d_model.jl
```

可直接传入 `epsilon`，例如：

```bash
julia 2dH_atom/hatom2d_model.jl 1e-8
```

目前預設 `enable_realspace_reconstruction=false`：只執行 Hamiltonian 建構、
DMRG、能量期望值與張量 shape 輸出，不把 MPS 轉回實空間，也不生成新的
ground-state CSV/SVG。既有 CSV/SVG 會保留在磁碟上，但不會被本次計算更新。

之後若要恢復實空間重建與畫圖，可用
`HAtom2DConfig(enable_realspace_reconstruction=true)`；
`realspace_points_list` 只在這個開關開啟時生效。

```bash
julia 2dH_atom/hatom2d_model.jl 161 161 1e-8
```

两个整数控制绘图网格；最后一个可选参数是 `epsilon`。

输出结构与 `1dsho`/`2dsho` 对应：

- `hatom2d_results.txt`：TCI/DMRG 诊断、能量、`<Tx>`、`<Ty>`、`<V>`、
  energy self-consistency、virial ratio、（原生网格算出的）density
  normalization、transpose asymmetry、验证结果与张量 shapes；
- `hatom2d_mps_dimensions.csv`：每个完整 DMRG sweep 结束后的 MPS 最大 bond
  dimension 及各内部 bond dimensions；
- `hatom2d_tensor_shapes.txt`：逐-site 的 MPS shapes，以及分开列出的动能
  MPO（按维度）、势能 MPO（非可分离，整体一份）与合成后 Hamiltonian MPO
  shapes；
- `hatom2d_ground_state.csv`：`(x, y)` 网格外积上的数值波函数（相位已固定，
  使峰值处为实数正值）与密度 `|ψ|^2`（画图分辨率，见上）；
- `hatom2d_ground_state.svg`：基态波函数 `Re ψ(x,y)` 热图；
- `hatom2d_density.svg`：基态概率密度 `|ψ(x,y)|^2` 热图。

## 与 `Juliatools` 的关系

这个模型在本地定义 `softened_coulomb2d_fourier_coeff`，再交给通用 TCI 工具：

```julia
softened_coulomb2d_fourier_coeff(kx, ky, Lx, Ly, epsilon; charge=1)
coefficient_mpo_tci_nd(coefficient, qubits_per_dim; # multi_dim_potential.jl
                       tolerance, maxbonddim)
```

`multi_dim_potential.jl` 里还有 `grid_potential_fourier_coefficients`/
`potential_mpo_tci_nd`：对**没有**解析 Fourier coefficient 的任意势能
用一次 2D FFT 在原生网格上算出离散 Fourier coefficients 再交给 TCI，是当前
解析闭式方法之外的通用后备方案。
