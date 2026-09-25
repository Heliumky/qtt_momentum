# Fourier-QTT with DFT potentials (`QTTDFT`)

这是 `../`（解析版 `QTTFourier`）的 DFT 改写版。Hamiltonian、基态与所有 DMRG 流程仍在
**同一个 Fourier basis、同一个 Fourier register** 中进行；差别只在势能 MPO 的构造：

| | 解析版 `QTTFourier` | DFT 版 `QTTDFT`（本目录） |
| --- | --- | --- |
| 势能矩阵元 | `V[n',n] = (1/2L)∫V(x)e^{iπ(n-n')x/L}dx`（解析 Fourier coefficient） | `V[n',n] = (1/M)Σ_j V(x_j)e^{iπ(n-n')x_j/L}`（网格 quadrature） |
| TCI 拟合对象 | 4^N paired-index 的 Toeplitz 算符 | 实空间 2^N 网格上的函数 `V(x_j)` |
| 需要 | 每种势能推一个解析系数公式 | 只要能在网格点上求值 `V(x)` |
| 算符 | TCI 直接给出 MPO | `V_k = U · diag(V) · U†`，`U` 为 quantics DFT MPO |

## 目录

```text
QTT_DFT/
├── Project.toml / Manifest.toml   # 在原依赖上加入 QuanticsTCI（DFT MPO）
├── Juliatools/
│   ├── QTTDFT.jl            # 模组入口与公开 API
│   ├── elemetary_func.jl    # index 约定、实空间网格、TCI（pivot 优化）、实空间函数 MPS
│   ├── diff_MPO.jl          # Fourier basis 的精确微分 / 动能 MPO（与原版相同）
│   ├── dft_mpo.jl           # DFT MPO、U V U†、1D DFT 势能
│   ├── dmrg_engine.jl       # ITensor 转换、Hamiltonian、DMRG、shape 工具
│   ├── k_to_r.jl            # Fourier MPS -> 实空间（原生网格用 U† MPO）
│   ├── plot.jl              # SVG 绘图（与原版相同）
│   ├── multi_dim.jl         # RegisterLayout（block / interleaved）、embed、联合 DFT MPO、可分离 Hamiltonian
│   ├── k_to_r_nd.jl         # 多维 Fourier MPS -> 实空间
│   └── multi_dim_potential.jl  # 不可分离多维势能（实空间联合 TCI + 联合 DFT MPO）
├── 1dsho/     # 一维简谐振子
├── 2dsho/     # 二维可分离简谐振子
├── 2dH_atom/  # 二维氢原子（裸 Coulomb，半移网格 + cell average，与解析方法比较）
└── test/      # 全部测试皆依赖 Juliatools/QTTDFT.jl
```

## 两个 register

张量 shape 约定与原版完全相同：

```text
MPS A_i.shape = (left_bond, physical_dim, right_bond)
MPO W_i.shape = (left_bond, output_dim, input_dim, right_bond)
```

- **Fourier register**（所有 Hamiltonian 与基态）：site 1 是 mode 的 LSB，最后一个 site 是
  two's-complement sign bit，`N=3` 的顺序为 `0,1,2,3,-4,-3,-2,-1`。与原版一致。
- **实空间 register**（只用来拟合 `V(x_j)`）：site 1 是网格 index `j` 的 **MSB**，

```math
x_j=-L+(j+\text{offset})\frac{2L}{M},\qquad j=0,\dots,M-1,\quad M=2^N .
```

  `offset=0` 时网格包含 `x=0`；`offset=0.5` 为 cell-centred 网格，永远不取到原点，
  适合 `1/r` 这类原点奇异的势能。

`QuanticsTCI.quanticsfouriermpo` 的输出腿恰为 LSB-first、输入腿为 MSB-first，正好把实空间
register 映到 Fourier register，**不需要任何 bit reversal**。

## DFT MPO

正规化 basis `φ_n(x)=e^{iπnx/L}/√(2L)` 下，从网格取样到 Fourier coefficient 的 unitary 是

```math
U_{nj}=\frac{e^{-i\pi n x_j/L}}{\sqrt M}
=\Phi(n)\,\frac{e^{-2\pi i kj/M}}{\sqrt M},\qquad k=n \bmod M,
\qquad
\Phi(n)=\prod_\ell e^{i\pi w_\ell b_\ell(1-2\,\text{offset}/M)} .
```

中间是标准 DFT（Chen–Lindsey 插值构造，`N=10` 时 bond dim 13、误差约 1e-12）；
`Φ` 是 two's-complement bit weights `w_ℓ` 的单 site 相位乘积（bond dim 1），直接吸收进
输出腿。`fourier_transform_mpo(N, L; offset)` 返回这个 array MPO。

势能算符：

```text
V(x_j) --TCI--> 实空间 MPS --> 对角 MPO D
V_k = U · D · U†        （ITensor apply，cutoff = 1e-26）
```

`V_k[n',n]` 就是 `V` 在网格上的 DFT 系数，是解析积分的 quadrature 版本。对平滑的周期性
势能这是 Fourier grid（pseudo-spectral）方法，谱精度收敛；对 `x^m` 这类多项式，实空间 MPS
的 rank 恰为 `m+1`，TCI 没有误差。

**关于 `cutoff`**：ITensor 的 cutoff 是相对的 *平方* 奇异值门槛，而 `‖V_k‖_F ~ √M·max|V|`。
`cutoff=1e-14` 只能给出约 1e-7 的相对算符误差（1D SHO 的 E0 误差 3e-8）；默认 `1e-26`
对应约 1e-13（E0 误差 1e-13），bond dim 只从 12 增至 14。

## TCI 与 pivot 优化

`fit_tensor_train` 包装 `TCI.crossinterpolate2`：

- **多个初始 pivot**：`initialpivots` 中每个 pivot 先经 `TCI.optfirstpivot` 局部优化，去掉零值
  与重复后全部交给 TCI。实空间版本以 `pivot_points`（物理坐标）指定，例如 SHO 用 `[0.0]`、
  Coulomb 用 `[(0.0, 0.0)]`；网格角落 `x=-L` 永远包含在内。离默认 pivot 很远的窄峰若不给
  pivot 会被 TCI 完全漏掉（`test/test_elementary.jl` 有对照测试）。
- **global pivot search**：开放 `nsearchglobalpivot`、`maxnglobalpivot`、`tolmarginglobalsearch`。
- **误差定义**：`normalizeerror=true`（相对最大取样值）或 `false`（绝对误差）。
- **求值**：默认 `TCI.CachedFunction`（同一点不重复求值）；`threaded=true` 改用
  `TCI.ThreadedBatchEvaluator`，适合昂贵且 thread-safe 的目标函数。
- **诊断**：除 TCI 自身的 pivot error 外，另以 `TCI.estimatetrueerror` 做随机 greedy 搜索，
  回传 `global_error_estimate`（写入各 `*_results.txt`）。

## 多维：`RegisterLayout`

`RegisterLayout(qubits_per_dim; scheme=:block | :interleaved)` 决定各维度 bit 在联合 register
中的位置；`embed_mpo`、`fourier_transform_mpo_nd`、`mps_to_realspace_nd` 都对两种 layout 通用。

- `:block`（默认，与原版相同）：维度间的张量积不增加 bond dim，联合 DFT MPO 的 bond dim
  与 1D 相同（~12）。
- `:interleaved`：同尺度的 bit 相邻，但联合 DFT MPO 的 bond dim 变成各维乘积（两个 8-qubit
  轴约 144），`U V U†` 成本高得多。对 2D Coulomb（`[8,8]`）block layout 的 TCI rank 反而
  更低（37 vs 52），因此不建议对 DFT 路径使用 interleaved。

## 结果摘要

| 例子 | 量 | 解析版 | DFT 版 |
| --- | --- | --- | --- |
| 1D SHO（N=10, L=8） | TCI evaluations | 12950 | 266 |
| | 势能 / H MPO max bond | 13 / 16 | 14 / 17 |
| | E0 误差 | −5.6e-17 | +1.0e-13 |
| 2D SHO（[8,8], L=8） | TCI evaluations（每维） | ~7000 | ~170 |
| | 势能 / H MPO max bond | 13 / 18 | 13 / 18 |
| | E0 误差 | −9.7e-15 | +3.4e-12 |
| 2D H（N=10 每维, L=16） | E0（精确孤立值 −2） | −1.99867 | −1.98865（cell average）/ −1.88464（点取样） |
| | TCI evaluations / 建构时间 | 1 030 168 / 199 s | 63 527 / 23 s |
| | 势能 MPO max bond | 71 | 134 |

对平滑势能（SHO）DFT 版与解析版精度相当而 TCI 成本小两个数量级；对奇异的 Coulomb 势，
解析有限盒系数更准（约 `h²` 收敛），DFT 版用半移网格 + cell average 约以 `h^1.6` 收敛，
但 TCI 求值少约 16 倍。完整比较见 `2dH_atom/README.md`。

## 奇异势能与 DMRG 初始态

- **半移网格**：`offset=0.5` 时原点不在网格上，`-Z/r` 可以不软化直接使用；只改变 DFT MPO 的
  bond-1 相位。
- **cell average**：`coulomb2d_cell_average(x, y, hx, hy)` 以闭式给出 `-Z/r` 在每个 cell 上的精确
  平均值，比点取样准确得多（N=8 时 E0 −1.891 vs −1.627）。
- **初始态**：`realspace_state_mps_nd(f, sites, layout, Ls)` 把实空间猜测（如 `exp(-2r)`）经 TCI 与
  DFT MPO 转成 Fourier-register MPS，传给 `ground_state(...; initial_state)`。随机初始态在
  `[10,10]` 的 Coulomb 问题会卡在 `E=-0.673` 的局部极小。

## 执行

在 conda 环境 `julia_qqt` 中（其默认 Julia project 已安装全部依赖，也可用 `--project=.`）：

```bash
conda activate julia_qqt
cd QTT_DFT
julia --project=. 1dsho/sho_model.jl
julia --project=. 2dsho/sho2d_model.jl
julia --project=. 2dH_atom/hatom2d_model.jl
cd test && julia --project=.. runtests.jl
```

测试涵盖：index/网格约定、多项式与窄峰 TCI、cached vs threaded evaluator、
DFT MPO 对稠密矩阵（多个 `N` 与 offset）与 unitarity、`U V U†` 对稠密结果、band-limited
势能的精确系数、实空间重建（含 offset 与 normalization）、DMRG、两种 layout 的
embed / 联合 DFT MPO / 可分离 Hamiltonian、不可分离势能、cell-averaged Coulomb 的闭式，
以及小型 2D Coulomb 束缚态（含实空间初始态）。

## 主要 API

```julia
# grids and TCI
realspace_grid(M, L; offset=0)
fit_tensor_train(f, localdims; tolerance, maxbonddim, initialpivots,
                 nsearchglobalpivot, maxnglobalpivot, tolmarginglobalsearch,
                 normalizeerror, threaded, ntrueerrorsearch, return_diagnostics)
realspace_function_mps(f, N, L; offset=0, pivot_points=[], kwargs...)
power_potential_mps(m, N, L; scale=1, offset=0, kwargs...)

# DFT
fourier_transform_mpo(N, L; offset=0)
fourier_space_operator(sites, realspace_arrays, transform_arrays; cutoff=1e-26)
potential_mpo_dft(sites, V, N, L; offset, pivot_points, tci_tolerance, cutoff, ...)
build_dft_hamiltonian(sites, V, N, L; kwargs...)
build_power_hamiltonian(sites, m, N, L; scale=1, kwargs...)

# differential operators, DMRG, shapes (same as QTTFourier)
kinetic_mpo(N, L); differential_mpo(order, N, L)
ground_state(H, sites; nsweeps, maxdim, cutoff)
mps_shapes, mpo_shapes, mps_bond_dims, mpo_bond_dims, max_bond_dim

# real space
fourier_to_realspace_mps(state, sites, L; offset=0)
mps_to_realspace(state, sites, L; npoints=nothing, x=nothing, offset=0)

# multi-dimensional
RegisterLayout(qubits_per_dim; scheme=:block)
embed_mpo(component, dim, layout); kron_mpo(operators)
fourier_transform_mpo_nd(layout, Ls; offsets)
build_separable_hamiltonian(sites, qubits_per_dim, Ls; potentials, powers, scales, offsets, scheme)
realspace_function_mps_nd(f, layout, Ls; offsets, pivot_points)
potential_mpo_dft_nd(sites, V, layout, Ls; offsets, pivot_points, ...)
build_dft_hamiltonian_nd(sites, V, layout, Ls; kwargs...)
mps_to_realspace_nd(state, sites, layout, Ls; points_per_dim, offsets)
realspace_state_mps_nd(f, sites, layout, Ls; offsets, pivot_points)   # DMRG initial state
coulomb2d_cell_average(x, y, hx, hy; charge=1)
ground_state(H, sites; initial_state=nothing, noise=0.0, ...)
```
