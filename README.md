# Fourier-QTT

这个专案用 Fourier basis、QTT/MPS、MPO 与 ITensor DMRG 求解一维连续量子力学问题。当前势能构造只保留一条路径：**解析 Fourier coefficient 函数直接由 Tensor Cross Interpolation（TCI）拟合成 MPO**。程式不会建立 `2^N × 2^N` 的稠密 Hamiltonian。

目前的端到端范例是一维简谐振子

```math
H=-\frac12\frac{d^2}{dx^2}+\frac{x^2}{2}.
```

## 目录

```text
Fourier-QTT/
├── Juliatools/
│   ├── QTTFourier.jl       # 模组入口与公开 API
│   ├── elemetary_func.jl   # 解析 x^m/2D Coulomb 系数、TCI MPS/MPO
│   ├── diff_MPO.jl         # Fourier basis 中的任意阶微分 MPO
│   ├── dmrg_engine.jl      # ITensor 转换、Hamiltonian、DMRG、shape 工具
│   ├── k_to_r.jl           # Fourier coefficient MPS -> 实空间函数
│   ├── plot.jl             # SVG 绘图（1D 线图与 2D 热图）
│   ├── multi_dim.jl        # 可分离多维 Hamiltonian（identity/embed MPO）
│   ├── k_to_r_nd.jl        # 多维 Fourier coefficient MPS -> 实空间函数
│   └── multi_dim_potential.jl  # 不可分离多维势能的 TCI 拟合（解析闭式或 grid-FFT + ND TCI）
├── 1dsho/
│   ├── sho_model.jl        # 一维简谐振子验证
│   └── README.md
├── 2dsho/
│   ├── sho2d_model.jl      # 二维（可分离）简谐振子验证
│   └── README.md
├── 2dH_atom/
│   ├── hatom2d_model.jl    # 二维氢原子验证，不可分离势能，解析闭式 MPO
│   └── README.md
└── test/                   # 全部测试皆依赖 Juliatools/QTTFourier.jl
```

## 统一的张量定义

本专案统一使用 `bond` 这个名称。一个内部 bond 是相邻两个张量共享的维度。

### MPS

```math
C_{s_1\ldots s_N}
=\sum_{a_1\ldots a_{N-1}}
A^{[1]}_{1,s_1,a_1}A^{[2]}_{a_1,s_2,a_2}\cdots
A^{[N]}_{a_{N-1},s_N,1}.
```

每个数组张量固定为：

```text
A_i.shape = (left_bond, physical_dim, right_bond)
```

qubit 的 `physical_dim = 2`，左右边界 bond 为 1。例如：

```text
A_1  = (1, 2, 2)
A_2  = (2, 2, 4)
A_3  = (4, 2, 3)
...
A_10 = (2, 2, 1)
```

第 `i` 个内部 bond dim 同时等于 `size(A_i,3)` 与 `size(A_{i+1},1)`。

### MPO

```math
O_{s'_1\ldots s'_N,s_1\ldots s_N}
=\sum_{a_1\ldots a_{N-1}}
W^{[1]}_{1,s'_1,s_1,a_1}\cdots
W^{[N]}_{a_{N-1},s'_N,s_N,1}.
```

每个数组张量固定为：

```text
W_i.shape = (left_bond, output_dim, input_dim, right_bond)
```

qubit MPO 的 `output_dim = input_dim = 2`。例如：

```text
W_1 = (1, 2, 2, 6)
W_2 = (6, 2, 2, 10)
...
W_N = (6, 2, 2, 1)
```

查询 API：

```julia
mps_shapes(state, sites)
mpo_shapes(operator, sites)
mps_bond_dims(state)
mpo_bond_dims(operator)
max_bond_dim(state_or_operator)
```

对应的 array MPS/MPO 也支援相同查询。

## Fourier-QTT index

在周期区间 `[-L,L)` 使用正规化 basis

```math
\phi_n(x)=\frac{e^{i\pi nx/L}}{\sqrt{2L}},\qquad
f(x)=\sum_n c_n\phi_n(x).
```

`N` 个 qubit 表示 `2^N` 个 Fourier modes。site 1 是 least-significant bit（LSB），最后一个 site 是 sign bit；整数采用 two's-complement：

```math
n(s)=\sum_{j=1}^{N-1}2^{j-1}(s_j-1)-2^{N-1}(s_N-1),
\qquad s_j\in\{1,2\}.
```

例如 `N=3` 的储存顺序为 `0,1,2,3,-4,-3,-2,-1`。一整串 physical indices 才对应一个 Fourier coefficient。

## `x^m` 势能如何变成 MPO

### 1. 解析 coefficient 函数

对 `f(x)=a x^m`，乘法算符所需的 Fourier coefficient 是

```math
t_m(k)=\frac{a}{2L}\int_{-L}^{L}x^m e^{i\pi kx/L}\,dx
      =aL^mJ_m(k).
```

`k=0` 时：

```math
J_m(0)=\begin{cases}
0,&m\text{ 为奇数},\\
1/(m+1),&m\text{ 为偶数}.
\end{cases}
```

对非零整数 `k`，令 `q=iπk`，有限递推为

```math
J_0(k)=0,\qquad
J_m(k)=\frac{(-1)^k[1-(-1)^m]}{2q}-\frac{m}{q}J_{m-1}(k).
```

`power_fourier_coeff(m,k,L; scale=a)` 直接计算这个解析递推。单次查询成本随 `m` 而非随 `2^N` 增长。

### 2. TCI 直接拟合 MPO

Fourier basis 中的乘法算符满足

```math
V_{n_{out},n_{in}}=t_m(n_{in}-n_{out}).
```

`power_mpo_tci` 在每个 site 把 `(output_bit,input_bit)` 配成一个 local dimension 4 的 TCI index：

```text
paired indices
    -> output/input bit strings
    -> signed n_out, n_in
    -> analytic t_m(n_in - n_out)
    -> TCI tensor train
    -> reshape each local tensor to (left_bond, 2, 2, right_bond)
```

TCI 只自适应查询它需要的矩阵元素，不先产生完整 coefficient vector 或稠密矩阵。`ordering=:lsb_first` 与本专案的 qubit 顺序一致；也可显式选择 `:msb_first`。

若只需要 coefficient MPS，`power_mps` 会以同一个解析 target 拟合 `t_m(n(s_1,\ldots,s_N))`，返回 shape 为 `(left_bond,2,right_bond)` 的张量。不过 Hamiltonian 的主路径会直接调用 `power_mpo_tci`。

## 微分 MPO 与 Hamiltonian

Fourier mode 上

```math
\frac{d^m}{dx^m}\phi_n(x)=\left(\frac{i\pi n}{L}\right)^m\phi_n(x).
```

`differential_mpo(m,N,L)` 用 binary weighted-sum automaton 精确建立这个对角 MPO，其理论 bond dim 为 `m+1`。简谐振子动能 `-\tfrac12 d^2/dx^2` 因此有 bond dim 3。

Hamiltonian 流程：

```text
power_mpo_tci(m, ...)       -> potential MPO
kinetic_mpo(N,L)            -> kinetic MPO
ITensor exact direct sum    -> Hamiltonian MPO
ground_state(...)           -> DMRG ground-state MPS
```

直接和的 Hamiltonian bond dim 上限是两个分量 bond dim 的和；实际每个 bond 的大小由两个分量在该位置的 bond dim 决定。

## 误差记录

TCI 回传的是在其自适应 pivot 上估计的 normalized interpolation error；DMRG observer 回传每个 sweep 中最大的 SVD truncation error。SHO 报告会同时记录：

```text
TCI requested tolerance
TCI estimated relative error
TCI target evaluations
DMRG requested cutoff
DMRG max truncation error
DMRG truncation error/sweep
```

这些算法误差与最后另外计算的 energy error、wavefunction L2 error 是不同的量。

## 从 Fourier MPS 回到实空间

DMRG 的 MPS 表示 Fourier coefficients `c_n`。`mps_to_realspace` 计算

```math
\psi(x)=\sum_n c_n\phi_n(x).
```

画图点数不受 `2^N` 限制：

```julia
mps_to_realspace(state, sites, L; npoints=1601)
mps_to_realspace(state, sites, L; x=my_points)
```

- 未指定 `x` 或 `npoints`：在原生 `2^N` 均匀网格上使用 inverse FFT。
- 指定均匀 `npoints`：产生该数量的实空间点。
- 指定任意 `x`：直接计算 Fourier basis sum。

## 一维 SHO 验证

从专案根目录执行：

```bash
julia 1dsho/sho_model.jl
```

可用第一个参数独立指定画图点数：

```bash
julia 1dsho/sho_model.jl 2500
```

目前 `N=10`、1024 Fourier modes、`TCI tolerance=10^-12`、
`DMRG cutoff=10^-6` 的验证结果约为：

```text
potential MPO max bond dim    = 13
kinetic MPO max bond dim      = 3
Hamiltonian MPO max bond dim  = 16
ground-state MPS max bond dim = 4
DMRG E0                       = 0.50000414177314
```

输出：

- `1dsho/sho_results.txt`：算法误差、能量、物理误差、最大 bond dim 与张量 shapes。
- `1dsho/sho_tensor_shapes.txt`：MPS、势能 MPO、动能 MPO、Hamiltonian MPO 的逐-site shapes。
- `1dsho/sho_ground_state.csv`：可自由指定点数的实空间数据。
- `1dsho/sho_ground_state.svg`、`1dsho/sho_error.svg`：波函数与误差图。

## 二维 SHO 验证（可分离多维 Hamiltonian）

`2dsho/sho2d_model.jl` 把同一套方法扩展到可分离的多维势能：每个维度独立用
`power_mpo_tci`/`kinetic_mpo` 拟合，`embed_mpo` 把它精确地张成
`component ⊗ I`，再以 exact direct sum 合成联合 Hamiltonian。`N_list=[Nx,Ny]`
按维度分开设置 qubit 数（与 `3dH_p0_l/onepar.py` 的 `N_list` 风格一致）：

```bash
julia 2dsho/sho2d_model.jl
```

默认 `N_list=[8,8]`、`half_width_list=[8.0,8.0]`，输出结构与 1dsho 对应：
`sho2d_results.txt`、`sho2d_mps_dimensions.csv`、`sho2d_tensor_shapes.txt`、
`sho2d_ground_state.csv`，以及用 `plot_heatmap` 绘制的
`sho2d_ground_state.svg`/`sho2d_error.svg` 热图。详见 `2dsho/README.md`。

## 二维氢原子验证（不可分离势能，解析闭式 Fourier coefficient）

`2dH_atom/hatom2d_model.jl` 求解 `H=-1/2(d²/dx²+d²/dy²)-Z/r`。`-Z/r` 型
Coulomb 势不可分离，因此不能用 `build_separable_hamiltonian`。
`coulomb2d_fourier_coeff` 直接計算有限矩形內 `-Z/r` 的精確 Fourier
coefficient：先解析移除原點奇異並做掉徑向積分，再以 `QuadGK` 計算剩下的
平滑一維積分；零 mode 則直接使用閉式。`coefficient_mpo_tci_nd`
（`coefficient_mpo_tci` 的多维推广）直接对这个解析函数在整个联合 register
上 TCI 拟合势能 MPO——不需要實空間網格、FFT 或 softening。當 `half_width`
增大（配合增加 qubit 数），基态能量趋向精确二维氢原子的 `E_0=-2`。因为有限
盒子里没有闭式基态解，验证改用 energy self-consistency、bound state、
（原生网格上精确算出的）density normalization、`x<->y` transpose 对称性等
不依赖闭式解的检查：

```bash
julia 2dH_atom/hatom2d_model.jl
```

默认 `N_list=[8,8]`、`half_width_list=[16,16]`，输出结构与 1dsho/2dsho
对应：`hatom2d_results.txt`、`hatom2d_mps_dimensions.csv`、
`hatom2d_tensor_shapes.txt`、`hatom2d_ground_state.csv`，以及
`hatom2d_ground_state.svg`（`Re ψ`）/`hatom2d_density.svg`（`|ψ|²`）热图。
详见 `2dH_atom/README.md`（含能量随 `half_width` 收敛到 `-2` 的数值表）。

## 测试

```bash
cd test
julia --startup-file=no runtests.jl
```

测试涵盖解析 `x^m` coefficients、TCI MPS/MPO、张量 shape 约定、任意阶微分 MPO、Hamiltonian/DMRG，以及 Fourier-to-real-space 重建。

## 主要 API

```julia
# analytic power coefficients and TCI
power_fourier_coeff(m, k, L; scale=1)
power_mps(m, N, L; scale=1, tolerance=1e-10, maxbonddim=200)
power_mpo_tci(m, N, L; scale=1, ordering=:lsb_first,
              tolerance=1e-10, maxbonddim=200)

# differential operators and Hamiltonian
differential_mpo(order, N, L; coefficient=1)
kinetic_mpo(N, L)
build_power_hamiltonian(sites, m, N, L; scale=1,
                        tci_tolerance=1e-10,
                        tci_maxbonddim=200)
ground_state(H, sites; nsweeps=15, maxdim=64, cutoff=1e-12)

# shape/bond inspection
mps_shapes(state, sites)
mpo_shapes(operator, sites)
mps_bond_dims(state)
mpo_bond_dims(operator)
max_bond_dim(state_or_operator)

# real-space reconstruction
mps_to_realspace(state, sites, L; npoints=nothing, x=nothing)
plot_realspace(x, values; filename="wavefunction.svg")

# separable multi-dimensional Hamiltonians
identity_mpo(N; eltype=ComplexF64)
embed_mpo(component, dim, qubits_per_dim; eltype=ComplexF64)
build_separable_hamiltonian(sites, qubits_per_dim, half_width_per_dim;
                            powers=fill(2, D), scales=fill(1, D),
                            tci_tolerance=1e-10, tci_maxbonddim=200)
mps_to_realspace_nd(state, sites, qubits_per_dim, half_width_per_dim;
                    points_per_dim=nothing)
coeffs_to_realspace_nd(coefficients, grids, Ls)
plot_heatmap(x, y, Z; filename="field.svg", colorbar_label="value")

# non-separable multi-dimensional potentials
coulomb2d_fourier_coeff(kx, ky, Lx, Ly; charge=1)  # analytic closed form
coefficient_mpo_tci_nd(coefficient, qubits_per_dim; tolerance=1e-10, maxbonddim=200)
grid_potential_fourier_coefficients(V, qubits_per_dim, half_width_per_dim)  # when no closed form exists
potential_mpo_tci_nd(V, qubits_per_dim, half_width_per_dim;
                     tolerance=1e-10, maxbonddim=200)
```
