# One-dimensional harmonic oscillator

这个目录用 Fourier-QTT 与 ITensor DMRG 验证

```math
H=-\frac12\frac{d^2}{dx^2}+\frac{x^2}{2},\qquad
E_0=\frac12,\qquad
\psi_0(x)=\pi^{-1/4}e^{-x^2/2}.
```

`sho_model.jl` 直接使用 `../Juliatools/`：

- `power_mpo_tci(2, ...; scale=0.5)` 从 `x²/2` 的解析 Fourier coefficient 函数直接拟合势能 MPO；
- `differential_mpo(2, ...; coefficient=-0.5)` 建立 bond dim 3 的动能 MPO；
- `build_power_hamiltonian` 以 exact direct sum 合成 Hamiltonian；
- `ground_state_with_dimension_log` 用 DMRG 求基态 Fourier-coefficient MPS，并由 observer 收集每个 sweep 的最大 truncation error 与完整 bond dimensions；
- `mps_to_realspace` 与 `plot_realspace` 在可自订的实空间点上重建并绘图。

数组 shape 统一为：

```text
MPS A_i: (left_bond, physical_dim, right_bond)
MPO W_i: (left_bond, output_dim, input_dim, right_bond)
```

从专案根目录运行：

```bash
julia 1dsho/sho_model.jl
```

画图点数独立于 `2^N` 个 Fourier modes。可由 `SHOConfig.realspace_points` 设置，或作为第一个参数传入：

```bash
julia 1dsho/sho_model.jl 2500
```

目前默认使用 10 qubits（1024 modes）、20 个 DMRG sweeps、
`TCI tolerance=10^-12`、`DMRG cutoff=10^-8` 与 1601 个实空间点。
当前势能、动能与 Hamiltonian 的最大 bond dim 分别为 13、3 与 16。
输出包括：

- `sho_results.txt`：TCI estimated error、DMRG truncation error、能量、验证误差、最大 bond dim 与张量 shapes；
- `sho_mps_dimensions.csv`：每个完整 DMRG sweep 结束后的 MPS 最大 bond dimension 及各内部 bond dimensions；
- `sho_tensor_shapes.txt`：逐-site 的 MPS、势能 MPO、动能 MPO、Hamiltonian MPO shapes；
- `sho_ground_state.csv`：实空间数值波函数、解析波函数与误差；
- `sho_ground_state.svg`：基态波函数；
- `sho_error.svg`：绝对误差。

报告会区分 requested threshold 与实际记录值：

```text
TCI requested tolerance
TCI estimated relative error
DMRG requested cutoff
DMRG max truncation error
DMRG truncation error/sweep
```

TCI error 是自适应 pivot 上的 normalized interpolation-error estimate；DMRG error 是 observer 记录的每个 sweep 最大 SVD truncation error。
`sho_mps_dimensions.csv` 的栏位为 `sweep,max_bond_dimension,bond_1_2,...`，每次完整 sweep 完成后立即写入并 flush，因此长时间运算时也可以直接追踪 MPS 的成长过程。

总能量误差通常与 discarded weight 同阶，但 `<T>`、`<V>` 各自对波函数误差是一阶敏感量，尺度约为 truncation error 的平方根。因此程式分别使用 total-energy tolerance 与 component-energy tolerance，不再错误地把两者套用同一个线性门槛。无论最终 PASS 或 FAIL，完整 diagnostics 都会先写入 `sho_results.txt`。
