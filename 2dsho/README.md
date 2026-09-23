# Two-dimensional harmonic oscillator

这个目录用 Fourier-QTT 与 ITensor DMRG 验证各向同性二维简谐振子

```math
H=-\frac12\left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)
+\frac{x^2+y^2}{2},\qquad
E_0=1,\qquad
\psi_0(x,y)=\pi^{-1/2}e^{-(x^2+y^2)/2}.
```

`sho2d_model.jl` 直接使用 `../Juliatools/`，采用与 `1dsho/sho_model.jl` 相同的
TCI + 精确微分 MPO + ITensor DMRG 方法，扩展到可分离的多维 Hamiltonian：

- `N_list = [Nx, Ny]` 独立指定每个维度的 qubit 数（写法与 `3dH_p0_l/onepar.py`
  里 `N_list`/`rmin_vec`/`rmax_vec` 按维度分开设置的风格一致），维度 `d` 占用
  联合 register 中连续的 `N_list[d]` 个 site（先 x 后 y）；
- `build_separable_hamiltonian` 对每个维度独立调用 `power_mpo_tci(2, ...; scale=0.5)`
  与 `kinetic_mpo` 得到该维度的势能、动能 MPO，用 `embed_mpo` 把它 tensor 成
  `component ⊗ I`（对另一维度是恒等算符，不引入任何截断），再以 exact direct sum
  合成完整 Hamiltonian；
- `ground_state_with_dimension_log` 用 DMRG 求基态 Fourier-coefficient MPS，
  并由 observer 收集每个 sweep 的最大 truncation error 与完整 bond dimensions；
- `mps_to_realspace_nd` 用可分离 Fourier basis 在 `x` 与 `y` 的实空间网格外积上
  重建波函数，`plot_heatmap` 绘制 2D SVG 热图。

数组 shape 与 1dsho 相同：

```text
MPS A_i: (left_bond, physical_dim, right_bond)
MPO W_i: (left_bond, output_dim, input_dim, right_bond)
```

从专案根目录运行：

```bash
julia 2dsho/sho2d_model.jl
```

默认 `N_list=[8,8]`（每维 256 个 Fourier modes）、`half_width_list=[8.0,8.0]`、
20 个 DMRG sweeps、`TCI tolerance=10^-12`、`DMRG cutoff=10^-8`，
每维 121 个实空间画图点。可用两个位置参数分别覆盖 `x`、`y` 方向的画图点数：

```bash
julia 2dsho/sho2d_model.jl 161 161
```

当前势能、动能（每个维度）与 Hamiltonian 的最大 bond dim 分别为 13、3 与 18；
基态 MPS 最大 bond dim 为 4。输出包括：

- `sho2d_results.txt`：每个维度的 TCI estimated error、DMRG truncation error、
  能量、`<Tx>`、`<Ty>`、`<Vx>`、`<Vy>`、验证误差、最大 bond dim 与张量 shapes；
- `sho2d_mps_dimensions.csv`：每个完整 DMRG sweep 结束后的 MPS 最大 bond
  dimension 及各内部 bond dimensions；
- `sho2d_tensor_shapes.txt`：逐-site 的 MPS shapes，以及按维度分开的势能、
  动能 MPO shapes 与合成后 Hamiltonian MPO shapes；
- `sho2d_ground_state.csv`：`(x, y)` 网格外积上的数值波函数、解析波函数与误差；
- `sho2d_ground_state.svg`：基态波函数 `Re ψ(x,y)` 热图；
- `sho2d_error.svg`：绝对误差 `|ψ_numerical-ψ_exact|` 热图。

报告会区分 requested threshold 与实际记录值，并按维度分别列出 TCI 诊断：

```text
TCI requested tolerance
TCI estimated relative error (dim 1/2)
DMRG requested cutoff
DMRG max truncation error
DMRG truncation error/sweep
```

验证逻辑与 1dsho 相同：总能量误差通常与 discarded weight 同阶，但
`<Tx>`、`<Ty>`、`<Vx>`、`<Vy>` 各自对波函数误差是一阶敏感量，尺度约为
truncation error 的平方根，因此分别使用 total-energy tolerance 与
component-energy tolerance。无论最终 PASS 或 FAIL，完整 diagnostics 都会先
写入 `sho2d_results.txt`。

## 与 `Juliatools` 的关系

这个模型没有引入新的截断方式，只是把 1D 组件按维度精确地张成联合 Hamiltonian。
新增的通用工具都在 `../Juliatools/`：

```julia
identity_mpo(N)                                   # multi_dim.jl
embed_mpo(component, dim, qubits_per_dim)          # multi_dim.jl
build_separable_hamiltonian(sites, qubits_per_dim, # multi_dim.jl
                            half_width_per_dim; powers, scales, ...)
coeffs_to_realspace_nd(coefficients, grids, Ls)    # k_to_r_nd.jl
mps_to_realspace_nd(state, sites, qubits_per_dim,  # k_to_r_nd.jl
                    half_width_per_dim; points_per_dim)
plot_heatmap(x, y, Z; filename, title, colorbar_label)  # plot.jl
```

它们对维度数没有硬编码限制，若未来需要三维或更高维的可分离势能，同一套函数
可以直接以更长的 `N_list`/`half_width_list` 重用。
