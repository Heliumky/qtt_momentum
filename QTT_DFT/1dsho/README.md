# One-dimensional harmonic oscillator (DFT potential)

这个目录用 Fourier-QTT、离散傅立叶变换（DFT）MPO 与 ITensor DMRG 验证

```math
H=-\frac12\frac{d^2}{dx^2}+\frac{x^2}{2},\qquad
E_0=\frac12,\qquad
\psi_0(x)=\pi^{-1/4}e^{-x^2/2}.
```

`sho_model.jl` 直接使用 `../Juliatools/`（`QTTDFT` 模组）：

- `build_power_hamiltonian(sites, 2, N, L; scale=0.5, pivot_points=[0.0])`：
  1. `realspace_function_mps` 在实空间网格 `x_j=-L+(j+offset)·2L/M` 上用 TCI 拟合 `x_j²/2`（二次多项式的 quantics rank 恰为 3，TCI 只查询 266 个网格点）；
  2. `fourier_transform_mpo` 用 `QuanticsTCI.quanticsfouriermpo` 建立 DFT MPO `U`（bond dim 13，精度 ~1e-12）；
  3. `V_k = U · diag(V) · U†` 以 ITensor `apply` 压缩（`cutoff=1e-26`，即约 1e-13 的相对算符误差）。
- `kinetic_mpo` 建立 bond dim 3 的精确动能 MPO；两者以 exact direct sum 合成 Hamiltonian。
- `ground_state_with_dimension_log` 用 DMRG 求基态 Fourier-coefficient MPS。
- `mps_to_realspace` 与 `plot_realspace` 在可自订的实空间点上重建并绘图；原生 `2^N` 网格直接用 `U†` 作用在 MPS 上（`method = :dft_mpo`）。

数组 shape 统一为：

```text
MPS A_i: (left_bond, physical_dim, right_bond)
MPO W_i: (left_bond, output_dim, input_dim, right_bond)
```

从 `QTT_DFT/` 目录运行：

```bash
julia --project=. 1dsho/sho_model.jl
julia --project=. 1dsho/sho_model.jl 2500   # 指定画图点数
```

`SHOConfig` 新增的参数：

- `grid_offset`：实空间网格相对 `-L` 的偏移（以格距为单位，默认 0，网格包含 `x=0`）；
- `dft_cutoff`：`U V U†` 的 SVD 压缩门槛（ITensor 的 relative squared cutoff）。

## 与解析版（`../../1dsho`）比较

同样的 `N=10`、`L=8`、`TCI tolerance=10^-12`、`DMRG cutoff=10^-8`、20 sweeps：

| 量 | 解析 Fourier coefficient + paired-index TCI | DFT（本目录） |
| --- | --- | --- |
| TCI target evaluations | 12950 | 266 |
| 实空间势能 MPS bond dim | – | 3 |
| k 空间势能 MPO max bond dim | 13 | 14 |
| Hamiltonian MPO max bond dim | 16 | 17 |
| ground-state MPS max bond dim | 4 | 4 |
| DMRG E0 − 0.5 | −5.6e-17 | +1.0e-13 |
| wavefunction L2 error | 9.9e-10 | 9.9e-10 |

两者的 k 空间势能 MPO bond dim 几乎相同；DFT 版本的主要优势是 TCI 只需在实空间拟合一个 rank-3 的函数，而且任何可取样的 `V(x)` 都能直接使用，不必推导解析 Fourier coefficient。

## 输出

- `sho_results.txt`：TCI estimated error、TCI global error estimate（`TCI.estimatetrueerror`）、DFT MPO bond dim、DMRG truncation error、能量、验证误差、最大 bond dim 与张量 shapes；
- `sho_mps_dimensions.csv`：每个 DMRG sweep 结束后的 MPS bond dimensions；
- `sho_tensor_shapes.txt`：逐-site 的 MPS、势能 MPO、动能 MPO、Hamiltonian MPO shapes；
- `sho_ground_state.csv`、`sho_ground_state.svg`、`sho_error.svg`：实空间波函数、解析解与误差。

总能量误差与 discarded weight 同阶，但 `<T>`、`<V>` 对波函数误差是一阶敏感量，因此程式分别使用 total-energy tolerance 与 component-energy tolerance。无论最终 PASS 或 FAIL，完整 diagnostics 都会先写入 `sho_results.txt`。
