# Two-dimensional harmonic oscillator (DFT potential)

这个目录用 Fourier-QTT、DFT MPO 与 ITensor DMRG 验证各向同性二维简谐振子

```math
H=-\frac12\left(\frac{\partial^2}{\partial x^2}+\frac{\partial^2}{\partial y^2}\right)
+\frac{x^2+y^2}{2},\qquad
E_0=1,\qquad
\psi_0(x,y)=\pi^{-1/2}e^{-(x^2+y^2)/2}.
```

`sho2d_model.jl` 直接使用 `../Juliatools/`（`QTTDFT` 模组），方法与
`1dsho/sho_model.jl` 相同，扩展到可分离的多维 Hamiltonian：

- `N_list = [Nx, Ny]` 独立指定每个维度的 qubit 数；默认 `:block` layout，维度 `d`
  占用联合 register 中连续的 `N_list[d]` 个 site（先 x 后 y）。也可传入
  `RegisterLayout(N_list; scheme=:interleaved)`。
- `build_separable_hamiltonian` 对每个维度独立做：实空间 TCI 拟合 `x_d²/2` →
  1D DFT MPO `U_d` → `V_k = U_d diag(V) U_d†`，再以 `embed_mpo` 精确嵌入为
  `component ⊗ I`；动能是精确的 `kinetic_mpo`。所有 `2D` 项以 exact direct sum 合成。
- `ground_state_with_dimension_log` 用 DMRG 求基态 Fourier-coefficient MPS。
- `mps_to_realspace_nd` 重建实空间波函数：原生网格用联合 `U†` MPO，其它点数用可分离
  Fourier sum；`plot_heatmap` 绘制 2D SVG 热图。

从 `QTT_DFT/` 目录运行：

```bash
julia --project=. 2dsho/sho2d_model.jl
```

默认 `N_list=[8,8]`、`half_width_list=[8.0,8.0]`、`TCI tolerance=10^-12`、
`DMRG cutoff=10^-8`、20 sweeps、`dft_cutoff=10^-26`。

## 与解析版（`../../2dsho`）比较

| 量 | 解析版 | DFT（本目录） |
| --- | --- | --- |
| TCI target evaluations（每维） | 7009 / 6491 | 166 / 173 |
| 势能 MPO max bond dim（每维） | 13 | 13 |
| Hamiltonian MPO max bond dim | 18 | 18 |
| ground-state MPS max bond dim | 4 | 4 |
| DMRG E0 − 1 | −9.7e-15 | +3.4e-12 |
| wavefunction L2 error | 6.3e-9 | 8.9e-7 |

波函数误差的差别来自 DMRG 在 20 个 sweep 内的收敛程度，而不是 Hamiltonian：
DFT 版的 truncation error 直到第 18 个 sweep 才降到 1e-11 以下（见
`sho2d_results.txt`）。用同一个 DFT Hamiltonian 跑 40 个 sweep，得到
`E0 − 1 = 1.1e-12`、wavefunction L2 error `1.4e-9`。

## 输出

- `sho2d_results.txt`：每维 TCI error、global error estimate 与 evaluation 数、DMRG truncation error、能量分量、验证结果、张量 shapes；
- `sho2d_mps_dimensions.csv`：每个 sweep 结束后的 MPS bond dimensions；
- `sho2d_tensor_shapes.txt`：逐-site 的 MPS / MPO shapes；
- `sho2d_ground_state.csv`、`sho2d_ground_state.svg`、`sho2d_error.svg`：实空间波函数、解析解与误差热图。
