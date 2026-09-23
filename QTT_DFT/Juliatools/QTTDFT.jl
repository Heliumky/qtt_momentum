"""
Tools for solving continuum quantum problems in a Fourier QTT basis, with
potentials built through the discrete Fourier transform.

Potentials are fitted by TCI as diagonal real-space MPOs `V(x_j)` and rotated
to the Fourier basis with the quantics DFT MPO; kinetic operators are exact
diagonal Fourier-space MPOs. The Fourier register (site 1 = LSB,
two's-complement modes) is identical to the analytic `QTTFourier` package.
"""
module QTTDFT

using ITensors, ITensorMPS
using LinearAlgebra: Diagonal, norm, svd
using QuanticsTCI: quanticsfouriermpo
using Random: seed!
import TensorCrossInterpolation as TCI

include(joinpath(@__DIR__, "elemetary_func.jl"))
include(joinpath(@__DIR__, "diff_MPO.jl"))
include(joinpath(@__DIR__, "dmrg_engine.jl"))
include(joinpath(@__DIR__, "dft_mpo.jl"))
include(joinpath(@__DIR__, "k_to_r.jl"))
include(joinpath(@__DIR__, "plot.jl"))
include(joinpath(@__DIR__, "multi_dim.jl"))
include(joinpath(@__DIR__, "k_to_r_nd.jl"))
include(joinpath(@__DIR__, "multi_dim_potential.jl"))

export twos_complement, freqs, sigma_to_n, sigma_to_grid_index, grid_index_to_sigma,
       vec_to_site_tensor, realspace_grid, grid_coordinate, nearest_grid_sigma,
       fit_tensor_train, fit_qtt_mps, realspace_function_mps, power_potential_mps,
       mps_to_diagonal_mpo,
       frequency_weights, diag_power_mpo, diag_quadratic_mpo, differential_mpo, kinetic_mpo,
       fourier_transform_mpo, adjoint_mpo, grid_register_order, dense_fourier_matrix,
       fourier_space_operator, potential_mpo_dft, power_potential_mpo_dft,
       array_mpo_to_itensor, array_mps_to_itensor, itensor_mpo_to_arrays,
       build_hamiltonian, build_dft_hamiltonian, build_power_hamiltonian,
       ground_state, expect_mpo,
       mps_shapes, mpo_shapes, mps_bond_dims, mpo_bond_dims, max_bond_dim,
       dense_to_mps, mps_to_dense, mpo_to_dense, mpo_add,
       itensor_mps_to_vec, fourier_basis, evaluate_fourier_series,
       fourier_to_realspace_mps, realspace_mps_to_vec, mps_to_realspace,
       plot_realspace,
       RegisterLayout, dimension_sites, identity_mpo, embed_mpo, kron_mpo,
       fourier_transform_mpo_nd, build_separable_hamiltonian,
       coeffs_to_realspace_nd, mps_to_realspace_nd, plot_heatmap,
       realspace_function_mps_nd, potential_mpo_dft_nd, build_dft_hamiltonian_nd,
       coulomb2d_cell_average, realspace_state_mps_nd

end # module QTTDFT
