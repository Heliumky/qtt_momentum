"""
Tools for solving one-dimensional problems in a Fourier QTT basis.

This is the implementation entry point used directly by the tests. Individual
components are split by responsibility in the same directory. Site 1 is the
least-significant bit and Fourier indices use two's-complement ordering.
"""
module QTTFourier

using FFTW: ifft
using ITensors, ITensorMPS
using LinearAlgebra: Diagonal, svd
using QuadGK: quadgk
using Random: seed!
import TensorCrossInterpolation as TCI

include(joinpath(@__DIR__, "elemetary_func.jl"))
include(joinpath(@__DIR__, "diff_MPO.jl"))
include(joinpath(@__DIR__, "dmrg_engine.jl"))
include(joinpath(@__DIR__, "k_to_r.jl"))
include(joinpath(@__DIR__, "plot.jl"))
include(joinpath(@__DIR__, "multi_dim.jl"))
include(joinpath(@__DIR__, "k_to_r_nd.jl"))
include(joinpath(@__DIR__, "multi_dim_potential.jl"))

export twos_complement, freqs, sigma_to_n, vec_to_site_tensor,
       fit_tensor_train, fit_qtt_mps, power_fourier_coeff, power_mps,
       coefficient_mpo_tci, power_mpo_tci, coulomb2d_fourier_coeff,
       periodic_coulomb2d_fourier_coeff,
       diag_power_mpo, diag_quadratic_mpo, differential_mpo, kinetic_mpo,
       array_mpo_to_itensor, array_mps_to_itensor,
       build_hamiltonian, build_power_hamiltonian, ground_state, expect_mpo,
       mps_shapes, mpo_shapes, mps_bond_dims, mpo_bond_dims, max_bond_dim,
       dense_to_mps, mps_to_dense, mpo_to_dense, mpo_add,
       itensor_mps_to_vec, realspace_grid, fourier_basis,
       evaluate_fourier_series, coeffs_to_realspace, mps_to_realspace,
       plot_realspace,
       identity_mpo, embed_mpo, build_separable_hamiltonian,
       coeffs_to_realspace_nd, mps_to_realspace_nd, plot_heatmap,
       grid_potential_fourier_coefficients, coefficient_mpo_tci_nd,
       potential_mpo_tci_nd

end # module QTTFourier
