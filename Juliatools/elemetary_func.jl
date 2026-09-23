"""Analytic elementary functions and their TCI-fitted MPS/MPO representations."""

"""Convert an unsigned `N`-bit integer to its signed two's-complement value."""
function twos_complement(index::Integer, N::Integer)
    N >= 1 || throw(ArgumentError("N must be positive"))
    0 <= index < 2^N || throw(ArgumentError("index must lie in 0:$(2^N - 1)"))
    return index >= 2^(N - 1) ? index - 2^N : index
end

"""Fourier mode numbers in QTT storage order (site 1 is the LSB)."""
freqs(N::Integer) = [twos_complement(index, N) for index in 0:(2^N - 1)]

"""Map 1-based binary physical indices to a signed Fourier mode."""
function sigma_to_n(sigma::AbstractVector{<:Integer})
    isempty(sigma) && throw(ArgumentError("sigma cannot be empty"))
    all(index -> index == 1 || index == 2, sigma) ||
        throw(ArgumentError("every physical index must be 1 or 2"))

    N = length(sigma)
    unsigned_part = sum(
        (sigma[site] - 1) * (1 << (site - 1)) for site in 1:(N - 1);
        init=0,
    )
    sign_part = (sigma[N] - 1) * (1 << (N - 1))
    return unsigned_part - sign_part
end

"""Reshape a dense length-`2^N` vector into `N` binary physical dimensions."""
function vec_to_site_tensor(values::AbstractVector, N::Integer)
    length(values) == 2^N || throw(DimensionMismatch("expected a vector of length 2^N"))
    return reshape(values, ntuple(_ -> 2, N))
end

"""
    fit_tensor_train(target, local_dimensions; tolerance=1e-10,
                     maxbonddim=200, initialpivot=nothing,
                     return_diagnostics=false)

Fit a scalar function of a discrete index string directly as a tensor train.
TCI chooses the queried entries; the full tensor is never formed. Returned MPS
tensors have shape `(left_bond, physical_dim, right_bond)`.

With `return_diagnostics=true`, return `(; tensors, diagnostics)` where the
diagnostics contain TCI's normalized interpolation-error history, bond-dimension
history, and number of target-function evaluations.
"""
function fit_tensor_train(target, local_dimensions::AbstractVector{<:Integer};
                          tolerance::Real=1e-10,
                          maxbonddim::Integer=200,
                          initialpivot::Union{Nothing,AbstractVector{<:Integer}}=nothing,
                          return_diagnostics::Bool=false)
    isempty(local_dimensions) && throw(ArgumentError("local_dimensions cannot be empty"))
    all(>(0), local_dimensions) || throw(ArgumentError("local dimensions must be positive"))
    tolerance > 0 || throw(ArgumentError("tolerance must be positive"))
    maxbonddim >= 1 || throw(ArgumentError("maxbonddim must be positive"))

    dimensions = Int.(local_dimensions)
    N = length(dimensions)
    cache = Dict{Tuple{Vararg{Int}},ComplexF64}()
    black_box(indices::Vector{Int}) = get!(
        () -> ComplexF64(target(indices)),
        cache,
        Tuple(indices),
    )

    starting_pivot = isnothing(initialpivot) ? ones(Int, N) : Int.(initialpivot)
    length(starting_pivot) == N || throw(DimensionMismatch("initialpivot must have length N"))
    all(1 <= starting_pivot[site] <= dimensions[site] for site in 1:N) ||
        throw(ArgumentError("initialpivot entries exceed their local dimensions"))

    optimized_pivot = TCI.optfirstpivot(black_box, dimensions, starting_pivot)
    iszero(black_box(optimized_pivot)) && throw(ArgumentError(
        "TCI could not find a nonzero pivot; pass a nonzero initialpivot explicitly",
    ))

    tci, ranks, errors = TCI.crossinterpolate2(
        ComplexF64,
        black_box,
        dimensions,
        [optimized_pivot];
        tolerance=tolerance,
        maxbonddim=maxbonddim,
    )
    tensors = TCI.sitetensors(tci)
    diagnostics = (
        requested_tolerance=Float64(tolerance),
        estimated_relative_error=isempty(errors) ? 0.0 : last(errors),
        error_history=collect(errors),
        bond_dimension_history=collect(ranks),
        target_evaluations=length(cache),
    )
    return return_diagnostics ? (; tensors, diagnostics) : tensors
end

"""Fit a function of `N` binary indices as a QTT/MPS."""
function fit_qtt_mps(target, N::Integer; kwargs...)
    N >= 1 || throw(ArgumentError("N must be positive"))
    return fit_tensor_train(target, fill(2, N); kwargs...)
end

"""
    power_fourier_coeff(m, k, L; scale=1)

Analytic Fourier coefficient
`(2L)^(-1) integral(scale*x^m*exp(i*pi*k*x/L), x=-L..L)`.
For nonzero integer `k`, only a finite analytic recurrence is evaluated.
"""
function power_fourier_coeff(m::Integer, k::Integer, L::Real; scale::Number=1)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    L > 0 || throw(ArgumentError("L must be positive"))

    if iszero(k)
        average = isodd(m) ? 0.0 : L^m / (m + 1)
        return complex(scale * average)
    end

    # J_m(k) = 1/2 integral_{-1}^{1} u^m exp(i*pi*k*u) du, q=i*pi*k.
    # J_0=0 and J_m=(-1)^k(1-(-1)^m)/(2q)-m*J_{m-1}/q.
    q = im * pi * k
    endpoint_phase = isodd(k) ? -1.0 : 1.0
    coefficient = zero(ComplexF64)
    for degree in 1:m
        boundary = isodd(degree) ? endpoint_phase / q : zero(ComplexF64)
        coefficient = boundary - degree * coefficient / q
    end
    return scale * L^m * coefficient
end

"""
    power_mps(m, N, L; scale=1, tolerance=1e-10, maxbonddim=200)

Fit the analytic Fourier-coefficient function of `scale*x^m` as an MPS with
shape `(left_bond, physical_dim=2, right_bond)` at every site.
"""
function power_mps(m::Integer, N::Integer, L::Real;
                   scale::Number=1,
                   tolerance::Real=1e-10,
                   maxbonddim::Integer=200,
                   return_diagnostics::Bool=false)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    coefficient_cache = Dict{Int,ComplexF64}()
    coefficient(k::Int) = get!(
        () -> power_fourier_coeff(m, k, L; scale=scale),
        coefficient_cache,
        k,
    )
    return fit_qtt_mps(
        sigma -> coefficient(sigma_to_n(sigma)),
        N;
        tolerance=tolerance,
        maxbonddim=maxbonddim,
        return_diagnostics=return_diagnostics,
    )
end


"""
    coefficient_mpo_tci(coefficient, N; ordering=:lsb_first,
                        tolerance=1e-10, maxbonddim=200,
                        return_diagnostics=false)

Directly fit `W[out,in] = coefficient(input_mode-output_mode)` as an MPO. Each
TCI site uses a paired local index `(output_bit,input_bit)` of dimension four.
The returned tensor shape is
`(left_bond, output_dim=2, input_dim=2, right_bond)`.
"""
function coefficient_mpo_tci(coefficient, N::Integer;
                             ordering::Symbol=:lsb_first,
                             tolerance::Real=1e-10,
                             maxbonddim::Integer=200,
                             return_diagnostics::Bool=false)
    N >= 1 || throw(ArgumentError("N must be positive"))
    ordering in (:lsb_first, :msb_first) ||
        throw(ArgumentError("ordering must be :lsb_first or :msb_first"))

    coefficient_cache = Dict{Int,ComplexF64}()
    cached_coefficient(delta::Int) = get!(
        () -> ComplexF64(coefficient(delta)),
        coefficient_cache,
        delta,
    )

    function operator_element(paired_indices::Vector{Int})
        output_sigma = Vector{Int}(undef, N)
        input_sigma = Vector{Int}(undef, N)
        for site in 1:N
            paired_digit = paired_indices[site] - 1
            output_sigma[site] = mod(paired_digit, 2) + 1
            input_sigma[site] = div(paired_digit, 2) + 1
        end
        if ordering === :msb_first
            reverse!(output_sigma)
            reverse!(input_sigma)
        end
        delta = sigma_to_n(input_sigma) - sigma_to_n(output_sigma)
        return cached_coefficient(delta)
    end

    fit = fit_tensor_train(
        operator_element,
        fill(4, N);
        tolerance=tolerance,
        maxbonddim=maxbonddim,
        return_diagnostics=true,
    )
    tensors = [
        reshape(tensor, size(tensor, 1), 2, 2, size(tensor, 3))
        for tensor in fit.tensors
    ]
    return return_diagnostics ? (; tensors, diagnostics=fit.diagnostics) : tensors
end

"""
    power_mpo_tci(m, N, L; scale=1, kwargs...)

Directly fit the Fourier-basis multiplication MPO for `scale*x^m` using the
analytic coefficient function and paired `(output_bit,input_bit)` site order.
"""
function power_mpo_tci(m::Integer, N::Integer, L::Real;
                       scale::Number=1, kwargs...)
    m >= 0 || throw(ArgumentError("m must be nonnegative"))
    return coefficient_mpo_tci(
        delta -> power_fourier_coeff(m, delta, L; scale=scale),
        N;
        kwargs...,
    )
end

"""
    periodic_coulomb2d_fourier_coeff(kx, ky, Lx, Ly; charge=1)

Coefficient of the neutralized, fully periodized 2D Coulomb sum. For every
nonzero mode this samples the infinite-plane transform `2*pi/|q|`; its
divergent zero mode is set to zero by the neutralizing-background convention.
This is retained for explicit comparisons with the finite-box coefficient.
"""
function periodic_coulomb2d_fourier_coeff(kx::Integer, ky::Integer,
                                          Lx::Real, Ly::Real;
                                          charge::Number=1)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    (kx == 0 && ky == 0) && return zero(ComplexF64)
    qx = pi * kx / Lx
    qy = pi * ky / Ly
    q = sqrt(qx^2 + qy^2)
    return complex(-charge * pi / (2 * Lx * Ly * q))
end

"""
    coulomb2d_fourier_coeff(kx, ky, Lx, Ly;
                            charge=1, quadrature_tolerance=1e-12)

Fourier-series coefficient of the finite-rectangle potential
`-charge/sqrt(x^2+y^2)` on `[-Lx,Lx] x [-Ly,Ly]`:

```
qx = pi*kx/Lx,  qy = pi*ky/Ly,  q = sqrt(qx^2+qy^2)
coefficient = -charge/(Lx*Ly) * integral(
    cos(qx*x)*cos(qy*y)/sqrt(x^2+y^2), x=0..Lx, y=0..Ly,
)
```

The integrable origin singularity is removed analytically. After splitting
the rectangle along the ray through its upper-right corner, the radial
integral is evaluated in closed form and only a smooth one-dimensional angle
integral is passed to `QuadGK`. The zero mode uses its exact closed form

`-charge * (Lx*asinh(Ly/Lx) + Ly*asinh(Lx/Ly)) / (Lx*Ly)`.

For a square this is `-2*charge*log(1+sqrt(2))/L`. No softening, real-space
grid, or FFT is used. `quadrature_tolerance` controls both relative and scaled
absolute error of the remaining one-dimensional integral.
"""
function coulomb2d_fourier_coeff(kx::Integer, ky::Integer, Lx::Real, Ly::Real;
                                 charge::Number=1,
                                 quadrature_tolerance::Real=1e-12)
    Lx > 0 && Ly > 0 || throw(ArgumentError("Lx and Ly must be positive"))
    quadrature_tolerance > 0 ||
        throw(ArgumentError("quadrature_tolerance must be positive"))
    if kx == 0 && ky == 0
        box_average = (
            Lx * asinh(Ly / Lx) + Ly * asinh(Lx / Ly)
        ) / (Lx * Ly)
        return complex(-charge * box_average)
    end

    # The coefficient is even in each mode, which also improves cache reuse
    # in callers that canonicalize keys with abs(kx), abs(ky).
    qx = pi * abs(kx) / Lx
    qy = pi * abs(ky) / Ly
    corner_angle = atan(Ly, Lx)

    # Integral_0^R cos(a*r)cos(b*r)dr, written with normalized sinc so the
    # removable a±b=0 limits are handled without branches.
    radial_integral(a, b, R) = R * (
        sinc((a - b) * R / pi) + sinc((a + b) * R / pi)
    ) / 2

    x_boundary_integrand(theta) = radial_integral(
        qx * cos(theta), qy * sin(theta), Lx / cos(theta),
    )
    y_boundary_integrand(theta) = radial_integral(
        qx * cos(theta), qy * sin(theta), Ly / sin(theta),
    )
    absolute_tolerance = quadrature_tolerance * max(Lx, Ly)
    first_part = quadgk(
        x_boundary_integrand, 0.0, corner_angle;
        rtol=quadrature_tolerance, atol=absolute_tolerance,
    )[1]
    second_part = quadgk(
        y_boundary_integrand, corner_angle, pi / 2;
        rtol=quadrature_tolerance, atol=absolute_tolerance,
    )[1]
    return complex(-charge * (first_part + second_part) / (Lx * Ly))
end
