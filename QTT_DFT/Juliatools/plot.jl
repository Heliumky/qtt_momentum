"""Small, dependency-free SVG plots for reconstructed real-space functions."""

function _plot_values(values::AbstractVector{<:Number}, quantity::Symbol)
    quantity === :real && return real.(values), "Re f(x)"
    quantity === :imag && return imag.(values), "Im f(x)"
    quantity === :abs && return abs.(values), "|f(x)|"
    quantity === :density && return abs2.(values), "|f(x)|²"
    throw(ArgumentError("quantity must be :real, :imag, :abs, or :density"))
end

function _xml_escape(text)
    escaped = replace(string(text), '&' => "&amp;")
    escaped = replace(escaped, '<' => "&lt;")
    escaped = replace(escaped, '>' => "&gt;")
    return replace(escaped, '"' => "&quot;")
end

"""
    plot_realspace(x, values; filename=nothing, quantity=:real, title="Real-space function")

Render a reconstructed function as SVG without introducing a plotting-package
dependency. If `filename` is omitted, return the SVG string; otherwise write
the file and return its absolute path. `quantity` may be `:real`, `:imag`,
`:abs`, or `:density`.
"""
function plot_realspace(x::AbstractVector{<:Real}, values::AbstractVector{<:Number};
                        filename::Union{Nothing,AbstractString}=nothing,
                        quantity::Symbol=:real,
                        title::AbstractString="Real-space function",
                        width::Integer=900,
                        height::Integer=520,
                        stroke::AbstractString="#2457c5")
    length(x) == length(values) || throw(DimensionMismatch("x and values must have equal length"))
    length(x) >= 2 || throw(ArgumentError("at least two points are required"))
    width >= 320 || throw(ArgumentError("width must be at least 320"))
    height >= 240 || throw(ArgumentError("height must be at least 240"))

    y, y_label = _plot_values(values, quantity)
    all(isfinite, x) && all(isfinite, y) || throw(ArgumentError("plot data must be finite"))
    x_min, x_max = extrema(x)
    y_min, y_max = extrema(y)
    x_min == x_max && throw(ArgumentError("x coordinates must span a nonzero interval"))
    if y_min == y_max
        padding = iszero(y_min) ? 1.0 : abs(y_min) * 0.05
        y_min -= padding
        y_max += padding
    end

    left, right, top, bottom = 82.0, 28.0, 52.0, 68.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    sx(value) = left + (value - x_min) / (x_max - x_min) * plot_width
    sy(value) = top + (y_max - value) / (y_max - y_min) * plot_height
    points = join(("$(round(sx(xi); digits=2)),$(round(sy(yi); digits=2))" for (xi, yi) in zip(x, y)), " ")
    zero_y = y_min <= 0 <= y_max ? sy(0) : top + plot_height

    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">
    <rect width="100%" height="100%" fill="white"/>
    <text x="$(width / 2)" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">$(_xml_escape(title))</text>
    <rect x="$left" y="$top" width="$plot_width" height="$plot_height" fill="none" stroke="#777" stroke-width="1"/>
    <line x1="$left" y1="$zero_y" x2="$(left + plot_width)" y2="$zero_y" stroke="#aaa" stroke-width="1"/>
    <polyline points="$points" fill="none" stroke="$(_xml_escape(stroke))" stroke-width="2" stroke-linejoin="round"/>
    <text x="$(width / 2)" y="$(height - 18)" text-anchor="middle" font-family="sans-serif" font-size="15">x</text>
    <text x="20" y="$(height / 2)" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 20 $(height / 2))">$(_xml_escape(y_label))</text>
    <text x="$left" y="$(height - bottom + 23)" text-anchor="middle" font-family="monospace" font-size="12">$(round(x_min; sigdigits=5))</text>
    <text x="$(left + plot_width)" y="$(height - bottom + 23)" text-anchor="middle" font-family="monospace" font-size="12">$(round(x_max; sigdigits=5))</text>
    <text x="$(left - 8)" y="$(top + 4)" text-anchor="end" font-family="monospace" font-size="12">$(round(y_max; sigdigits=5))</text>
    <text x="$(left - 8)" y="$(top + plot_height + 4)" text-anchor="end" font-family="monospace" font-size="12">$(round(y_min; sigdigits=5))</text>
</svg>
"""

    if isnothing(filename)
        return svg
    end
    path = abspath(filename)
    open(path, "w") do io
        write(io, svg)
    end
    return path
end

"""Interpolate a diverging blue-white-red colormap at `t in [0,1]`."""
function _diverging_color(t::Real)
    t = clamp(t, 0.0, 1.0)
    stops = ((0.0, (33, 102, 172)), (0.5, (247, 247, 247)), (1.0, (178, 24, 43)))
    lo, hi = t <= 0.5 ? (stops[1], stops[2]) : (stops[2], stops[3])
    t0, c0 = lo
    t1, c1 = hi
    s = t1 == t0 ? 0.0 : (t - t0) / (t1 - t0)
    channel(i) = round(Int, c0[i] + s * (c1[i] - c0[i]))
    return "rgb($(channel(1)),$(channel(2)),$(channel(3)))"
end

"""
    plot_heatmap(x, y, Z; filename=nothing, title="Heatmap", colorbar_label="value")

Render a 2D scalar field `Z[i,j]` (evaluated at `x[i], y[j]`) as an SVG
heatmap with a diverging color scale and colorbar. If `filename` is omitted,
return the SVG string; otherwise write the file and return its absolute
path.
"""
function plot_heatmap(x::AbstractVector{<:Real}, y::AbstractVector{<:Real},
                      Z::AbstractMatrix{<:Real};
                      filename::Union{Nothing,AbstractString}=nothing,
                      title::AbstractString="Heatmap",
                      colorbar_label::AbstractString="value",
                      width::Integer=760, height::Integer=680)
    size(Z) == (length(x), length(y)) ||
        throw(DimensionMismatch("Z must have size (length(x), length(y))"))
    length(x) >= 2 && length(y) >= 2 ||
        throw(ArgumentError("at least two points are required per axis"))
    all(isfinite, Z) || throw(ArgumentError("plot data must be finite"))
    width >= 320 || throw(ArgumentError("width must be at least 320"))
    height >= 240 || throw(ArgumentError("height must be at least 240"))

    z_min, z_max = extrema(Z)
    if z_min == z_max
        padding = iszero(z_min) ? 1.0 : abs(z_min) * 0.05
        z_min -= padding
        z_max += padding
    end

    left, right, top, bottom, legend_width = 82.0, 96.0, 52.0, 68.0, 26.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_min, x_max = extrema(x)
    y_min, y_max = extrema(y)
    cell_width = plot_width / length(x)
    cell_height = plot_height / length(y)

    rects = IOBuffer()
    for (i, _) in enumerate(x), (j, _) in enumerate(y)
        px = left + (i - 1) * cell_width
        py = top + plot_height - j * cell_height
        color = _diverging_color((Z[i, j] - z_min) / (z_max - z_min))
        print(
            rects,
            "<rect x=\"$(round(px; digits=2))\" y=\"$(round(py; digits=2))\" ",
            "width=\"$(ceil(cell_width; digits=2))\" height=\"$(ceil(cell_height; digits=2))\" ",
            "fill=\"$color\" stroke=\"none\"/>",
        )
    end

    legend_x = left + plot_width + 24
    legend_stops = join(
        (
            "<stop offset=\"$(round(100 * (1 - t); digits=1))%\" stop-color=\"$(_diverging_color(t))\"/>"
            for t in range(0, 1; length=9)
        ),
        "\n        ",
    )

    svg = """<svg xmlns="http://www.w3.org/2000/svg" width="$width" height="$height" viewBox="0 0 $width $height">
    <defs>
        <linearGradient id="legend" x1="0" y1="0" x2="0" y2="1">
        $legend_stops
        </linearGradient>
    </defs>
    <rect width="100%" height="100%" fill="white"/>
    <text x="$(width / 2)" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">$(_xml_escape(title))</text>
    <g>$(String(take!(rects)))</g>
    <rect x="$left" y="$top" width="$plot_width" height="$plot_height" fill="none" stroke="#777" stroke-width="1"/>
    <rect x="$legend_x" y="$top" width="$legend_width" height="$plot_height" fill="url(#legend)" stroke="#777" stroke-width="1"/>
    <text x="$(legend_x + legend_width + 6)" y="$(top + 4)" font-family="monospace" font-size="12">$(round(z_max; sigdigits=5))</text>
    <text x="$(legend_x + legend_width + 6)" y="$(top + plot_height + 4)" font-family="monospace" font-size="12">$(round(z_min; sigdigits=5))</text>
    <text x="$(legend_x - 6)" y="$(top - 10)" text-anchor="start" font-family="sans-serif" font-size="12">$(_xml_escape(colorbar_label))</text>
    <text x="$(width / 2)" y="$(height - 18)" text-anchor="middle" font-family="sans-serif" font-size="15">x</text>
    <text x="20" y="$(height / 2)" text-anchor="middle" font-family="sans-serif" font-size="15" transform="rotate(-90 20 $(height / 2))">y</text>
    <text x="$left" y="$(height - bottom + 23)" text-anchor="middle" font-family="monospace" font-size="12">$(round(x_min; sigdigits=5))</text>
    <text x="$(left + plot_width)" y="$(height - bottom + 23)" text-anchor="middle" font-family="monospace" font-size="12">$(round(x_max; sigdigits=5))</text>
    <text x="$(left - 8)" y="$(top + plot_height)" text-anchor="end" font-family="monospace" font-size="12">$(round(y_min; sigdigits=5))</text>
    <text x="$(left - 8)" y="$(top + 8)" text-anchor="end" font-family="monospace" font-size="12">$(round(y_max; sigdigits=5))</text>
</svg>
"""

    if isnothing(filename)
        return svg
    end
    path = abspath(filename)
    open(path, "w") do io
        write(io, svg)
    end
    return path
end

"""Reconstruct Fourier coefficients on a native or user-selected grid and render them."""
function plot_realspace(coefficients::AbstractVector{<:Number}, L::Real;
                        x::Union{Nothing,AbstractVector}=nothing,
                        npoints::Union{Nothing,Integer}=nothing,
                        kwargs...)
    !isnothing(x) && !isnothing(npoints) &&
        throw(ArgumentError("pass either x or npoints, not both"))
    if !isnothing(x)
        grid = collect(x)
        values = evaluate_fourier_series(coefficients, grid, L)
    else
        point_count = isnothing(npoints) ? length(coefficients) : npoints
        grid = realspace_grid(point_count, L)
        values = point_count == length(coefficients) ?
                 coeffs_to_realspace(coefficients, L) :
                 evaluate_fourier_series(coefficients, grid, L)
    end
    return plot_realspace(grid, values; kwargs...)
end

"""Contract a Fourier-coefficient MPS, reconstruct it, and render it."""
function plot_realspace(state::MPS, sites::AbstractVector{<:Index}, L::Real;
                        x::Union{Nothing,AbstractVector}=nothing,
                        npoints::Union{Nothing,Integer}=nothing,
                        normalize_coefficients::Bool=false,
                        kwargs...)
    result = mps_to_realspace(
        state,
        sites,
        L;
        x=x,
        npoints=npoints,
        normalize_coefficients=normalize_coefficients,
    )
    return plot_realspace(result.x, result.values; kwargs...)
end
