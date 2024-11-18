using LinearAlgebra
using OMEinsum
using Zygote
using Optim
using SuperVUMPS

const χ₁ = 8
const χ₂ = 2
const d = 2
const D = 4

struct PEPS2 <: Manifold end

function proj!(Q, R)
    Q .= (Q .+ conj.(ein"ijkl -> klij"(Q))) ./ 2
    R .= (R .+ conj.(ein"ijk, k -> jik"(R, [1.0, -1.0, 1.0]))) ./ 2
end

function Optim.retract!(::PEPS2, qr)
    Q = reshape(qr[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂) .+ im * reshape(qr[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
    R = reshape(qr[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1) .+ im * reshape(qr[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)
    proj!(Q, R)
    qr .= (x -> hcat(real.(x), imag.(x)))(vcat(vec(Q), vec(R)))
end

function Optim.project_tangent!(::PEPS2, dqr, qr)
    Q = reshape(dqr[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂) .+ im * reshape(dqr[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
    R = reshape(dqr[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1) .+ im * reshape(dqr[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)
    proj!(Q, R)
    dqr .= (x -> hcat(real.(x), imag.(x)))(vcat(vec(Q), vec(R)))
end

function main()
    Sx = [0.0 0.5; 0.5 0.0]
    Sy = [0.0 0.5; -0.5 0.0]
    Sz = [0.5 0.0; 0.0 -0.5]
    M = zeros(2, 2, D - 1)
    M[:, :, 1] .= Sx
    M[:, :, 2] .= Sy
    M[:, :, 3] .= Sz

    Q = randn(ComplexF64, χ₂, χ₂, χ₂, χ₂)
    R = randn(ComplexF64, χ₂, χ₂, D - 1)
    proj!(Q, R)

    println("========================")
    println("χ = (χ₁, χ₂) = ", (χ₁, χ₂))
    println("========================")

    A3 = canonicalMPS(ComplexF64, χ₁, d * χ₂ ^ 2)
    A2 = canonicalMPS(ComplexF64, χ₁, χ₂ ^ 2)

    function fg!(F, G, x)
        E, (dx,) = withgradient(x) do x
            Q = reshape(x[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂) .+ im * reshape(x[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
            R = reshape(x[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1) .+ im * reshape(x[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)

            h3 = -(reshape(ein"abcd, ik, jl, αγ, βδ -> aiαbjβckγdlδ"(conj.(Q), Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2) .+
            reshape(ein"ac, bd, ik, jl, αβγδ -> aiαbjβckγdlδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d), Q), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2) .+
            ein"ik, jl -> ijkl"(reshape(ein"(ijx, klx), mn -> ikmjln"(conj.(R), M, Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2), Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(reshape(ein"ij, (klx, mnx) -> ikmjln"(Matrix{Float64}(I, χ₂, χ₂), M, R), d * χ₂ ^ 2, d * χ₂ ^ 2), Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2), reshape(ein"(ijx, klx), mn -> ikmjln"(conj.(R), M, Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2), reshape(ein"ij, (klx, mnx) -> ikmjln"(Matrix{Float64}(I, χ₂, χ₂), M, R), d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            reshape(ein"ac, bd, (ikx, jlx), αγ, βδ -> aiαbjβckγdlδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), M, M, Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2))
            E3, A3 = svumps(h3, A3)

            h2 = -(reshape(ein"abcd, αγ, βδ -> aαbβcγdδ"(conj.(Q), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2) .+
            reshape(ein"ac, bd, αβγδ -> aαbβcγdδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), Q), χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2) .+
            ein"ik, jl -> ijkl"(reshape(ein"ijx, klx -> ikjl"(conj.(R), R), χ₂ ^ 2, χ₂ ^ 2), Matrix{Float64}(I, χ₂ ^ 2, χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(Matrix{Float64}(I, χ₂ ^ 2, χ₂ ^ 2), reshape(ein"ijx, klx -> ikjl"(conj.(R), R), χ₂ ^ 2, χ₂ ^ 2)) ./ 2)
            E2, A2 = svumps(h2, A2)
            E3 - E2
        end

        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return E
        end
    end

    res = optimize(Optim.only_fg!((x, y, z) -> fg!(x, y, z)), (x -> hcat(real.(x), imag.(x)))(vcat(vec(Q), vec(R))), LBFGS(manifold = PEPS2()), Optim.Options(f_tol = 1e-8, allow_f_increases = true, show_trace = true))
    println(res)
end

main()