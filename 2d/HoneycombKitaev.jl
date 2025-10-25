using LinearAlgebra
using OMEinsum
using Zygote
using Optim
using SuperVUMPS

const χ₁ = 8
const χ₂ = 2
const d = 4
const D = 2

struct PEPS2 <: Manifold end

function proj!(Ql, Rl, Qr, Rr)
    Ql .= (Ql .+ conj.(ein"ijkl -> klij"(Ql))) ./ 2
    Rl .= (Rl .+ conj.(ein"ijk -> jik"(Rl))) ./ 2 # ?
    Qr .= (Qr .+ conj.(ein"ijkl -> klij"(Qr))) ./ 2
    Rr .= (Rr .+ conj.(ein"ijk -> jik"(Rr))) ./ 2 # ?
end

function Optim.retract!(::PEPS2, qr)
    Ql = reshape(qr[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂)
    Rl = reshape(qr[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1)
    Qr = reshape(qr[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
    Rr = reshape(qr[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)
    proj!(Ql, Rl, Qr, Rr)
    qr .= hcat(vcat(vec(Ql), vec(Rl)), vcat(vec(Qr), vec(Rr)))
end

function Optim.project_tangent!(::PEPS2, dqr, qr)
    Ql = reshape(dqr[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂)
    Rl = reshape(dqr[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1)
    Qr = reshape(dqr[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
    Rr = reshape(dqr[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)
    proj!(Ql, Rl, Qr, Rr)
    dqr .= hcat(vcat(vec(Ql), vec(Rl)), vcat(vec(Qr), vec(Rr)))
end

struct ThreeLegLadder
    M1::Array{ComplexF64, 3}
    M2::Array{ComplexF64, 3}
    M3::Array{ComplexF64, 4}
    Ql::Array{ComplexF64, 4}
    Rl::Array{ComplexF64, 3}
    Qr::Array{ComplexF64, 4}
    Rr::Array{ComplexF64, 3}
end

struct TwoLegLadder
    Q::Array{ComplexF64, 4}
    R::Array{ComplexF64, 3}
end

function SuperVUMPS.local_energy(AL, AC, h::ThreeLegLadder)
    χ₁, = size(AL)
    χ₂, = size(h.Ql)
    l1 = reshape(AL, χ₁, χ₂, :, χ₂, χ₁)
    l2 = ein"ijklm, inklo -> jmno"(conj.(l1), l1)
    l3 = ein"ijklm, ijnlo -> kmno"(conj.(l1), l1)
    l4 = ein"ijklm, ijkno -> lmno"(conj.(l1), l1)
    r1 = reshape(AC, χ₁, χ₂, :, χ₂, χ₁)
    r2 = ein"ijklm, noklm -> ijno"(conj.(r1), r1)
    r3 = ein"ijklm, njolm -> ikno"(conj.(r1), r1)
    r4 = ein"ijklm, njkom -> ilno"(conj.(r1), r1)
    -real(ein"imkn, (ijkl, jmln) -> "(conj.(h.Ql), l2, r2)[] + ein"imkn, (ijkl, jmln) -> "(h.M3, l3, r3)[] + ein"imkn, (ijkl, jmln) -> "(h.Qr, l4, r4)[] +
    ein"jnx, (kox, (ijklm, inolm)) -> "(conj.(h.Rl), h.M1, conj.(r1), r1)[] + ein"knx, (lox, (ijklm, ijnom)) -> "(h.M2, h.Rr, conj.(r1), r1)[])
end

function SuperVUMPS.local_energy(AL, AC, h::TwoLegLadder)
    χ₁, = size(AL)
    χ₂, = size(h.Q)
    l1 = reshape(AL, χ₁, χ₂, χ₂, χ₁)
    l2 = ein"ijkl, imkn -> jlmn"(conj.(l1), l1)
    l3 = ein"ijkl, ijmn -> klmn"(conj.(l1), l1)
    r1 = reshape(AC, χ₁, χ₂, χ₂, χ₁)
    r2 = ein"ijkl, mnkl -> ijmn"(conj.(r1), r1)
    r3 = ein"ijkl, mjnl -> ikmn"(conj.(r1), r1)
    -real(ein"imkn, (ijkl, jmln) -> "(conj.(h.Q), l2, r2)[] + ein"imkn, (ijkl, jmln) -> "(h.Q, l3, r3)[] +
    ein"jmx, (knx, (ijkl, imnl)) -> "(conj.(h.R), h.R, conj.(r1), r1)[])
end

function main()
    Sx = [0.0 0.5; 0.5 0.0]
    Sy = [0.0 -0.5im; 0.5im 0.0]
    Sz = [0.5 0.0; 0.0 -0.5]
    M1 = zeros(4, 4, D - 1)
    M1[:, :, 1] .= reshape(ein"ij, kl -> ikjl"(Sz, Matrix{Float64}(I, 2, 2)), 4, 4)
    M2 = zeros(4, 4, D - 1)
    M2[:, :, 1] .= reshape(ein"ij, kl -> ikjl"(Matrix{Float64}(I, 2, 2), Sz), 4, 4)
    M3 = zeros(ComplexF64, 4, 4, 4, 4)
    M3 .+= ein"ij, kl -> ikjl"(reshape(ein"ij, kl -> ikjl"(Sx, Matrix{Float64}(I, 2, 2)), 4, 4), reshape(ein"ij, kl -> ikjl"(Matrix{Float64}(I, 2, 2), Sx), 4, 4))
    M3 .+= ein"ij, kl -> ikjl"(reshape(ein"ij, kl -> ikjl"(Sy, Sy), 4, 4), Matrix{Float64}(I, 4, 4))

    Ql = randn(ComplexF64, χ₂, χ₂, χ₂, χ₂)
    Rl = randn(ComplexF64, χ₂, χ₂, D - 1)
    Qr = randn(ComplexF64, χ₂, χ₂, χ₂, χ₂)
    Rr = randn(ComplexF64, χ₂, χ₂, D - 1)
    proj!(Ql, Rl, Qr, Rr)

    A3 = canonicalMPS(ComplexF64, χ₁, d * χ₂ ^ 2)
    A2l = canonicalMPS(ComplexF64, χ₁, χ₂ ^ 2)
    A2r = canonicalMPS(ComplexF64, χ₁, χ₂ ^ 2)

    println("========================")
    println("χ = (χ₁, χ₂) = ", (χ₁, χ₂))
    println("========================")

    function fg!(F, G, x)
        E, (dx,) = withgradient(x) do x
            Ql = reshape(x[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂)
            Rl = reshape(x[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1)
            Qr = reshape(x[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
            Rr = reshape(x[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)
            E3, A3 = svumps(ThreeLegLadder(M1, M2, M3, Ql, Rl, Qr, Rr), A3)
            E2l, A2l = svumps(TwoLegLadder(Ql, Rl), A2l)
            E2r, A2r = svumps(TwoLegLadder(Qr, Rr), A2r)
            E3 - (E2l + E2r) / 2
        end
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return E
        end
    end

    res = optimize(Optim.only_fg!((x, y, z) -> fg!(x, y, z)), hcat(vcat(vec(Ql), vec(Rl)), vcat(vec(Qr), vec(Rr))), LBFGS(manifold = PEPS2()), Optim.Options(f_reltol = 1e-8, allow_f_increases = true, show_trace = true))
    x = Optim.minimizer(res)
    Ql .= reshape(x[1 : χ₂ ^ 4, 1], χ₂, χ₂, χ₂, χ₂)
    Rl .= reshape(x[χ₂ ^ 4 + 1 : end, 1], χ₂, χ₂, D - 1)
    Qr .= reshape(x[1 : χ₂ ^ 4, 2], χ₂, χ₂, χ₂, χ₂)
    Rr .= reshape(x[χ₂ ^ 4 + 1 : end, 2], χ₂, χ₂, D - 1)

    println(res)
end

main()