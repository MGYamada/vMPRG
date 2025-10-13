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

function Optim.retract!(mfd::PEPS2, qr)
    Q = reshape(qr[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
    R = reshape(qr[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)
    proj!(Q, R)
    qr .= vcat(vec(Q), vec(R))
end

function Optim.project_tangent!(mfd::PEPS2, dqr, qr)
    Q = reshape(dqr[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
    R = reshape(dqr[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)
    proj!(Q, R)
    dqr .= vcat(vec(Q), vec(R))
end

struct ThreeLegLadder
    M::Array{ComplexF64, 3}
    Q::Array{ComplexF64, 4}
    R::Array{ComplexF64, 3}
end

struct TwoLegLadder
    Q::Array{ComplexF64, 4}
    R::Array{ComplexF64, 3}
end

function SuperVUMPS.local_energy(AL, AC, h::ThreeLegLadder)
    χ₁, = size(AL)
    χ₂, = size(h.Q)
    l1 = reshape(AL, χ₁, χ₂, :, χ₂, χ₁)
    l2 = ein"ijklm, inklo -> jmno"(conj.(l1), l1)
    l3 = ein"ijklm, ijnlo -> kmno"(conj.(l1), l1)
    l4 = ein"ijklm, ijkno -> lmno"(conj.(l1), l1)
    r1 = reshape(AC, χ₁, χ₂, :, χ₂, χ₁)
    r2 = ein"ijklm, noklm -> ijno"(conj.(r1), r1)
    r3 = ein"ijklm, njolm -> ikno"(conj.(r1), r1)
    r4 = ein"ijklm, njkom -> ilno"(conj.(r1), r1)
    -(ein"imkn, (ijkl, jmln) -> "(conj.(h.Q), l2, r2)[] + ein"ikx, (mnx, (ijkl, jmln)) -> "(h.M, h.M, l3, r3)[] + ein"imkn, (ijkl, jmln) -> "(h.Q, l4, r4)[] +
    ein"jnx, (kox, (ijklm, inolm)) -> "(conj.(h.R), h.M, conj.(r1), r1)[] + ein"knx, (lox, (ijklm, ijnom)) -> "(h.M, h.R, conj.(r1), r1)[])
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
    -(ein"imkn, (ijkl, jmln) -> "(conj.(h.Q), l2, r2)[] + ein"imkn, (ijkl, jmln) -> "(h.Q, l3, r3)[] +
    ein"jmx, (knx, (ijkl, imnl)) -> "(conj.(h.R), h.R, conj.(r1), r1)[])
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

    A3 = canonicalMPS(ComplexF64, χ₁, d * χ₂ ^ 2)
    A2 = canonicalMPS(ComplexF64, χ₁, χ₂ ^ 2)

    println("========================")
    println("χ = (χ₁, χ₂) = ", (χ₁, χ₂))
    println("========================")

    function fg!(F, G, x)
        E, (dx,) = withgradient(x) do x
            Q0 = reshape(x[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂) 
            R0 = reshape(x[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)
            E3, A3 = svumps(ThreeLegLadder(M, Q0, R0), A3)
            E2, A2 = svumps(TwoLegLadder(Q0, R0), A2)
            E3 - E2
        end
        if G !== nothing
            G .= dx
        end
        if F !== nothing
            return E
        end
    end

    res = optimize(Optim.only_fg!((x, y, z) -> fg!(x, y, z)), vcat(vec(Q), vec(R)), LBFGS(manifold = PEPS2()), Optim.Options(f_reltol = 1e-8, allow_f_increases = true, show_trace = true))
    x = Optim.minimizer(res)
    Q .= reshape(x[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
    R .= reshape(x[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)

    println(res)
end

main()
