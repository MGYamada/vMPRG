const version = v"0.1.0"

using LinearAlgebra
using OMEinsum
using KrylovKit
using Handagote

const χ = 16
const D = 2
const d = 2
const Γ = 1.0
const J = 2.0

safesign(x::Number) = iszero(x) ? one(x) : sign(x)

qrpos(A) = qrpos!(copy(A))
function qrpos!(A)
    F = qr!(A)
    Q = Matrix(F.Q)
    R = F.R
    phases = safesign.(diag(R))
    rmul!(Q, Diagonal(phases))
    lmul!(Diagonal(conj!(phases)), R)
    Q, R
end

lqpos(A) = lqpos!(copy(A))
function lqpos!(A)
    F = qr!(Matrix(A'))
    Q = Matrix(Matrix(F.Q)')
    L = Matrix(F.R')
    phases = safesign.(diag(L))
    lmul!(Diagonal(phases), Q)
    rmul!(L, Diagonal(conj!(phases)))
    L, Q
end

function fixedpointlinear(f, g, args...; kwargs...)
    λs, Gs, = eigsolve(x -> f(x, args...), g, 1, :LR; ishermitian = false, maxiter = 1, kwargs...)
    normalize(Gs[1])
end

function leftorth(A, C = Matrix{eltype(A)}(I, size(A, 1), size(A, 1)); tol = 1e-12, maxiter = 100, kwargs...)
    ρ = fixedpointlinear(C' * C; tol = tol, kwargs...) do ρ
        ein"(ij, ikl), jkm -> lm"(ρ, conj.(A), A)
    end
    ρ += ρ'
    ρ /= tr(ρ)
    U, S, V = svd(ρ)
    C = Diagonal(sqrt.(S)) * V'
    _, C = qrpos(C)

    D, d, = size(A)
    Q, R = qrpos(reshape(C * reshape(A, D, d * D), D * d, D))
    AL = reshape(Q, D, d, D)
    λ = norm(R)
    R /= λ
    numiter = 1
    while norm(C .- R) > tol && numiter < maxiter
        C = fixedpointlinear(R; tol = tol, kwargs...) do X
            ein"(ij, ikl), jkm -> lm"(X, conj.(AL), A)
        end
        _, C = qrpos(C)
        Q, R = qrpos(reshape(C * reshape(A, D, d * D), D * d, D))
        AL = reshape(Q, D, d, D)
        λ = norm(R)
        R /= λ
        numiter += 1
    end
    C = R
    AL, C, λ
end

function rightorth(A, C = Matrix{eltype(A)}(I, size(A,1), size(A,1)); tol = 1e-12, kwargs...)
    AL, C, λ = leftorth(ein"ijk -> kji"(A), transpose(C); tol = tol, kwargs...)
    transpose(C), ein"ijk -> kji"(AL), λ
end

Base.:*(A::DualVector, B::Adjoint{<: Number, <:DualVector}) = dualein"i, j -> ij"(A, conj(parent(B)))

function power(A::Matrix, x0)
    # dummy function
    @assert dot(x0, x0) ≈ 1
    copy(x0)
end

function power(A::DualMatrix, x0)
    x = power(realpart(A), realpart(x0))
    Ax = realpart(A) * x
    normAx = sqrt(dot(Ax, Ax))
    B = (I - x * x') * realpart(A) / normAx
    dx = _linsolve(I - B, (I - x * x') * (εpart(A) * x / normAx))
    DualArray(x, dx)
end

function power(A::HyperDualMatrix, x0)
    x = power(realpart(A), realpart(x0))
    Ax = realpart(A) * x
    normAx = sqrt(dot(Ax, Ax))
    B = (I - x * x') * realpart(A) / normAx
    dx = _linsolve(I - B, (I - x * x') * (εpart(A) * x / normAx))
    HyperDualArray(x, dx)
end

function _linsolve(A, b::Vector)
    x, = linsolve(A, b; ishermitian = false)
    x
end

function _linsolve(A, b::DualVector)
    x = _linsolve(realpart(A), realpart(b))
    dx = _linsolve(realpart(A), εpart(b) - εpart(A) * x)
    DualArray(x, dx)
end

function f!(M::HyperDualArray{T, 4}, A, FL, FR) where T
    AL, = leftorth(A)
    C, AR = rightorth(AL)
    AC = ein"ijk, kl -> ijl"(AL, C)

    val = 0.0
    valold = val
    while true
        L = reshape(hyperdualein"ilm, ljnkp -> ijkmnp"(conj(AL), hyperdualein"lojn, kop -> ljnkp"(M, AL)), D * χ ^ 2, D * χ ^ 2)
        FL .= reshape(power(hyperdualein"ij -> ji"(L), vec(FL)), χ, D, χ)
        R = reshape(hyperdualein"ilm, ljnkp -> ijkmnp"(conj(AR), hyperdualein"lojn, kop -> ljnkp"(M, AR)), D * χ ^ 2, D * χ ^ 2)
        FR .= reshape(power(R, vec(FR)), χ, D, χ)

        F = χ * reshape(hyperdualein"ijk, mnjop -> imoknp"(FL, hyperdualein"mnjl, olp -> mnjop"(M, FR)), d * χ ^ 2, d * χ ^ 2)
        vals, vecs = eigsolve(ɛ₁ε₂part(F), 1, :LR)
        val1 = vals[1]
        AC = reshape(vecs[1], χ, d, χ)
        G = χ * reshape(hyperdualein"ijk, ojp -> iokp"(FL, FR), χ ^ 2, χ ^ 2)
        vals, vecs = eigsolve(ɛ₁ε₂part(G), 1, :LR)
        val2 = vals[1]
        val = real(val1 - val2)
        if abs((val - valold) / val) < 1e-10
            break
        end
        valold = val
        C = reshape(vecs[1], χ, χ)

        U, S, V = svd(reshape(AC, :, χ) * C')
        AL = reshape(U * V', χ, d, χ)
        U, S, V = svd(C' * reshape(AC, χ, :))
        AR = reshape(U * V', χ, d, χ)
    end
    val, FL, FR / hyperdualein"io, io -> "(conj(C), hyperdualein"ijk, ojk -> io"(FL, hyperdualein"ojp, kp -> ojk"(FR, C)))[]
end

function main()
    A0 = [1.0 ; 0.0 ;; 0.0 ; 1.0 ;;; 0.0 ; 0.0 ;; 0.0 ; 0.0 ;;;; 0.0 ; 0.0 ;; 0.0 ; 0.0 ;;; 0.0 ; 0.0 ;; 0.0 ; 0.0]
    A1 = [0.0 ; 0.0 ;; 0.0 ; 0.0 ;;; 0.5 ; 0.0 ;; 0.0 ; -0.5 ;;;; 0.5 ; 0.0 ;; 0.0 ; -0.5 ;;; 0.0 ; 0.0 ;; 0.0 ; 0.0]
    A2 = [0.0 ; 0.5 ;; 0.5 ; 0.0 ;;; 0.0 ; 0.0 ;; 0.0 ; 0.0 ;;;; 0.0 ; 0.0 ;; 0.0 ; 0.0 ;;; 0.0 ; 0.0 ;; 0.0 ; 0.0]
    A = HyperDualArray(A0, A1 .* sqrt(J / 2), A1 .* sqrt(J / 2), Γ .* A2)

    temp = randn(ComplexF64, χ, D, χ)
    FL = HyperDualArray(zeros(ComplexF64, χ, D, χ), temp, copy(temp), randn(ComplexF64, χ, D, χ))
    hyperrealpart(FL)[:, 1, :] .+= Matrix{Float64}(I, χ, χ) ./ sqrt(χ)
    temp = randn(ComplexF64, χ, D, χ)
    FR = HyperDualArray(zeros(ComplexF64, χ, D, χ), temp, copy(temp), randn(ComplexF64, χ, D, χ))
    hyperrealpart(FR)[:, 1, :] .+= Matrix{Float64}(I, χ, χ) ./ sqrt(χ)

    λ, E1, E2 = f!(A, randn(χ, d, χ), FL, FR)
    println("GS: ", -λ)

    EAE = reshape(hyperdualein"ilm, jnlkp -> ijkmnp"(E1, hyperdualein"jnlo, kop -> jnlkp"(A, E2)), d * χ ^ 2, d * χ ^ 2)
    vals1, vecs1 = eigen(ɛ₁ε₂part(EAE))
    EE = reshape(hyperdualein"ikl, jkm -> ijlm"(E1, E2), χ ^ 2, χ ^ 2)
    vals2, vecs2 = eigen(ɛ₁ε₂part(EE))
    for β in 10 .^ (-2.0 : 0.01 : 2.0)
        λ1 = sum(exp.(β .* (vals1 .- vals1[end])))
        λ2 = sum(exp.(β .* (vals2 .- vals2[end])))
        lnZ = real(β * (vals1[end] - vals2[end]) + log(λ1) - log(λ2))
        println(β, " ", lnZ)
    end
end

main()