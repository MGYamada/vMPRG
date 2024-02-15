const version = v"0.1.0"

using LinearAlgebra
using OMEinsum
using Zygote
using ForwardDiff
using Optim
using SuperVUMPS

Base.Float64(x::ForwardDiff.Dual) = x.value # is it correct?

const χ₁s = [8, 16, 24, 32]
const χ₂ = 2
const d = 2
const D = 4
const β = 1.0

function f(AL0, QR, M)
    Q = reshape(QR[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
    R = reshape(QR[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)

    AL = Complex.(AL0[:, :, :, 1], AL0[:, :, :, 2])
    i, j, = size(AL)
    L1 = reshape(AL, i * j, i)
    L = L1 * (3 / 2 * I - 1 / 2 * (L1' * L1))
    AL = reshape(L, i, j, i)
    h3 = -(reshape(ein"abcd, ik, jl, αγ, βδ -> aiαbjβckγdlδ"(Q, Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2) .+
    reshape(ein"ac, bd, ik, jl, αβγδ -> aiαbjβckγdlδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d), Q), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2) .+
    ein"ik, jl -> ijkl"(reshape(ein"(ijx, klx), mn -> ikmjln"(R, M, Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2), Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
    ein"ik, jl -> ijkl"(reshape(ein"ij, (klx, mnx) -> ikmjln"(Matrix{Float64}(I, χ₂, χ₂), M, R), d * χ₂ ^ 2, d * χ₂ ^ 2), Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
    ein"ik, jl -> ijkl"(Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2), reshape(ein"(ijx, klx), mn -> ikmjln"(R, M, Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
    ein"ik, jl -> ijkl"(Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2), reshape(ein"ij, (klx, mnx) -> ikmjln"(Matrix{Float64}(I, χ₂, χ₂), M, R), d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
    reshape(ein"ac, bd, (ikx, jlx), αγ, βδ -> aiαbjβckγdlδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), M, M, Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2))

    A = reshape(ein"ijk, ljm -> ilkm"(AL, conj.(AL)), i ^ 2, i ^ 2)
    v = randn(ComplexF64, i ^ 2)
    v /= norm(v)
    for i in 1 : 100 # fix later
        v = A * v
        v /= norm(v)
    end
    x = reshape(v, i, i)
    real(ein"ijk, (klq, (jlno, (inp, (pom, mq)))) -> "(conj.(AL), conj.(AL), h3, AL, AL, x)[] / tr(x)) + β / 4 * norm(L1' * L1 - I) ^ 2
end

function g(AL0, QR)
    Q = reshape(QR[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
    R = reshape(QR[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)

    AL = Complex.(AL0[:, :, :, 1], AL0[:, :, :, 2])
    i, j, = size(AL)
    L1 = reshape(AL, i * j, i)
    L = L1 * (3 / 2 * I - 1 / 2 * (L1' * L1))
    AL = reshape(L, i, j, i)
    h2 = -(reshape(ein"abcd, αγ, βδ -> aαbβcγdδ"(Q, Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2) .+
    reshape(ein"ac, bd, αβγδ -> aαbβcγdδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), Q), χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2) .+
    ein"ik, jl -> ijkl"(reshape(ein"ijx, klx -> ikjl"(R, R), χ₂ ^ 2, χ₂ ^ 2), Matrix{Float64}(I, χ₂ ^ 2, χ₂ ^ 2)) ./ 2 .+
    ein"ik, jl -> ijkl"(Matrix{Float64}(I, χ₂ ^ 2, χ₂ ^ 2), reshape(ein"ijx, klx -> ikjl"(R, R), χ₂ ^ 2, χ₂ ^ 2)) ./ 2)

    A = reshape(ein"ijk, ljm -> ilkm"(AL, conj.(AL)), i ^ 2, i ^ 2)
    v = randn(ComplexF64, i ^ 2)
    v /= norm(v)
    for i in 1 : 100 # fix later
        v = A * v
        v /= norm(v)
    end
    x = reshape(v, i, i)
    real(ein"ijk, (klq, (jlno, (inp, (pom, mq)))) -> "(conj.(AL), conj.(AL), h2, AL, AL, x)[] / tr(x)) + β / 4 * norm(L1' * L1 - I) ^ 2
end

function main()
    Sx = [0.0 0.5; 0.5 0.0]
    Sy = [0.0 0.5; -0.5 0.0]
    Sz = [0.5 0.0; 0.0 -0.5]
    M = zeros(2, 2, D - 1)
    M[:, :, 1] .= Sx
    M[:, :, 2] .= Sy
    M[:, :, 3] .= Sz

    Q = randn(χ₂, χ₂, χ₂, χ₂)
    Q .= (Q .+ ein"ijkl -> klij"(Q)) ./ 2
    R = randn(χ₂, χ₂, D - 1)
    R .= (R .+ ein"ijk, k -> jik"(R, [1.0, -1.0, 1.0])) ./ 2

    for χ₁ in χ₁s
        println("========================")
        println("χ = (χ₁, χ₂) = ", (χ₁, χ₂))
        println("========================")

        A3 = canonicalMPS(ComplexF64, χ₁, d * χ₂ ^ 2)
        A2 = canonicalMPS(ComplexF64, χ₁, χ₂ ^ 2)

        function fg!(F, G, x)
            Q = reshape(x[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
            R = reshape(x[χ₂ ^ 4 + 1 : end], χ₂, χ₂, D - 1)
            Q .= (Q .+ ein"ijkl -> klij"(Q)) ./ 2
            R .= (R .+ ein"ijk, k -> jik"(R, [1.0, -1.0, 1.0])) ./ 2

            h3 = -(reshape(ein"abcd, ik, jl, αγ, βδ -> aiαbjβckγdlδ"(Q, Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2) .+
            reshape(ein"ac, bd, ik, jl, αβγδ -> aiαbjβckγdlδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, d, d), Matrix{Float64}(I, d, d), Q), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2) .+
            ein"ik, jl -> ijkl"(reshape(ein"(ijx, klx), mn -> ikmjln"(R, M, Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2), Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(reshape(ein"ij, (klx, mnx) -> ikmjln"(Matrix{Float64}(I, χ₂, χ₂), M, R), d * χ₂ ^ 2, d * χ₂ ^ 2), Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2), reshape(ein"(ijx, klx), mn -> ikmjln"(R, M, Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(Matrix{Float64}(I, d * χ₂ ^ 2, d * χ₂ ^ 2), reshape(ein"ij, (klx, mnx) -> ikmjln"(Matrix{Float64}(I, χ₂, χ₂), M, R), d * χ₂ ^ 2, d * χ₂ ^ 2)) ./ 2 .+
            reshape(ein"ac, bd, (ikx, jlx), αγ, βδ -> aiαbjβckγdlδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), M, M, Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2, d * χ₂ ^ 2))

            E3, A3 = svumps(h3, A3)

            h2 = -(reshape(ein"abcd, αγ, βδ -> aαbβcγdδ"(Q, Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂)), χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2) .+
            reshape(ein"ac, bd, αβγδ -> aαbβcγdδ"(Matrix{Float64}(I, χ₂, χ₂), Matrix{Float64}(I, χ₂, χ₂), Q), χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2, χ₂ ^ 2) .+
            ein"ik, jl -> ijkl"(reshape(ein"ijx, klx -> ikjl"(R, R), χ₂ ^ 2, χ₂ ^ 2), Matrix{Float64}(I, χ₂ ^ 2, χ₂ ^ 2)) ./ 2 .+
            ein"ik, jl -> ijkl"(Matrix{Float64}(I, χ₂ ^ 2, χ₂ ^ 2), reshape(ein"ijx, klx -> ikjl"(R, R), χ₂ ^ 2, χ₂ ^ 2)) ./ 2)

            E2, A2 = svumps(h2, A2)
            val = E3 - E2

            if G !== nothing
                QR = vcat(vec(Q), vec(R))
                AL3 = cat(real.(A3.AL), imag.(A3.AL); dims = 4)
                AL2 = cat(real.(A2.AL), imag.(A2.AL); dims = 4)
                G .= gradient(z -> f(AL3, z, M) - g(AL2, z), QR)[1]
                df = gradient(x -> f(x, QR, M), AL3)[1]
                Hf = hessian(x -> f(x, QR, M), AL3)
                zx = Zygote.forward_jacobian(z -> gradient(x -> f(x, z, M), AL3)[1], QR)[2]
                G .-= zx * (Hf \ vec(df))
                dg = gradient(x -> -g(x, QR), AL2)[1]
                Hg = hessian(x -> g(x, QR), AL2)
                zy = Zygote.forward_jacobian(z -> gradient(y -> g(y, z), AL2)[1], QR)[2]
                G .-= zy * (Hg \ vec(dg))
            end
            if F !== nothing
                return val
            end
        end

        res = optimize(Optim.only_fg!((x, y, z) -> fg!(x, y, z)), vcat(vec(Q), vec(R)), LBFGS(), Optim.Options(f_tol = 1e-8, allow_f_increases = true, show_trace = true))
        x = Optim.minimizer(res)
        Q .= reshape(x[1 : χ₂ ^ 4], χ₂, χ₂, χ₂, χ₂)
        R .= reshape(x[χ₂ ^ 4 + 1 : χ₂ ^ 4 + χ₂ ^ 2 * (D - 1)], χ₂, χ₂, D - 1)

        # EGS = Ebulk(x)
        println("========================")
        println(res)
        # println("========================")
        # println("GS: ", EGS)
        println("========================")

        # vals1, vecs1 = eigen(Hermitian(reshape(H1, d * χ₁ ^ 2 * χ₂ ^ 2, d * χ₁ ^ 2 * χ₂ ^ 2)))
        # vals2, vecs2 = eigen(Hermitian(reshape(H2, χ₁ ^ 2, χ₁ ^ 2)))
        # vals3, vecs3 = eigen(Hermitian(reshape(H3, χ₁ ^ 2 * χ₂ ^ 2, χ₁ ^ 2 * χ₂ ^ 2)))
        # vals4, vecs4 = eigen(Hermitian(reshape(H4, χ₁ ^ 2, χ₁ ^ 2)))
        # for β in 10 .^ (-2.0 : 0.01 : 2.0)
        #     λ1 = sum(exp.(β .* (vals1[1] .- vals1)))
        #     λ2 = sum(exp.(β .* (vals2[1] .- vals2)))
        #     λ3 = sum(exp.(β .* (vals3[1] .- vals3)))
        #     λ4 = sum(exp.(β .* (vals4[1] .- vals4)))
        #     lnZ = real(log(λ1) - log(λ2) - log(λ3) + log(λ4) - β * (vals1[1] - vals2[1] - vals3[1] + vals4[1]))
        #     println(β, " ", lnZ)
        # end
    end
end

main()
