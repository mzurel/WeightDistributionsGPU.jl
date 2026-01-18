module GF4WeightsGPU

using StaticArrays
using CUDA
using Nemo

export partialweightdistribution, weightdistribution

const q = 4
const F, a = finite_field(q, "a")
const MAXN = 128


## ---------------------------------- ##
## Encode GF(4) elements as 0x00:0x03 ##
## ---------------------------------- ##

const fqtou8 = Dict{FqFieldElem,UInt8}(
    e => UInt8(i-1) for (i,e) ∈ enumerate(collect(F))
)
const u8tofq = Dict{UInt8,FqFieldElem}(
    fqtou8[k] => k for k ∈ keys(fqtou8)
)

"""
    encodematrixu8(M::FqMatrix)

    Take a Nemo matrix of GF(4) elements and encode it as a UInt8 matrix.
"""
function encodematrixu8(G::FqMatrix)::Matrix{UInt8}
    k, n = size(G)
    M = Matrix{UInt8}(undef, k, n)
    for i ∈ 1:k, j ∈ 1:n
        M[i,j] = fqtou8[G[i,j]]
    end
    return M
end

function buildgfqtblsu8()::Tuple{Matrix{UInt8},Matrix{UInt8}}
    addtbl = Matrix{UInt8}(undef, q, q)
    multbl = Matrix{UInt8}(undef, q, q)
    @inbounds for i ∈ 1:q, j ∈ 1:q
        x = u8tofq[UInt8(i-1)]
        y = u8tofq[UInt8(j-1)]
        addtbl[i,j] = fqtou8[x + y]
        multbl[i,j] = fqtou8[x * y]
    end
    return addtbl, multbl
end

@inline function gf9add(x::UInt8, y::UInt8, addtbl)::UInt8
    return @inbounds addtbl[Int(x)+1, Int(y)+1]
end

@inline function gf9mul(x::UInt8, y::UInt8, multbl)::UInt8
    return @inbounds multbl[Int(x)+1, Int(y)+1]
end


## --------------------- ##
## Binomial coefficients ##
## --------------------- ##

"""
    buildbintbl(n)

    Builds a table of binomial coefficients of size (n+1)x(n+1) (0 to n).
"""
function buildbintbl(n)::Matrix{UInt64}
    bintbl = Matrix{UInt64}(undef, n+1, n+1)
    @inbounds for i ∈ 0:n, j ∈ 0:n
        bintbl[i+1,j+1] = binomial(i,j)
    end
    return bintbl
end


## ----- ##
## Tools ##
## ----- ##

function unrankcombinations(n, k, r)
    mask = 0
    rem = k
    for pos = n-1:-1:0
        if rem == 0
            break
        end
        c = binomial(pos, rem)
        if r ≥ c
            mask |= 1 << pos
            r -= c
            rem -= 1
        end
    end
    return mask
end

"""
    complementarygenerators(G::FqMatrix)

    Given a generator matrix G of a self-dual linear code, returns two generator matrices
    G1 and G2 with the structure G₁=[I|A₁] and G₂=[A₂|I]. Assumes the code length is twice
    the dimension, and that the first and last k rows each form an information set.
"""
function complementarygenerators(G::FqMatrix)::Tuple{FqMatrix,FqMatrix}
    k, n = size(G)
    @assert n % 2 == 0 "Code length is not even"
    @assert k == div(n, 2) "Code is not self-dual, dimension too small"
    @assert rank(G) == k "Code is not self-dual, dimension too small"
    @assert rank(G[1:end,1:k]) == k "First k columns of generator matrix are not an information set"
    G1 = rref(G)[2]
    G2 = hcat(G1[:, k+1:end], G1[:, 1:k])
    G2 = rref(G2)[2]
    G2 = hcat(G2[:, k+1:end], G2[:, 1:k])
    return G1, G2
end

@inline function fmttime(secs::Real)
    s = max(0.0, float(secs))
    h = floor(Int, s / 3600)
    s -= 3600h
    m = floor(Int, s / 60)
    s -= 60m
    if h > 0
        return string(h, "h ", lpad(m, 2, '0'), "m ", lpad(round(Int, s), 2, '0'), "s")
    elseif m > 0
        return string(m, "m ", lpad(round(Int, s), 2, '0'), "s")
    else
        return string(round(Int, s), "s")
    end
end


## --------------------------------------------------------- ##
## GPU kernel for computing weights of codewords in parallel ##
## --------------------------------------------------------- ##

function partialweightskernel(Gt::CuDeviceMatrix{UInt8},
                              addtbl::CuDeviceMatrix{UInt8},
                              multbl::CuDeviceMatrix{UInt8},
                              bintbl::CuDeviceMatrix{UInt64},
                              hist::CuDeviceVector{UInt64},
                              k::Int, n::Int,
                              t::Int, ttotal::Int,
                              batchstart::UInt64, batchcount::UInt64,
                              strict::Bool)
    tid    = (blockIdx().x - 1) * blockDim().x + threadIdx().x
    stride = gridDim().x * blockDim().x

    # start and end indices for this batch
    idx   = batchstart + UInt64(tid - 1)
    final = batchstart + batchcount #UInt64(ttotal)
    step  = UInt64(stride)

    cw = MVector{MAXN,UInt8}(undef)
    @inbounds for j ∈ 1:MAXN
        cw[j] = UInt8(0)
    end

    while idx < final
        @inbounds for j ∈ 1:n
            cw[j] = UInt8(0)
        end
        
        comb = div(idx, UInt64(q-1)^(t-1))
        coeffs = (idx % UInt64(q-1)^(t-1)) * UInt64(q-1)

        rem = t
        @inbounds for pos ∈ k-1:-1:0  # run unrank algorithm to fill vector with coefficients
            if rem == 0
                break
            end
            c = bintbl[pos+1,rem+1]
            if comb ≥ c
                i = pos + 1
                coeff = UInt8(coeffs % 0x03) + 0x01
                for j ∈ 1:k
                    prod = gf9mul(coeff, Gt[j,i], multbl)
                    old = cw[j]
                    new = gf9add(old, prod, addtbl)
                    cw[j] = new
                end

                comb -= c
                rem -= 1
                coeffs = coeffs ÷ UInt64(q-1)
            end
        end

        w = t
        @inbounds for col ∈ 1:k
            if cw[col] != 0
                w += 1
            end
        end

        if (!strict && w ≥ 2t) || (strict && w > 2t)
            @inbounds CUDA.@atomic hist[w+1] += UInt64(q-1)
        end

        idx += step
    end

    return
end


## -------------- ##
## Main functions ##
## -------------- ##

"""
    partialweightdistribution(G::FqMatrix, ub::Int64)

    G is the generator matrix of a self-dual linear code over GF(4). This function returns
    a Dict with the counts of the numbers of codewords of a given weight for weights 0 up to
    ub computed using the algorithm of Gaborit-Nedeloaia-Wassermann
    [https://doi.org/10.1109/ISIT.2004.1365525].
"""
function partialweightdistribution(G::FqMatrix, ub::Int, batchsize::Int=10_000_000_000)
    k, n = size(G)
    ub = min(n, 2k, ub)
    G1, G2 = complementarygenerators(G)

    A1u8 = encodematrixu8(G1[1:end,(k+1):end])
    A2u8 = encodematrixu8(G2[1:end,1:k])

    addtbl, multbl = buildgfqtblsu8()
    bintbl = buildbintbl(div(MAXN, 2))

    dA1tu8 = CuMatrix(permutedims(A1u8))
    dA2tu8 = CuMatrix(permutedims(A2u8))
    daddtbl = CuMatrix(addtbl)
    dmultbl = CuMatrix(multbl)
    dbintbl = CuMatrix(bintbl)
    dhist = CuVector(zeros(UInt64, n+1))

    total = 2 * sum(binomial(k,t)*(q-1)^t for t ∈ 1:fld(ub,2))
    threads = 256

    for t ∈ 1:fld(ub, 2)
        ttotal = binomial(k, t) * (q-1)^(t-1)
        ttotalcodewords = 2 * binomial(k, t) * (q-1)^t

        println("t=$t start: idx-per-half=$ttotal, codewords-this-t=$ttotalcodewords, batchsize=$batchsize")
        tstart = time()
        processedidx = UInt64(0)

        batchstart = UInt64(0)
        while batchstart < UInt64(ttotal)
            batchcount = UInt64(min(batchsize, Int(UInt64(ttotal) - batchstart)))

            blocks = Int(min(UInt64(65535), cld(batchcount, UInt64(threads))))

            # First half (non-strict)
            @cuda threads=threads blocks=blocks partialweightskernel(
                dA1tu8, daddtbl, dmultbl, dbintbl, dhist,
                k, n, t, ttotal,
                batchstart, batchcount,
                false
            )
            synchronize()

            # Second half (strict)
            @cuda threads=threads blocks=blocks partialweightskernel(
                dA2tu8, daddtbl, dmultbl, dbintbl, dhist,
                k, n, t, ttotal,
                batchstart, batchcount,
                true
            )
            synchronize()

            processedidx += batchcount
            elapsed = time() - tstart
            frac = processedidx / UInt64(ttotal)
            eta = (frac > 0) ? elapsed * (1 / frac - 1) : Inf

            tcheckedcodewords = 2 * (q-1) * processedidx
            println("t=$t progress: $(round(100*frac; digits=1))%  idx=$(processedidx)/$(ttotal)  codewords=$(tcheckedcodewords)/$(ttotalcodewords)  elapsed=$(fmttime(elapsed))  eta=$(fmttime(eta))")

            batchstart += batchcount
        end
        completed = 2 * sum(binomial(k,l)*(q-1)^l for l ∈ 1:t)
        println("t=$t done. Overall checked: $completed / $total codewords ($(round(100 * completed / total))%)")
    end

    hist = Vector{UInt64}(undef, n+1)
    copyto!(hist, dhist)
    hist[1] = 1
    return hist
end

end  # module