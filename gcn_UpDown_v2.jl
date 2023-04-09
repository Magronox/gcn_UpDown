using LinearAlgebra, SparseArrays, Random


function gcn(A::SparseMatrixCSC, X::Matrix, W:: Matrix)

    d = sum(A, dims = 2)
    d = Diagonal(vec(d))
    tild_A = d^(-1/2)*A*d^(-1/2)
    W*X*tild_A
end

function Relu(X::Union{Matrix,Vector})
    Int.(X.>0).*X
end

function LSM(v :: Vector{Float64})
    log.(exp.(v)/sum(exp.(v)))
end

function overall_function(A::SparseMatrixCSC, X::Matrix, Ws)
    W1, W2 = Ws
    n = size(A,2)
    output = zeros(size(W2,1),n)
    for i in 1:n
        output[:,i] = LSM(gcn(A,Relu(gcn(A,X,W1)),W2)[:,i])
    end
    output
end

function nll(Z_opt,Z)
    sum(-Z[Z_opt])
end

@inline function Mmat(A::SparseMatrixCSC, H::Matrix, idx:: Int64, k::Int64)
    degs = sum(A, dims = 2)
    M = I(k)*sum(H[1,rowvals(A)[nzrange(A,idx)]]./sqrt.(degs[idx]*degs[rowvals(A)[nzrange(A,idx)]]))
    for j in 2:size(H,1)
        M = hcat(M,I(k)*sum(H[j,rowvals(A)[nzrange(A,idx)]]./sqrt.(degs[idx]*degs[rowvals(A)[nzrange(A,idx)]])) )
    end
    for j in M
        if isnan(j)
            print("here",M,"\n")
        end
    end
    M
end

function gd_W2(Z_opt, A::SparseMatrixCSC, X::Matrix, Ws, α)
    W1, W2 = Ws
    Z = overall_function(A,X,Ws)
    H2 = Relu(gcn(A,X,W1))
    n = size(A,1)
    l = size(W1,1)
    k = size(W2,1)
    temp = zeros(1,k*l)
    for i in 1:n
        temp += -Z_opt[:,i]'*(I(k) - ones(k)*exp.(Z[:,i]'))*Mmat(A,H2,i,k)

    end
    W2[:] -= α*temp[:]
    #for j in W2
    #    if(isnan(j))
    #        print(temp,(I(k) - ones(k)*exp.(Z[:,n]')), Mmat(A,H2,n,k))
    #    end
    #end
    return W2
end

function gd_W1(Z_opt, A::SparseMatrixCSC, X::Matrix, Ws, α)
    W1, W2 = Ws
    H2 = gcn(A,X,W1)
    n = size(A,1)
    l = size(W1,1)
    k = size(W2,1)
    temp = zeros(1,l*size(X,1))
    degs = sum(A, dims = 2)
    for i in 1:n
        temp2 = zeros(k,size(X,1)*l)
        for j in rowvals(A)[nzrange(A,i)]
            temp2 += W2 * Diagonal(vec(H2[:,j].>0))* Mmat(A,X,j,l)/(sqrt(degs[i]*degs[j]))
        end
        
        temp += -Z_opt[:,i]'*(I(k) - ones(k)*exp.(Z[:,i]'))*temp2
    end
    
    W1[:] -= α*temp[:]
    return W1
end
    


