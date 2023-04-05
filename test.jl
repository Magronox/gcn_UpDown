include("gcn_UpDown")
##### test:

n = 10
k = 10
m = 6
X = randn(n,k)
W1 = randn(k,n)
W2 = randn(n,m)

A = sprand(Bool,n,n,0.6)
A = triu(A)
A = A + A'

## Adding self-loop
A[diagind(A)] .= 1



Z_opt = zeros(Bool,n,m)
ind = map(1:n) do i
    x1 = i
    y1 = rand(1:m)

    CartesianIndex(x1, y1)
end
Z_opt[ind] .= 1
W1=randn(10,5)
W2 = randn(5,6)
Z = overall_function(A,X,W1,W2)
l1 = nll_loss(Z_opt,Z)

niter = 30000
idx = 1
for i in 1:niter
    W1 = scd_w1!(Z_opt,A,X,W1,W2,0.02)
    W2 = scd_w2!(Z_opt,A,X,W1,W2,0.02)
end
Z = overall_function(A,X,W1,W2)
l2 = nll_loss(Z_opt,Z)
result = zeros(Bool,size(Z))
for i in 1:size(Z,1)
    result[i,findmax(Z[i,:])[2]] = 1
end

ncorrect = sum(result[Z_opt])
correct_ratio = ncorrect/n
