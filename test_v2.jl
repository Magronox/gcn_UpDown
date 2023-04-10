include("gcn_UpDown_v2.jl")

##### test
n = 5
m = 4
l = 4
k = 6
A = sprand(Bool,n,n,0.6)
A = triu(A)
A = A + A'
## Adding self-loop
A[diagind(A)] .= 1
X = rand(m,n)


Z_opt = zeros(Bool,k,n)
ind = map(1:n) do i
    x1 = rand(1:k)
    y1 = i
    CartesianIndex(x1, y1)
end
Z_opt[ind] .= 1


W1 = rand(l,m)
W2 = rand(k,l)
Z = overall_function(A,X,(W1,W2))

l1 = nll(Z_opt,Z)
W2 = gd_W2(Z_opt,A,X,(W1,W2),0.05)
W1 = gd_W1(Z_opt,A,X,(W1,W2),0.05)
Z = overall_function(A,X,(W1,W2))
l2 = nll(Z_opt,Z)

niter = 10000
for i in 1:niter
    W2 = gd_W2(Z_opt,A,X,(W1,W2),0.05)
    W1 = gd_W1(Z_opt,A,X,(W1,W2),0.05)
end
    
Z = overall_function(A,X,(W1,W2))
l2 = nll(Z_opt,Z)

result = zeros(Bool,size(Z))
for i in 1:size(Z,2)
    result[findmax(Z[:,i])[2],i] = 1
end

ncorrect = sum(result[Z_opt])
correct_ratio = ncorrect/n


#### finite difference test
#Mmat
W1 = rand(l,m)
H = gcn(A,X,W1)
dw = ones(size(W1))* norm(W1)/1000
W11 = W1 +dw
H11 = gcn(A,X,W11)
res = H11 - H
M = Mmat(A,X,1,4)
res11 = M*(dw[:])


#Pmat \partial H \slash \partial X
W1 = rand(l,m)
H = gcn(A,X,W1)
dX = ones(size(X,1))* norm(X)/1000
X11 = copy(X)
X11[:,1] += dX
H11 = gcn(A,X11,W1)
res = H11 - H
P = W1/sqrt(degs[2]*degs[1])
res11 = P*(dX)


### LSM layer test
k = 4
l = 5
H = rand(k,l)

dH = ones(size(H,1))*norm(H)/100
H11 = copy(H)
H11[:,2] += dH
Z11 = zeros(size(H))
Z = zeros(size(H))
for i in 1:l
    Z[:,i] = LSM(H[:,i])
    Z11[:,i] = LSM(H11[:,i])
end
res = Z11-Z
(I(k) - ones(k)exp.(Z[:,2]'))*dH

