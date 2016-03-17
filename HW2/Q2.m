%% parameters
rng('default');
n = 100;
tol = 1e-6;
maxiter = 100;
M = spdiags(max(0,randn(n,1) - 1.5),0,n,n);
b = zeros(n,1);
x0 = 0.1*randn(n,1);

%% (a) testing my CG solver
testA = [4 -1 1; -1 4 -2; 1 -2 4];
testb = [12; -1; 5];
x = mycg(testA,testb,maxiter,tol,ones(3,1))

%% (b)
e = ones(n,1);
R = spdiags([-e 2*e -e], -1:1, n, n);
R(1,1) = 1;
R(n,n) = 1;


x1 = mycg(R,b,maxiter,tol,x0);
x2 = mycg(R,b,maxiter,tol,-x0);

norm(R*(x1-x2))


%% (c)
gamma_list = [1, 1e-4, 1e-8];
x_sol = zeros(n,length(gamma_list));
r_sol = zeros(length(gamma_list),1);
for gamma_index = 1:length(gamma_list)
    gamma = gamma_list(gamma_index);
    
    figure;
    semilogy(svd(full(M+gamma*R)),'o')
    
    [x_sol(:,gamma_index), r_sol(gamma_index)] = mycg(M+gamma*R,b,maxiter,tol,x0);
end

r_sol


%% (d)
x_sol_preconditioned = zeros(n,2*length(gamma_list));
r_sol_preconditioned = zeros(2*length(gamma_list),1);

for gamma_index = 1:length(gamma_list)
    gamma = gamma_list(gamma_index);
    L1 = diag(sqrt(diag(M+gamma*R)));
%     L2 = sqrt(gamma)*chol(R + 1e-10*eye(n), 'lower');
%     L2 = chol(M+gamma*R, 'lower');
%     L2 = ichol(M+gamma*R);
    
    L1inv = diag(1./sqrt(diag(M+gamma*R)));
    L2inv = inv(L2);
    
    pL1 = L1inv'*(M+gamma*R)*L1inv;
    pL2 = L2inv'*(M+gamma*R)*L2inv;
    
    figure;
    semilogy(svd(pL1),'o')
    
    figure;
    semilogy(svd(full(pL2)),'o')
    
    x_sol_preconditioned(:,2*gamma_index-1) = mycg(pL1,b,maxiter,tol,x0);
    x_sol_preconditioned(:,2*gamma_index) = mycg(pL2,b,maxiter,tol,x0);
    
    r_sol_preconditioned(2*gamma_index-1) = norm((M+gamma*R)*(L1\x_sol_preconditioned(:,2*gamma_index-1)));
    r_sol_preconditioned(2*gamma_index)   = norm((M+gamma*R)*(L2\x_sol_preconditioned(:,2*gamma_index)));
end

r_sol_preconditioned
