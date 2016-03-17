function [x,res] = mycg(A,b,maxiter,tol,x0)

r0 = A*x0 - b;
p0 = -r0;
k = 0;

while (norm(r0) > tol) && (k <= maxiter)
    curvature = p0'*A*p0;
    if curvature <= 0
        'non-positive curvature detected'
        break
    end
    
    r_old = r0;
    
    alpha = r0'*r0 / curvature;
    x0 = x0 + alpha*p0;
    r0 = r0 + alpha*A*p0;
    beta = r0'*r0 / (r_old'*r_old);
    p0 = -r0 + beta*p0;
    
    k = k+1;
end

k
x = x0;
res = norm(r0);
end