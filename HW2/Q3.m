%% Newton-CG iterations 

A = [5, 1, 0, 0.5; ...
     1, 4, 0.5, 0; ...
     0, 0.5, 3, 0; ...
     0.5, 0, 0, 2];

% tolerance for the Newton iterations
tol = 1e-16; 
angle = 70/180*pi;
x = [cos(angle); sin(angle); cos(angle); sin(angle)];
sigma = 1;
mu = 10;

sol = [];
method = 1;
k = 0;

while 1
    grad = (eye(4) + mu*A)*x + sigma*(x'*A*x)*(A*x);
    sol = [sol, norm(grad)];
    if norm(grad) < tol
        break
    end

    hessian = eye(4) + mu*A + sigma*( 2*(A*x)*(A*x)' + (x'*A*x)*A);
    
    switch method
        case 1
            eta = 0.5;
        case 2
            eta = min(0.5, sqrt(norm(grad)));
        case 3
            eta = min(0.5, norm(grad));
    end
    
    % perform CG
    z0 = 0;
    r0 = grad;
    d0 = -r0;
    
    j = 0;
    
    while 1
        curvature = d0'*hessian*d0;
        if curvature <= 0
            if j == 0
                p0 = -grad;
                break
            else
                p0 = z0;
                break
            end
        end
        
        r_old = r0;
        
        alpha = r0'*r0 / curvature;
        z0 = z0 + alpha*d0;
        r0 = r0 + alpha*hessian*d0;
       
        if norm(r0) < eta*norm(grad)
            p0 = z0;
            break
        end
        
        beta = r0'*r0 / (r_old'*r_old);
        d0 = -r0 + beta*d0;
        j = j+1;        
     
    end
    j    
   
    % do line search according to Armijo here
    step_size = 1;
    c1 = 1e-4;
    while 1
        value = 0.5*x'*(eye(4)+mu*A)*x + 0.25*sigma*(x'*A*x)^2;
        x_new = x + step_size*p0;
        value_new = 0.5*x_new'*(eye(4)+mu*A)*x_new + 0.25*sigma*(x_new'*A*x_new)^2;
        if value_new <= value + c1*step_size*grad'*p0
            break
        else
            step_size = step_size / 2;
        end    
    end
    
    x = x + step_size*p0;
    
    k = k+1;
    
end  

figure;
semilogy(sol,'-o')

%% steepest descent with Armijo line search

x = [cos(angle); sin(angle); cos(angle); sin(angle)];
iter = 0;
sigma = 1;
mu = 10;
sol_steepest_descent = [];

while 1
    grad = (eye(4) + mu*A)*x + sigma*(x'*A*x)*(A*x);
    sol_steepest_descent = [sol_steepest_descent, norm(grad)];
    if norm(grad) < tol
        break
    end
    hessian = eye(4) + mu*A + sigma*( 2*(A*x)*(A*x)' + (x'*A*x)*A);
          
    % do line search according to Armijo here
    p0 = -grad;
    step_size = 1;
    c1 = 1e-4;
    while 1
        value = 0.5*x'*(eye(4)+mu*A)*x + 0.25*sigma*(x'*A*x)^2;
        x_new = x + step_size*p0;
        value_new = 0.5*x_new'*(eye(4)+mu*A)*x_new + 0.25*sigma*(x_new'*A*x_new)^2;
        if value_new <= value + c1*step_size*grad'*p0
            break
        else
            step_size = step_size / 2;
        end    
    end
    
    x = x + step_size*p0;
    
    iter = iter+1;
    
end  

figure;
semilogy(sol_steepest_descent,'-o')

