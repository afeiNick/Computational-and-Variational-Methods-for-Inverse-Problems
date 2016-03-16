% One-dimensional de-blurring example

% Number of discretization points for 1D interval including end points
N = 200;
c = 0.2;

K = zeros(N,N);
h = 1/(N-1);
x = linspace(0,1,N)';

for l = 1:N
    for k = 1:N
        K(l,k) = h / c^2 * max(0, c-h*abs(l-k));
    end
end

p = 1*(x>0.1).*(x<0.25) + 0.25*(x>0.25).*(x<0.4) + (sin(2*pi*x)).^4.*(x>0.5).*(x<1);
d = K * p;


% no noise
p_alpha = K\d;
figure;
plot(x,p,x,p_alpha,'Linewidth', 2), axis([0,1,-1.5,1.5]);
legend('exact data','reconstruction no noise, no regularizaton');

% noisy data
n = sqrt(0.05) * randn(N,1);
dn = d + n;

p_alpha = K\dn;
figure;
plot(x,p,x,p_alpha,'Linewidth', 2), axis([0,1,-1.5,1.5]);
legend('exact data','reconstruction tiny noise, no regularizaton');


% TSVD
alpha_list = [1e-4, 1e-3, 1e-2, 1e-1];
s = svd(K);
[U,~,V] = svd(K);

for alpha_index = 1:length(alpha_list)
    p_alpha = zeros(N,1);
    alpha = alpha_list(alpha_index);
    for i = 1:N
        if s(i)^2 >= alpha
            p_alpha = p_alpha + (U(:,i)'*dn)*U(:,i) / s(i);
        end
    end
    figure;
    plot(x,p,x,p_alpha,'Linewidth', 2), axis([0,1,-1.5,1.5]);
    legend('exact data','TSVD reconstruction');    
    title(['alpha = ', num2str(alpha)]);
end
    


% Tikhonov
misfit = zeros(size(alpha_list));
p_alpha_norm = zeros(size(alpha_list));
p_diff_norm = zeros(size(alpha_list));

for alpha_index = 1:length(alpha_list)
    alpha = alpha_list(alpha_index);
    % solve Tikhonov system (this could also be done using a QR decomposition)
    p_alpha = (K'*K + alpha * eye(N))\(K'*dn);
    misfit(alpha_index) = norm(K*p_alpha - dn);
    p_alpha_norm(alpha_index) = norm(p_alpha);
    p_diff_norm(alpha_index) = norm(p-p_alpha);
    
    figure;
    plot(x,p,x,p_alpha,'Linewidth', 2), axis([0,1,-1.5,1.5]);
    legend('exact data', 'Tikhonov reconstruction');
    title(['alpha = ', num2str(alpha)]);    
end

figure;
loglog(misfit, p_alpha_norm, '-o');
xlabel('misfit');
ylabel('||p_\alpha||');


delta = norm(n);
figure;
plot(alpha_list, misfit, '-o', alpha_list, delta*ones(size(alpha_list)),'r-');
xlabel('\alpha');
ylabel('misfit');

figure;
plot(alpha_list, p_diff_norm, '-o');
xlabel('\alpha');
ylabel('||p_{true} - p_\alpha||');


return;
