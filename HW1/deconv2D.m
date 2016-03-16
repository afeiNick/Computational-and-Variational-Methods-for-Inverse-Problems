% The imported image called coto is a matrix of size 300x154.
% The image has values between 0 and 255 corresponding to
% different gray values. Note that the image values are unsigned
% integers (uint8) and need to be coverted to doubles for computation
N1 = 300;
N2 = 154;
N = N1 * N2;
x=linspace(0,N1-1,N1);
y=linspace(0,N2-1,N2);
[xx,yy] = meshgrid(x,y);
I = double(coto);
% plot the imported image as surface
figure(1)
surf(xx,yy,I); shading flat; colormap gray;
title('original image draws as 3D surface');
% alternative way to draw image
figure(2);
imagesc(I); colormap gray;
title('original image');

% Use different Gaussian blurring in x and y-direction
gamma1 = 4;
C1 = 1 / (sqrt(2*pi)*gamma1);
gamma2 = 3;
C2 = 1 / (sqrt(2*pi)*gamma2);

% setup blurring operators for x and y directions
K1 = zeros(N1,N1);
K2 = zeros(N2,N2);
for l = 1:N1
    for k = 1:N1
    	K1(l,k) = C1 * exp(-(l-k)^2 / (2 * gamma1^2));
    end
end
for l = 1:N2
    for k = 1:N2
    	K2(l,k) = C2 * exp(-(l-k)^2 / (2 * gamma2^2));
    end
end

% blur the image: first, K2 is applied to each column of I,
% then K1 is applied to each row of the resulting image
Ib = (K1 * (K2 * I)')';

% plot blurred image
figure(3); colormap gray;
surf(xx,yy,Ib);
shading interp
view(0,-90);   % top view
title('blurred image');


% add noise and plot noisy blurred image
Ibn = Ib + 3 * randn(N2,N1);
figure(4);
surf(xx,yy,Ibn);
colormap gray;
shading interp
view(0,-90);   % top view
title('blurred noisy image');

% compute Tikhonov reconstruction with regularization
% parameter alpha, i.e. compute p = (K'*K + alpha*I)\(K'*d)
K_Ibn = (K1 * (K2 * Ibn)')';

alpha_list = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1];
misfit = zeros(size(alpha_list));
I_diff_norm = zeros(size(alpha_list));
I_alpha_norm = zeros(size(alpha_list));

for alpha_index = 1:length(alpha_list)
    alpha = alpha_list(alpha_index);
    I_alpha = pcg(@(in)apply(in,K1,K2,N1,N2,alpha),K_Ibn(:),1e-6,1500);
    
%     figure;  colormap gray;
%     surf(xx,yy,reshape(I_alpha,N2,N1));
%     title('Tikhonov reconstruction');
%     shading interp
%     view(0,-90);   % top view
    
    % compute the misfit here
    misfit(alpha_index) = norm((K1 * (K2 * reshape(I_alpha,N2,N1))')' - Ibn, 'fro');
    I_alpha_norm(alpha_index) = norm(I_alpha);
    I_diff_norm(alpha_index) = norm(I(:) - I_alpha);
    
end

figure;
loglog(misfit, I_alpha_norm, '-o');
xlabel('misfit');
ylabel('norm of I_\alpha');

figure;
plot(alpha_list, I_diff_norm, '-o');
xlabel('\alpha');
ylabel('||I_{true} - I_\alpha||');

% write out image in PNG format
print -dpng coto_tikhonov

