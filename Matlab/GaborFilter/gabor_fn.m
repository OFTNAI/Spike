function gb=gabor_fn(bw,gamma,psi,lambda,theta,plot)
% bw    = bandwidth, (1)
% gamma = aspect ratio, (0.5)
% psi   = phase shift, (0)
% lambda= wave length, (>=2) (pixels)
% theta = angle in rad, [0 pi)
%The output is always real. For a complex Gabor filter, you may use (0 phase) + j(pi/2 phase).

if nargin<6; plot=false; end

sigma = lambda/pi*sqrt(log(2)/2)*(2^bw+1)/(2^bw-1);
sigma_x = sigma;
sigma_y = sigma/gamma;

sz=fix(8*max(sigma_y,sigma_x));
if mod(sz,2)==0, sz=sz+1;end

% alternatively, use a fixed size
% sz = 60;
 
[x y]=meshgrid(-fix(sz/2):fix(sz/2),fix(sz/2):-1:fix(-sz/2));
% x (right +)
% y (up +)

% Rotation 
x_theta=x*cos(theta)+y*sin(theta);
y_theta=-x*sin(theta)+y*cos(theta);
 
gb=exp(-.5*(x_theta.^2/sigma_x^2+y_theta.^2/sigma_y^2)).*cos(2*pi/lambda*x_theta+psi);

if (plot)
    figure(); imshow(gb/2+0.5); % Rescale from [-1,1] to [0,1]
    figure(); meshc(x,y,gb);
end

% http://www.mathworks.com/matlabcentral/fileexchange/23253