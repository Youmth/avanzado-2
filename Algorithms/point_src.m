function [P] = point_src(M,z,x0,y0,lambda,dx)
%POINT_SRC Function to create a point source illumination centered in
%(x0,y0) and observed in a plane at a distance z.
%   P = point_src(M,z,x0,y0,lambda,dx)
%       z           screen distance
%       x0,y0       center
%       lambda      wavelength
%       dx,dy       sampling pitches


N = M;
dy = dx;

[m,n] = meshgrid(1-M/2:M/2,1-N/2:N/2);

k = 2 * pi / lambda;
r = sqrt(z^2 + (m*dx - x0).^2 + (n*dy - y0).^2);

P = exp(1i * k * r) ./ r;

end

