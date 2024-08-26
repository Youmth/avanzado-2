% FUNCTION TO RECONSTRUCT AN IN LINE HOLOGRAM WITH THE METHODOLOGY FROM THE
% JÜRGEN KREUZER'S PATENT

function [K] = kreuzer3F(CH_m,z,L,lambda,deltax,deltaX,FC,str)

%Square pixels:
deltaY = deltaX;

%Matrix size
[row,~] = size(CH_m);

%Parameters
k = 2*pi/lambda;
W = deltax*row;

%Matriz coordinates
[X,Y] = meshgrid(1:row,1:row);

%Hologram origin coordinates
xo = -W/2;  
yo = -W/2;

%Prepared Hologram, coordinates origin
xop = xo * L/sqrt(L^2 + xo^2);
yop = yo * L/sqrt(L^2 + yo^2);

%Pixel size for the prepared hologram
deltaxp = xop/(-row/2);
deltayp = deltaxp;

%Coordinates origin for the reconstruction plane
Yo = -deltaX*(row)/2 ;
Xo = -deltaX*(row)/2 ;

Xp = (deltax)*(X-row/2)*L./realsqrt(L.^2 + (deltax^2)*(X-row/2).^2 + (deltax^2)*(Y-row/2).^2);
Yp = (deltax)*(Y-row/2)*L./realsqrt(L.^2 + (deltax^2)*(X-row/2).^2 + (deltax^2)*(Y-row/2).^2);

%Search for prepared hologram if needed
current_folder = pwd;
file = strcat(current_folder,'\',str);

%Preparation of the hologram when neccesary
if exist(file,'file')==0
    
    %Prepare holo
    [CHp_m] = prepairholoF(CH_m,xop,yop,Xp,Yp);

%    save(str,'CHp_m');
    
else
    
    %load .mat file with the saved prepared hologram
    load(str);

end

%Multiply prepared hologram with propagation phase

Rp = sqrt((L^2)-(deltaxp*X+xop).^2-(deltayp*Y+yop).^2);
r = sqrt((deltaX^2)*((X-row/2).^2+(Y-row/2).^2) + (z)^2); 
CHp_m = CHp_m.*((L./Rp).^4).*exp(-0.5*1i*k*(r.^2 - 2*z*L).*Rp./(L^2));
    %%.*exp(0.125*1i*k*((L^2)./Rp).*((r.^2 - 2*z*L).*(Rp./(L^2)).^2).^2);...
    %%.*exp(-(3/48)*1i*k*((L^2)./Rp).*((r.^2 - 2*z*L).*(Rp./(L^2)).^2).^3)...
    %%.*exp((15/384)*1i*k*((L^2)./Rp).*((r.^2 - 2*z*L).*(Rp./(L^2)).^2).^4);

%Padding constant value
pad = row/2;

%Padding on the cosine rowlter
FC = padarray(FC,[pad pad]);

%Convolution operation
%First transform
T1 = CHp_m.*exp((1i*k/(2*L))*( 2*Xo*X*deltaxp + 2*Yo*Y*deltayp + (X).^2*deltaxp*deltaX + (Y).^2*deltayp*deltaY));
T1 = padarray(T1,[pad pad]);
T1 = fftshift(fft2(fftshift(T1.*FC)));

%Second transform
T2 = exp(-1i*(k/(2*L))*((X-row/2).^2*deltaxp*deltaX + (Y-row/2).^2*deltayp*deltaY));
T2 = padarray(T2,[pad pad]);
T2 = fftshift(fft2(fftshift(T2.*FC)));

%Third transform
K = ifftshift(ifftn(ifftshift(T2.*T1)));
K = K(pad+1:pad+row,pad+1:pad+row);

%Multiply by aditional terms after the propagation
% K = K.*deltaxp.*deltayp.*(exp(sqrt(-1)*(k/L)*((Xo+X*deltaX).*xop+(Yo+Y*deltaY).*yop)))...
%     .*exp(sqrt(-1)*(0.5*k/L)*((X-0*row/2).^2*deltaxp*deltaX + (Y-0*row/2).^2*deltayp*deltaY));
