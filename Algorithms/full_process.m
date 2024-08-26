%% Docstring

% This code reconstructs a DHLM hologram via the Kreuzer patent's method
% and the angular spectrum method.

% Inputs:
%  + image_name = name of the image file to reconstruct, and its reference
%  hologram
%  

% Carlos Trujillo.
% 2022 EAFIT University

clear all
close all

%% DLHM Hologram loading and reconstruction (Kreuzer's method, asuming spherical illumination)

% Sample info loading (blue)
filename = 'hol3..bmp (blue).tif';
holo=double(imread(filename));
[fi,co] = size(holo);

filename = 'ref3.bmp (blue).tif';
ref=double(imread(filename));
holoContrast = holo - ref;

% Geometrical parameters
z_rec = 2e-3;
L = 15e-3;
dx = 3.45e-6; %This value needs to be checked again.
lambda = 457e-9;

%Let's find the best compensated phase reconstruction
for z_rec = 1e-3:0.1e-3:3e-3
    [K, fi, dX] = reconstruct(holoContrast,z_rec,L,lambda,dx);
    amplitude = abs(K).^2;
    figure('WindowState','maximized'),imagesc(amplitude),colormap(gray),title('amplitude'),daspect([1 1 1])
end

% Spherical phase compensation (blue)

Ph_comp_final_2 = angle(K);
figure('WindowState','maximized'),imagesc(Ph_comp_final_2),colormap(gray),title('eri1_r3_blue'),daspect([1 1 1])
z_ini = 30e-3;
z_fin = 50e-3;
z_step = 1e-3;
[z_out] = compensate(K,fi,lambda,dX, z_ini, z_fin, z_step, 0);
z_out
[Ph_comp_final, comp_field_final] = recons_compens(z_out,K,fi,lambda,dX, 0);
figure('WindowState','maximized'),imagesc(Ph_comp_final),colormap(gray),title('eri1_r3_blue_comp'),daspect([1 1 1])
figure('WindowState','maximized'),imagesc(abs(comp_field_final).^2),colormap(gray),title('eri1_r3_blue_amp_rec'),daspect([1 1 1])


%% DLHM Hologram loading and reconstruction (Angular spectrum method - low NA)

% Sample info loading (blue)
filename = 'hol3..bmp (blue).tif';
holo=double(imread(filename));
[fi,co] = size(holo);

filename = 'ref3.bmp (blue).tif';
ref=double(imread(filename));
holoContrast = holo - ref;

% Geometrical parameters
z_rec = 2e-3;
L = 15e-3;
dx = 3.45e-6; %This value needs to be checked again.
lambda = 457e-9;
dy = dx;

amplitude_stack = [];
%Let's find the best focus amplitude reconstruction
for z_rec = 1e-3:0.1e-3:5e-3
    K = ang_spectrum(holoContrast,z_rec,lambda,dx,dy);
    amplitude = abs(K).^2;
    imageNormalized = normalize(amplitude);
    amplitude_stack = cat(3, amplitude_stack, imageNormalized);
    %figure('WindowState','maximized'),imagesc(amplitude),colormap(gray),title([num2str(z_rec*1e3),' mm rec amplitude']),daspect([1 1 1])
end
[~,~,l] = size(amplitude_stack);
implay(amplitude_stack,l);

%% Save this intensity as a file

z_rec = 2e-3;
K = ang_spectrum(holoContrast,z_rec,lambda,dx,dy);

G = abs(K).^2;
MAXPhase= max(max(G));
MINPhase = min(min(G));
FilePhase = (G- MINPhase)/(MAXPhase-MINPhase);
FilePhase = 255*FilePhase;
imwrite(FilePhase, gray(256), ['Resulting_intensity_', int2str(z_rec), '.jpg']);


%% Functions

% DLHM reconstruction
function [ K, fi, dX ] = reconstruct( hologram,z,L,lambda,dx )

    [fi,co] = size(hologram);

    %cosenus filter creation
    [FC] = filtcosenoF(100,fi);

    %pixel size at reconstruction plance
    dX = (z)*dx/L;%um

    %Reconstruct
    tic
    K = kreuzer3F(hologram,z,L,lambda,dx,dX,FC,'save');
    toc

    %intensity and phase calculation
    Am = abs(K);
    In = (abs(K)).^2;
    Ph = angle(K);

    %Rendering
    %figure('WindowState','maximized'),imagesc(In),colormap(gray),title([num2str(z*1e3),' mm rec In']),daspect([1 1 1])
    %figure('WindowState','maximized'),imagesc(Ph),colormap(gray),title([num2str(z*1e3),' mm rec Phase']),daspect([1 1 1])

end 
