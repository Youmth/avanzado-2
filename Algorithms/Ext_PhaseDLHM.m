%% Extending unambiguous phase measurements in DLHM
% Carlos Trujillo, Oct 13, 2021 (Initial version)
% Modified: Feb 01, 2022.

clear all;
close all;

%% Let's initially model a DLHM hologram of a pure phase object
% Phase Object creation

N = 512;
M = N;
[m,n] = meshgrid(1-M/2:M/2,1-N/2:N/2);

r = 0.1; %Height of peak and valley
m = r*(m/N);

X3 = padarray(m,[256 256],0,'both');

%figure('WindowState','maximized'),imagesc(X3),colormap(gray),title('Peak&valley'),daspect([1 1 1])
P = exp(1i*X3); %We are modelling a phase only object

figure('WindowState','maximized'),imagesc(angle(P)),colormap(gray),title('Peak&ValleyPhase'),daspect([1 1 1])
%figure('WindowState','maximized'),imagesc(abs(P)),colormap(gray),title('Peak&valleyAmp'),daspect([1 1 1])

%% Geometrical parameters

z = 0.5e-3;
z_rec = 1.36e-3;
L = 8e-3;
dx = 3.75e-6;

%First acquisition
lambda = 528e-9;
[hologram,reference,contrast,AN] = modeling(P,z,L,lambda,dx);
[K, fi, dX] = reconstruct(hologram,z_rec,L,lambda,dx);

Ph_comp_final_1 = angle(K);

[z_out] = compensate(K,fi,lambda,dX);
z_out
[Ph_comp_final_1] = recons_compens(z_out,K,fi,lambda,dX);

figure('WindowState','maximized'),imagesc(Ph_comp_final_1),colormap(gray),title('FirstADQ'),daspect([1 1 1])

AN
%max(max(Ph_comp_final_1))
%min(min(Ph_comp_final_1))

%Second acquisition
lambda = 632.8e-9;
[hologram,reference,contrast,AN] = modeling(P,z,L,lambda,dx);
[K, fi, dX] = reconstruct(hologram,z_rec,L,lambda,dx);

Ph_comp_final_2 = angle(K);

[z_out] = compensate(K,fi,lambda,dX);
z_out
[Ph_comp_final_2] = recons_compens(z_out,K,fi,lambda,dX);
figure('WindowState','maximized'),imagesc(Ph_comp_final_2),colormap(gray),title('SecondADQ'),daspect([1 1 1])

%% Let's reconstruct experimental DLHM holograms

%sample info loading (blue)
filename = 'eri1_r3_blue.tif';
holo=double(imread(filename));
[fi,co] = size(holo);

filename = 'ref3_blue.tif';
ref=double(imread(filename));
holoContrast = holo - ref;

% Geometrical parameters
%z_rec = 2.75e-3;
z_rec = 3.35e-3;
L = 7e-3;
dx = 3.6e-6; %This value needs to be checked again.

lambda = 473e-9;

%Let's find the best compensated phase reconstruction
%for z_rec = 2.6e-3:0.05e-3:3.2e-3
    [K, fi, dX] = reconstruct(holo,z_rec,L,lambda,dx);
%end

Ph_comp_final_2 = angle(K);

z_ini = 18e-3;
z_fin = 19e-3;
z_step = 0.01e-3;
[z_out] = compensate(K,fi,lambda,dX, z_ini, z_fin, z_step);
z_out
[Ph_comp_final_2] = recons_compens(z_out,K,fi,lambda,dX);
figure('WindowState','maximized'),imagesc(Ph_comp_final_2),colormap(gray),title('eri1_r3_blue'),daspect([1 1 1])


%% sample info loading (red)
filename = 'eri1_r3_red.tif';
holo=double(imread(filename));
[fi,co] = size(holo);

filename = 'ref3_red.tif';
ref=double(imread(filename));
holoContrast = holo - ref;

% Geometrical parameters
%z_rec = 2.75e-3;
z_rec = 3.35e-3;
L = 7e-3;
dx = 3.6e-6; %This value needs to be checked again.

lambda = 632.8e-9;

%Let's find the best compensated phase reconstruction
%for z_rec = 2.6e-3:0.1e-3:4e-3
    [K, fi, dX] = reconstruct(holo,z_rec,L,lambda,dx);
%end

Ph_comp_final_1 = angle(K);

z_ini = 18e-3;
z_fin = 19e-3;
z_step = 0.01e-3;
[z_out] = compensate(K,fi,lambda,dX, z_ini, z_fin, z_step);
z_out
[Ph_comp_final_1] = recons_compens(z_out,K,fi,lambda,dX);
figure('WindowState','maximized'),imagesc(Ph_comp_final_1),colormap(gray),title('eri1_r3_red'),daspect([1 1 1])

%% sample info loading (green)
filename = 'eri1_r3_green.tif';
holo=double(imread(filename));
[fi,co] = size(holo);

filename = 'ref3_green.tif';
ref=double(imread(filename));
holoContrast = holo - ref;

% Geometrical parameters
%z_rec = 2.75e-3;
z_rec = 3.35e-3;
L = 7e-3;
dx = 3.6e-6; %This value needs to be checked again.

lambda = 533e-9;

%Let's find the best compensated phase reconstruction
%for z_rec = 2.6e-3:0.1e-3:4.5e-3
    [K, fi, dX] = reconstruct(holo,z_rec,L,lambda,dx);
%end

Ph_comp_final_3 = angle(K);

z_ini = 17e-3;
z_fin = 20e-3;
z_step = 0.01e-3;
[z_out] = compensate(K,fi,lambda,dX, z_ini, z_fin, z_step);
z_out
[Ph_comp_final_3] = recons_compens(z_out,K,fi,lambda,dX);
figure('WindowState','maximized'),imagesc(Ph_comp_final_3),colormap(gray),title('eri1_r3_green'),daspect([1 1 1])

%% Extendeding Phase range

lambda1 = 528e-9;
lambda2 = 632.8e-9;
lam_synt = lambda1*lambda2/abs(lambda1 - lambda2)

for x = 1:1:fi
    for y = 1:1:fi
        
        if (Ph_comp_final_1(x,y) >= Ph_comp_final_2(x,y))
            Ph_comp_final(x,y) = Ph_comp_final_1(x,y) - Ph_comp_final_2(x,y);
        else
            Ph_comp_final(x,y) = Ph_comp_final_1(x,y) - Ph_comp_final_2(x,y) + 2*pi;
        end
       
    end
end

figure('WindowState','maximized'),imagesc(Ph_comp_final),colormap(gray),title('ExtendedPhase'),daspect([1 1 1])

max(max(Ph_comp_final))
min(min(Ph_comp_final))


%% Functions

% DLHM modelling
function [ hologram,reference,contrast,AN ] = modeling( P,z,L,lambda,dx )

    tic
    [ hologram, reference, contrast, AN ] = dlhm_sim(P, z, L, lambda, dx);
    toc

    %figure('WindowState','maximized'),imagesc(abs(hologram)),colormap(gray),title('hologram'),daspect([1 1 1])
    %figure('WindowState','maximized'),imagesc(abs(contrast)),colormap(gray),title('contrast holo'),daspect([1 1 1])
    %figure('WindowState','maximized'),imagesc(abs(reference)),colormap(gray),title('DF reference'),daspect([1 1 1])

end 

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
    figure('WindowState','maximized'),imagesc(In),colormap(gray),title([num2str(z*1e3),' mm rec In']),daspect([1 1 1])
    %figure('WindowState','maximized'),imagesc(Ph),colormap(gray),title([num2str(z*1e3),' mm rec Phase']),daspect([1 1 1])

end 

% Phase compensation
function [z_out ] = compensate( K, fi,lambda,dX, z_ini, z_fin, z_step)

    suma_maxima=0; %small number for the metric (thresholding)
    phase_stack = [];
    tic
    
    %Let's find the best compensated phase reconstruction
    for z = z_ini:z_step:z_fin
    
        comp_ref = point_src(fi,z,0,0,lambda,dX);
        comp_field = K .* comp_ref;

        %Rendering
        Ph_comp = angle(comp_field); 
    
        Ph_comp = (Ph_comp - min(min(Ph_comp(:)))) / (max(max(Ph_comp(:))) - min(min(Ph_comp(:))));
        Ph_comp = uint8(Ph_comp * 255);
    
        BW = im2bw(Ph_comp, 0.1);
        suma=sum(sum(BW)); %summation of all elements in the resulting matrix
    
        if (suma > suma_maxima)
            z_out = z;
            suma_maxima=suma;
        end   
    
        phase_stack = cat(3, phase_stack, Ph_comp);

    end

    [~,~,l] = size(phase_stack);
    %implay(phase_stack,l);

    toc

end 

% Reconstructing compensated phase image
function [Ph_comp_final] = recons_compens( z_out, K, fi,lambda,dX )

    comp_ref_final = point_src(fi,z_out,0,0,lambda,dX);

    comp_field_final = K .* comp_ref_final;

    % Phase Calculation
    Ph_comp_final = angle(comp_field_final); 

    %figure('WindowState','maximized'),imagesc(Ph_comp_final),colormap(gray),title([num2str(z_out*1e3),' mm rec PhaseCompensated']),daspect([1 1 1])

end
