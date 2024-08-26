%FUNCIÓN PARA CREAR FILTRO COSENO
%
%ENTRADAS:par->par: Parámetro de tamaño
%           fi-> tamaño de la matriz
%           num_fig(Opcional) -> número de la figura en que se quiere se
%           represente el filtro
%SALIDAS:FC->MAtriz reescalada de 0 a 1 del filtro
%
%EJO: [FC] = filtcosenoF(2,1024,1)
% 

function [FC] = filtcosenoF(par,fi,num_fig)

%Coordenadas
[Xfc,Yfc]=meshgrid(linspace(-fi/2,fi/2,fi),linspace(fi/2,-fi/2,fi));

% Normalizar cooredenadas en intervalo (-pi, pi) y crear filtros en
% dirección horizontal y vertical
FC1 = cos(Xfc*(pi/par)*(1/max(max(Xfc))))^2; 
FC2 = cos(Yfc*(pi/par)*(1/max(max(Yfc))))^2;

%Intersectar ambas direcciones
FC = (FC1>0).*(FC1).*(FC2>0).*(FC2);

%Re-escalar de 0 a 1
FC = FC/max(max(FC));

if nargin == 3
    %Representar
    imagescp(num_fig,FC)
end

