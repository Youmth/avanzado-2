% USER FUNCTION TO PREPARE THE HOLOGRAM USING NEAREST NEIGHBOOR INTERPOLATION
% STRATEGY

function [CHp_m] = prepairholoF(CH_m,xop,yop,Xp,Yp)

[row,~] = size(CH_m);

%New coordinates measured in units of the -2*xop/(row) pixel size
Xcoord = (Xp - xop)/(-2*xop/(row));
Ycoord = (Yp - yop)/(-2*xop/(row));

%Find lowest integer
iXcoord = floor(Xcoord);
iYcoord = floor(Ycoord);

%Assure there isn't null pixel positions
iXcoord(iXcoord==0) = 1;
iYcoord(iYcoord==0) = 1;

% Calculate the fractioning for interpolation
x1frac = (iXcoord + 1.0)-Xcoord;                %upper value to integer
x2frac = 1.0-x1frac;                            %lower value to integer
y1frac = (iYcoord + 1.0)-Ycoord;
y2frac = 1.0-y1frac;

x1y1 = x1frac.*y1frac;                          %Corresponding pixel areas for each direction
x1y2 = x1frac.*y2frac;
x2y1 = x2frac.*y1frac;
x2y2 = x2frac.*y2frac;

%Pre-allocate the prepared hologram
CHp_m = zeros(row);

%Prepare hologram (preparation1 - every pixel remapping)
for it = 1:row-1
    for jt = 1:row-1
        
        CHp_m(iYcoord(it,jt),iXcoord(it,jt)) = CHp_m(iYcoord(it,jt),iXcoord(it,jt)) + (x1y1(it,jt))*CH_m(it,jt);
        CHp_m(iYcoord(it,jt),iXcoord(it,jt)+1) =  CHp_m(iYcoord(it,jt),iXcoord(it,jt)+1) + (x2y1(it,jt))*CH_m(it,jt);
        CHp_m(iYcoord(it,jt)+1,iXcoord(it,jt)) = CHp_m(iYcoord(it,jt)+1,iXcoord(it,jt)) + (x1y2(it,jt))*CH_m(it,jt);
        CHp_m(iYcoord(it,jt)+1,iXcoord(it,jt)+1) = CHp_m(iYcoord(it,jt)+1,iXcoord(it,jt)+1) + (x2y2(it,jt))*CH_m(it,jt);
        
    end
end