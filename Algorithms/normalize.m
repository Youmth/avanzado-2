function imageNormalized = normalize(image)

%function to scale image to range [0,1]

% example: 

% [image,~ ] = meshgrid(-10:10,-10:10);
% figure, imagesc(image), colorbar
% 
% imageNormalized =normalize(image);
% 
% figure,imagesc(imageNormalized), colorbar

imageNormalized = (image - min(image,[],'all'))/(max(image,[],'all') - min(image,[],'all'));

end