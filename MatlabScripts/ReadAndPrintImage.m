clc
clear all
clear vars
close all 
Image = csvread('./Image.csv');
Image_noisy = csvread('./NoisyImage.csv');
Image_denoised = csvread('denoised_cuda.csv');

subplot(1,4,1);
imshow(Image,[]);
title('Original');

subplot(1,4,2);
imshow(Image_noisy,[]);
title('Noisy');

subplot(1,4,3);
imshow(Image_denoised,[]);
title('denoised');

subplot(1,4,4);
Residual = Image_denoised - Image_noisy
imshow(Residual,[]);
title('Residual');

%   figure('Name', 'Filtered image');
%   imagesc(Image_denoised); axis image;
%   colormap gray;
%   
%   figure('Name', 'Residual');
%   imagesc(abs(Image_denoised - Image_noisy)); axis image;
%   colormap gray;