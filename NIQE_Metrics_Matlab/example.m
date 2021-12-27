
 load modelparameters.mat
 
 blocksizerow    = 96;
 blocksizecol    = 96;
 blockrowoverlap = 0;
 blockcoloverlap = 0;

% im =imread('image1.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('image2.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% im =imread('image3.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
% 
% 
% im =imread('image4.bmp');
% 
% quality = computequality(im,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
%     mu_prisparam,cov_prisparam)
total=0
count=0
file_path =  '/Users/zhangfeng/Documents/GitHub/HEP/LUM_VV_Eng/';% 图像文件夹路径
img_path_list =  dir(strcat(file_path,'*.png'));%获取该文件夹中所有jpg格式的图像
img_num = length(img_path_list);%获取图像总数量
for j = 1:img_num %逐一读取图像
     image_name = img_path_list(j).name;% 图像名
     image = imread(strcat(file_path,image_name));
     quality = computequality(image,blocksizerow,blocksizecol,blockrowoverlap,blockcoloverlap, ...
    mu_prisparam,cov_prisparam)
     total=total+quality 
     count=count+1
end
avg = total/count
