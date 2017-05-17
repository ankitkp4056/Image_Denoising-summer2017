function makeBMVCData_copy 
imdb.images.id = {} ;
imdb.images.data = {} ;
imdb.images.set = {} ;
imdb.images.label = {} ;
data = imageDatastore('D:\summer_proj\data\ccd_noise_train_Dataset_1k\*_noise.png');
label = imageDatastore('D:\summer_proj\data\bmvc_data_shortlist_orig\*_orig.png');
[row,~] = size(data.Files);
imdb.images.id = 1:row;
imdb.images.set = ones(1, row);
%imdb.images.set(7001:row) = 2;
r = randi([1,row],1,uint8(30*row/100));
imdb.images.set(r(:)) = 2;
imdb.images.data = data.Files;
imdb.images.label = label.Files;
%save('D:\summer_proj\data\noise_bmvc_original_imdb.mat', 'imdb') ;
save('noise_bmvc_original_imdb.mat', '-struct', 'imdb') ;
end