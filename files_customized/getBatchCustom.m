function [im, label] = getBatchCustom(imdb, batch)
%GETBATCH  Get a batch of training data
%   [IM, LABEL] = The GETBATCH(IMDB, BATCH) extracts the images IM
%   and labels LABEL from IMDB according to the list of images
%   BATCH.
l = length(batch);
[row,col,channel] = size(imread(char(imdb.images.data(1))));
im = zeros(121,121,channel,l,'single');
[row,col,channel] = size(imread(char(imdb.images.label(1))));
label = zeros(121,121,channel,l,'single');
for i =1:l
tempim = imread(char(imdb.images.data(batch(i))));
%mea = mean(mean(mean(tempim)));
%tempim = tempim - mea;
randnum = randi([122 178],1,2);
tempimcrop = imcrop(tempim,[randnum(1) randnum(2) 120 120]);
templabel = imread(char(imdb.images.label(batch(i))));
templabelcrop = imcrop(templabel,[randnum(1) randnum(2) 120 120]);
%mea = mean(mean(mean(templabel)));
%templabel = templabel - mea;
im(:,:,:,i) = single(tempimcrop);
label(:,:,:,i) = single(templabelcrop - tempimcrop);
end
im = gpuArray(im) ;
label = gpuArray(label) ;
end