function [net, info] = dagnet_train(net, imdb, expDir, varargin)

%run (fullfile(fileparts(mfilename('fullpath')),'../../', 'matlab','vl_stupnn.m'));
run ('matconvnet-1.0-beta24/matlab/vl_setupnn');

%some options:
opts.train.batchSize = 100;
opts.train.numEpochs = 300;
opts.train.continue = true ;   % If set to true training will start where it stopt before (to start from zero use false)
opts.train.gpus = [1] ;
opts.train.learningRate = logspace(-3,-5,20);
opts.train.weightDecay = 3e-4;
opts.train.momentum = 0.9;
opts.train.numSubBatches = 1;
%opts.train.imdbDir = 'data/noise_bmvc_original_imdb.mat';
%opts.train.imdb = imdb;
%opts.train.expDir  = fullfile('data');
opts.train.expDir = expDir;
opts.train.gradClipping = true;

%%
%%%------------;-------------------------------------------------------------
%%%   Initialize model 
%%%-------------------------------------------------------------------------
%%%  model
%net  = feval(['dagnet_init']);

%% initialization of parameters randomly:

function initNet_xavier(net)
net.initParams();
for l=1:length(net.layers)
    
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv')) %TO CHECK CONV LAYER
        f_ind = net.layers(l).paramIndexes(1);           %Getting FILTERS index
        b_ind = net.layers(l).paramIndexes(2);           %Getting BIASES index 
        
        [h,w,in,out] = size(net.params(f_ind).value);    
        xav = 0.5*sqrt(2/(h*w*in));                      % sqrt(1/fan_in)
        net.params(f_ind).value = xav*randn(size(net.params(f_ind).value), 'single');
        net.params(f_ind).learningRate = 0.5;
        net.params(f_ind).weightDecay = 1;
        
        net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate = 1;
        net.params(b_ind).weightDecay = 1;
    end
end
end

%% 
%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------
%do the training!
%[net,info] = cnn_train_dagCustom(net, imdb, @(i,b) getBatchDisk(bopts,i,b), opts.train, 'val', find(imdb.images.set == 2)) ;
[net,info] = cnn_train_dagCustom(net, imdb, @getBatchCustom, opts.train) ;


%% getBatchCustom:

function inputs = getBatchCustom(imdb, batch)
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
inputs = {'data', im, 'label', label} ;
%im = gpuArray(im) ;
%label = gpuArray(label) ;
end

end

