function [net, info] = dagnet_train_new(net, imdb, expDir, varargin)

%run (fullfile(fileparts(mfilename('fullpath')),'../../', 'matlab','vl_stupnn.m'));
run ('matconvnet-1.0-beta24/matlab/vl_setupnn');

%some options:
opts.train.batchSize = 100;
opts.train.numEpochs = 2;
opts.train.continue = true ;   % If set to true training will start where it stopt before (to start from zero use false)
opts.train.gpus = [] ;
opts.train.learningRate = 0;
opts.train.learningRate = logspace(-3,-5,20);
opts.train.weightDecay = 3e-4;
opts.train.momentum = 0.9;
opts.train.numSubBatches = 1;
%opts.train.imdbDir = 'data/noise_bmvc_original_imdb.mat';
%opts.train.imdb = imdb;
%opts.train.expDir  = fullfile('data');
opts.train.expDir = expDir;


%% initialization of parameters randomly:

net = initNet_xavier(net);

%% 
%%%-------------------------------------------------------------------------
%%%   Train 
%%%-------------------------------------------------------------------------

%[net,info] = cnn_train_dagCustom(net, imdb, @(i,b) getBatchDisk(bopts,i,b), opts.train, 'val', find(imdb.images.set == 2)) ;
[net,info] = cnn_train_dagCustom(net, imdb, @getBatch, opts.train) ;

end
