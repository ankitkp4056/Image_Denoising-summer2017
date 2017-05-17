
imdbDir = 'noise_bmvc_original_imdb.mat';
expDir  = fullfile('data\','exp');

%%% load training data
imdbPath    = fullfile(imdbDir);
imdb = load(imdbPath) ;

net  = feval('dagnet_init');
[net, info] = dagnet_train(net, imdb, expDir);
