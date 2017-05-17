function net = dagnet_init(varargin)

% network definition!
% MATLAB handle, passed by reference

net = dagnn.DagNN() ;

net.addLayer('conv1', dagnn.Conv('size', [3 3 3 32], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'data'}, {'conv1'},  {'conv1f'  'conv1b'});
net.addLayer('relu1', dagnn.ReLU(), {'conv1'}, {'relu1'}, {});

net.addLayer('conv2', dagnn.Conv('size', [3 3 32 32], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu1'}, {'conv2'},  {'conv2f'  'conv2b'});
net.addLayer('relu2', dagnn.ReLU(), {'conv2'}, {'relu2'}, {});

net.addLayer('conv3', dagnn.Conv('size', [3 3 32 32], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu2'}, {'conv3'},  {'conv3f'  'conv3b'});
net.addLayer('relu3', dagnn.ReLU(), {'conv3'}, {'relu3'}, {});

net.addLayer('conv4', dagnn.Conv('size', [3 3 32 32], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu3'}, {'conv4'},  {'conv4f'  'conv4b'});
net.addLayer('relu4', dagnn.ReLU(), {'conv4'}, {'relu4'}, {});

net.addLayer('conv5', dagnn.Conv('size', [3 3 32 3], 'hasBias', true, 'stride', [1, 1], 'pad', [0 0 0 0]), {'relu4'}, {'conv5'},  {'conv5f'  'conv5b'});

net.addLayer('prediction' , ...
                 dagnn.SumCustom(), ...
                 {'data', 'conv5'}, ...
                 { 'prediction'}) ;

%net.addLayer('loss', dagnn.Loss('loss', 'softmaxlog'), {'prediction', 'label'}, {'objective'});
net.addLayer('l2_loss', dagnn.L2Loss(), {'prediction', 'label'}, {'objective'});
end






