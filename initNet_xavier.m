%% initialization of parameters randomly:

function net = initNet_xavier(net)
net.initParams();
for l=1:length(net.layers)
    
    if(strcmp(class(net.layers(l).block), 'dagnn.Conv')) %TO CHECK CONV LAYER
        f_ind = net.layers(l).paramIndexes(1);           %Getting FILTERS index
        b_ind = net.layers(l).paramIndexes(2);           %Getting BIASES index 
        
        [h,w,in,out] = size(net.params(f_ind).value);    
        xav = 0.5*sqrt(2/(h*w*in))                      % sqrt(1/fan_in)
        net.params(f_ind).value = xav*randn(size(net.params(f_ind).value), 'single');
        net.params(f_ind).learningRate = 0.5;
        net.params(f_ind).weightDecay = 1;
        
        net.params(b_ind).value = zeros(size(net.params(b_ind).value), 'single');
        net.params(b_ind).learningRate = 1;
        net.params(b_ind).weightDecay = 1;
    end
end
end