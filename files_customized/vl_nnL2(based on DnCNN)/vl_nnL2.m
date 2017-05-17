function Y = vl_nnL2(X,c,dzdy,varargin)

% based on DnCNN:

if nargin <= 2 || isempty(dzdy)
    t = ((X-c).^2)/2;
    Y = sum(t(:))/size(X,4); % reconstruction error per sample;
else
    Y = bsxfun(@minus,X,c).*dzdy;
end