function Y = vl_nnL2(X,c,dzdy)

% based on VLfeat issue 15:

if nargin <= 2
	Y = 0.5*sum((squeeze(X)'-c).^2);
else
	Y = +((squeeze(X)'-c))*dzdy;
	Y = reshape(Y,size(X));
end