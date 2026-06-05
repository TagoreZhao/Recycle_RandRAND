function Phi = eval_cosine_modes(p, kvec, bbox)
% Evaluate cosine modes cos(pi*kx*xhat)cos(pi*ky*yhat) on a mapped box.
% p:    N x 2 points
% kvec: K x 2 mode indices
% bbox: [xmin xmax ymin ymax] (optional). Default [-2 2 -2 2].

if nargin < 3, bbox = [-2 2 -2 2]; end
x = p(:,1); y = p(:,2);

xmin=bbox(1); xmax=bbox(2); ymin=bbox(3); ymax=bbox(4);
xhat = (x - xmin) / (xmax - xmin);
yhat = (y - ymin) / (ymax - ymin);

K = size(kvec,1);
Phi = zeros(numel(x), K);
for m = 1:K
    kx = kvec(m,1); ky = kvec(m,2);
    Phi(:,m) = cos(pi*kx*xhat) .* cos(pi*ky*yhat);
end
end
