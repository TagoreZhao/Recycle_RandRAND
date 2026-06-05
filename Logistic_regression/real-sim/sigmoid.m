function s = sigmoid(z)
%SIGMOID  Numerically stable logistic function s = 1 ./ (1 + exp(-z)).
%   S = SIGMOID(Z) evaluates the logistic sigmoid element-wise on Z without
%   overflowing exp for large |Z|. For z >= 0 it uses 1/(1+e^{-z}); for z < 0
%   it uses e^{z}/(1+e^{z}). Both branches avoid exp of a large positive
%   argument.
%
%   Input
%     z : real array.
%   Output
%     s : array the same size as z with values in (0, 1).

    s = zeros(size(z));
    pos = z >= 0;

    s(pos) = 1 ./ (1 + exp(-z(pos)));

    ez = exp(z(~pos));
    s(~pos) = ez ./ (1 + ez);
end
