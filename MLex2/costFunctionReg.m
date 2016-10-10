function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


[j,g]=costFunction(theta, X, y);
h=sigmoid(X*theta);
q=0;
for i=2:size(theta,1)
	q=q+theta(i,:).^2;
J=j+(lambda/(2*m))*q;
X=X';
grad(1)=(1/m)*sum(X(1,:)*h-X(1,:)*y);

for i=2:size(theta,1)
	grad(i)=(1/m).*((sum(X(i,:)*(h-y)))+lambda*(theta(i,:)));
end
X=X';




% =============================================================

end
