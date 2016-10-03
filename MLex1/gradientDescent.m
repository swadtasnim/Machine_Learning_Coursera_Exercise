function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)


%GRADIENTDESCENT Performs gradient descent to learn theta

%   theta = GRADIENTDESENT(X, y, theta, alpha, num_iters) updates theta by 

%   taking num_iters gradient steps with learning rate alpha


% Initialize some useful values
 
 % number of training examples


   
       
 % ====================== YOUR CODE HERE ======================
    m = length(y);
   J_history = zeros(num_iters, 1);
for i=1:num_iters,
   p=X*theta;
   err=X'*(p-y);
   err=err/m;
theta=theta-alpha.*err;
    J_history(i)=computeCost(X,y,theta);
end

   
    

% Instructions: Perform a single gradient step on the parameter vector
   
 %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
   
 %








 % ============================================================

    
% Save the cost J in every iteration     J_history(iter) = computeCost(X, y, theta);


