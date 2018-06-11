function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);
prev_theta= theta;
for iter = 1:num_iters
    
    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %
    sum= zeros(2,1);
    for i=1:m
      error=(theta(1,1)*X(i,1)+theta(2,1)*X(i,2)-y(i,1));
      for j=1:2
        sum(j,1)= sum(j,1) + error*X(i,j);
      end
    end

    prev_theta= theta;
    for i=1:2
      theta(i,1)= prev_theta(i,1)- alpha*(1/m)*sum(i,1);
    end
   

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
