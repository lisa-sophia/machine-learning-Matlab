function E = WeakClassifierError(C, D, Y)
% WEAKCLASSIFIERERROR  Calculate the error of a single decision stump.
% Takes a vector C of classifications from a weak classifier, a vector D
% with weights for each example, and a vector Y with desired
% classifications. Calculates the weighted error of C, using the 0-1 cost
% function.

% Try to not use loops, since a loop will be too slow to use
% with a reasonable amount of Haar features and training images.

E = sum(0.5 * D .* abs(C - Y));

end

