X = table2array(data(:,3:32)); % Discard ID, diagnosis, and last columns
label = table2array(data(:,2));
k = 3; % Number of principal components
[Y, ratio,loading] = PCA(X,k);
color = zeros(569,3);
for i = 1 : 569
    if(label(i) == 'M')
        color(i,:) = [0 0 1];
    else
        color(i,:) = [1 0 0];
    end
end

figure(1);
gscatter(Y(:,1), Y(:,2), label);
xlabel("First Principal Component");
ylabel("Second Principal Component");

figure(2);
bar(ratio(1:k).*100, 0.5);
set(gca, 'xtick', 0:k);
title("Explained Variance of Principal Components");
xlabel("Principal Component");
ylabel("Explained Variance (%)");

xvalues = {'radius mean','texture mean', 'perimeter mean', 'area mean', ...
    'smoothness mean', 'compactness mean', 'concavity mean', ...
    'concave points mean', 'symmetry mean', 'fractal dimension mean', ...
    'radius se',  'texture se', 'perimeter se',  'area se', ...
    'smoothness se',  'compactness se', 'concavity se',  ...
    'concave points se', 'symmetry se', 'fractal dimension_se', ...
    'radius worst', 'texture worst', 'perimeter worst', 'area worst', ...
    'smoothness worst', 'compactness worst', 'concavity worst',  ...
    'concave points worst',  'symmetry worst', 'fractal dimension_worst'};
figure(3);
h = heatmap(xvalues, 1:k, loading','Colormap',hot);
title(h, "Principal Components Correlation with the Features");
xlabel("Features");
ylabel("Principal Components");

function [Y, ratio, loading]= PCA(X, k)
% m = number of samples
% n = number of features
[m,n] = size(X);

% Transpose the matrix to match the algorithm
X = X';

% Normalization
for i = 1 : n
    avg = mean(X(i,:));
    s = var(X(i,:));
    X(i,:) = (X(i,:) - avg) / sqrt(s);
end

X_hat = X' ./ sqrt(m-1);
[U, S, V] = svd(X_hat);

%Compute ratio of explained variance
L = [];
total = 0;
for i = 1 : n
    L(i) = S(i,i)^2;
    total = total + L(i);
end
ratio = [];
for i = 1 : k
    ratio(i) = L(i) / total;
end

Y = (V(:,1:k)'*X)';
loading = V(:,1:k);
end