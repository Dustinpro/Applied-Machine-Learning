feature = zeros(0,0);
label = feature;
% vertically combine data batches, stored in "feature" and "label"
for i = 1:5
    load(strcat(strcat('cifar-10-batches-mat/data_batch_',int2str(i)),'.mat'));
    feature = vertcat(feature, data);
    label = vertcat(label, labels);
end
labelNames = {'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck'};
% get a horizontally combined whole dataset (feature + label)
data = horzcat(feature, label);
[nrows, ncols] = size(data);
% sort the data by the label column and convert to the double type
data = double(sortrows(data, ncols));

loadings = zeros(10, ncols-1, ncols-1);	% eigenvectors
means = zeros(10, ncols-1);				% mean image
errors = zeros(10, 1);					% MSE error
npcs = 20;								% the number of principal components

for num_label = 1:10
    X = data((num_label-1)*5000+1:num_label*5000, 1:ncols-1);
    % refer to Matlab Documentation: https://www.mathworks.com/help/releases/R2017b/stats/pca.html
    % "coeff": p*p loadings
    % "score": Principal component scores are the representations of X in the principal component space. Rows of score correspond to observations, and columns correspond to components.
    % "explained": the percentage of the total variance explained by each pc
    % "mu": the estimated mean of each variable in X
    [coeff,score,~,~,explained,mu] = pca(X, 'NumComponents', npcs);
    loadings(num_label, :, 1:npcs) = coeff;
    means(num_label, :) = mu;
    % the second argument of immse is the result of reconstruction
    errors(num_label) = immse(X, score*coeff' + repmat(mu,5000,1)) * (ncols-1);
end

% begin to plot the error resulting from representing the images using 20 pcs.
bar(errors);
xticklabels(labelNames);
xlabel("Ten Categories");
ylabel("Errors");
title("Errors of Each Category");
% download barvalues() from https://www.mathworks.com/matlabcentral/fileexchange/64963-barvalues-h-precision
barvalues([]);
saveas(gcf,'error.png');

% begin to plot 10 mean images
for num_label = 1:10
	% make it a cubic image
	Image = reshape(uint8(means(num_label, :)), [32, 32, 3]);
	% exchange row and col.
	Image = permute(Image, [2, 1, 3]);
    imwrite(Image, strcat(strcat('mean',int2str(num_label)),'.png'));
end


% get Euclidean distance
distances = pdist(means);

% multidimensional scaling gets posisions form distances
[positions, ~] = mdscale(distances, 2);

% plot the 2d map
scatter(positions(:, 1), positions(:, 2));
text(positions(:, 1), positions(:, 2), labelNames);
saveas(gcf,'2dmap.png');