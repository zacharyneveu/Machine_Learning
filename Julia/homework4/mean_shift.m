function [shiftedClusterCenter] = mean_shift(clusterCenter,bandw,weight)
% this computes one iteration of the mean shift algorithm
% using a Gaussian kernel with standard deviation given by 'bandw'
shiftedClusterCenter=clusterCenter;

% form averaging distributions, one per cluster center, in columns of 'distMatrix'
distMatrix = exp(-pdist2(clusterCenter,clusterCenter)/(2*bandw^2));
distMatrix = (weight*ones(1,size(distMatrix,2))).*distMatrix;
normalization = sum(distMatrix,1);
distMatrix = distMatrix ./ (ones(size(distMatrix,1),1)*normalization);
for count = 1:size(shiftedClusterCenter,1)
	shiftedClusterCenter(count,:) = sum(distMatrix(:,count)*ones(1,size(clusterCenter,2)).*clusterCenter,1);
end

end

