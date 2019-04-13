using Distances 
"""
This computes one iteration of the mean shift algorithm
using a Gaussian kernel with standard deviation given by "bandw"

*(clusterCenter::AbstractArray, bandw::Float64, weight::Float64)
"""
function mean_shift(clusterCenter,bandw,weight)
    shiftedClusterCenter=clusterCenter

    # form averaging distributions, one per cluster center, in columns of "distMatrix"
    distMatrix = exp(-pairwise[clusterCenter,clusterCenter]/(2*bandw^2))
    distMatrix = (weight*ones(1,size(distMatrix,2))).*distMatrix
    normalization = sum(distMatrix,1)
    distMatrix = distMatrix ./ (ones(size(distMatrix,1),1)*normalization)
    for count = 1:size(shiftedClusterCenter,1)
            shiftedClusterCenter[count,:] = sum(distMatrix[:,count]*ones(1,size(clusterCenter,2)).*clusterCenter,1)
    end

    [shiftedClusterCenter]
end