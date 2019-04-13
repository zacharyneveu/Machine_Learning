% this script imports the color bird image
clear
%
data=imread('42049_colorBird.jpg');
figure(1),subplot(131),image(data),title('42049_colorBird.jpg')

% form the raw features
nc=		size(data,2);% column size of image
nr=		size(data,1);% row size of image
featureSize = 	nc*nr; % number of feature vectors

numSamples = round(featureSize*randSampFrac); % random sample size
raw_feature=zeros(featureSize,5);
idx=0;
for rowcount=1:nr
    for colcount=1:nc
        idx=idx+1;
        raw_feature(idx,:)=[rowcount colcount double(data(rowcount,colcount,1)) double(data(rowcount,colcount,2)) double(data(rowcount,colcount,3))];
    end
end

% normalize the feature vectors
[feature,mu,sigma]=zscore(raw_feature);

% find the principal components of the features
[coeff,score,latent] = pca(feature);
