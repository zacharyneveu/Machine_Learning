
%% PCA
load normFeatures
[coeff,score,latent,~,~,mu] = pca(normFeatures,'Algorithm','ALS');
%normFeat_rec = score*coeff' + repmat(mu,NUMtrain,1);
pareto(latent)
%% Find # of principal components
tot = sum(latent);
perc = cumsum(latent)./tot;
figure()
bar(latent/tot)
hold on
yyaxis right
plot(1:length(latent),perc,'o-')

%% Find final feature table after dimensionality reduction
NumOfPC = find(perc>0.95, 1 );
Data_array = score(:,1:NumOfPC);

trainLabel = readtable('/home/zach/Downloads/train.csv');
Class = categorical(trainLabel.Class);
Data = [table(Class) array2table(Data_array)];

%% Partition training and test set
part = cvpartition(Data.Class, 'HoldOut',0.2);
trainData = Data(training(part),:);
testData = Data(test(part),:);
mdl = fitcknn(trainData,'Class');
loss(mdl,testData)
predictedClass = predict(mdl,testData);
hold off
C = confusionchart(testData.Class, predictedClass);
