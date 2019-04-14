
%% Exported from Jupyter Notebook
% Run each section by placing your cursor in it and pressing Ctrl+Enter

%% Code Cell[1]:

addpath(".")

%% Code Cell[2]:

% Set this to the directory where the data is stored
dataFolder = "~/Music/urbansound8k";

%% Code Cell[3]: Get labels for dataset

ads = audioDatastore(strcat(dataFolder,"/Train"));
[~, names, ~] = cellfun(@fileparts, ads.Files, 'UniformOutput', false);
names = cellfun(@str2num, names);
[~, idxs] = sort(names);
ads.Files = ads.Files(idxs);
labels = readtable(strcat(dataFolder,"/train.csv"));
ads.Labels = categorical(labels.Class);

%% Code Cell[4]:

[sTrain, sVal] = splitEachLabel(ads, 0.8);

% # Neural Networks
% Create spectrograms from the data. The dimension of the output data will then be 40x396. Shorter audio clips are padded equally on both sides, see `spectrograms.m` for details.

%% Code Cell[20]:

segmentDuration = 4;
frameDuration = 0.05;
hopDuration = 0.010;
numBands = 40;
reset(sTrain);
reset(sVal);
epsilon = 1e-6; % Added so that log doesn't encounter 0
XTrain = log10(spectrograms(sTrain, segmentDuration, frameDuration, hopDuration, numBands)+epsilon);
XVal = log10(spectrograms(sVal, segmentDuration, frameDuration, hopDuration, numBands)+epsilon);


% Create categorical vectors for labels
YTrain = categorical(sTrain.Labels);
YVal = categorical(sVal.Labels);


% Visualize some training data
n = randi([0,500], 1, 1);
pcolor(XTrain(:,:,:,n))
title(strrep(YTrain(n,:), '_', ' '))
shading flat
[samps, sampfreq] = audioread(sTrain.Files{n});
sound(samps, sampfreq)

% ## Conv Net from Scratch
% This section uses a simple convolutional neural network to classify spectrograms and achieves a validation set accuracy of 95.86% using holdout validation set of 20% of the data. The network for this section is largely the same as one from a speech recognition tutorial by Mathworks [here](https://www.mathworks.com/help/deeplearning/examples/deep-learning-speech-recognition.html?s_tid=mwa_osa_a). This CNN is trained using the [Adam](https://arxiv.org/abs/1412.6980) optimizer, and begins to overfit starting at about 7 epochs using the learning rate $3*10^{-3}$.

sz = size(XTrain);
specSize = sz(1:2);
imageSize = [specSize 1];
numClasses = numel(categories(YTrain));

timePoolSize = ceil(imageSize(2)/8);
dropoutProb = 0.2;
numF = 12;
layers = [
    imageInputLayer(imageSize, 'Name', 'Input_Layer')

    convolution2dLayer(3,numF,'Padding','same', 'Name', 'Conv_1')
    batchNormalizationLayer('Name', 'BN_1')
    reluLayer('Name', 'Relu_1')

    maxPooling2dLayer(3,'Stride',2,'Padding','same', 'Name', 'MaxPool_1')

    convolution2dLayer(3,2*numF,'Padding','same', 'Name', 'Conv_2')
    batchNormalizationLayer('Name', 'BN_2')
    reluLayer('Name', 'Relu_2')

    maxPooling2dLayer(3,'Stride',2,'Padding','same', 'Name', 'MaxPool_2')

    convolution2dLayer(3,4*numF,'Padding','same', 'Name', 'Conv_3')
    batchNormalizationLayer('Name', 'BN_3')
    reluLayer('Name', 'Relu_3')

    maxPooling2dLayer(3,'Stride',2,'Padding','same', 'Name', 'MaxPool_3')

    convolution2dLayer(3,4*numF,'Padding','same', 'Name', 'Conv_4')
    batchNormalizationLayer('Name', 'BN_4')
    reluLayer('Name', 'Relu_4')
    convolution2dLayer(3,4*numF,'Padding','same', 'Name', 'Conv_5')
    batchNormalizationLayer('Name', 'BN_5')
    reluLayer('Name', 'Relu_5')

    maxPooling2dLayer([1 timePoolSize], 'Name', 'MaxPool_4')

    dropoutLayer(dropoutProb, 'Name', 'Dropout_1')
    fullyConnectedLayer(numClasses, 'Name', 'FC_1')
    softmaxLayer('Name', 'Softmax_1')
    classificationLayer('Name', 'Classification')];


% Plot the layer graph of the network
plot(layerGraph(layers))

% Training options
miniBatchSize = 128;
valFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam',...
    'InitialLearnRate',3e-3, ...
    'MaxEpochs',25, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XVal,YVal}, ...
    'ValidationFrequency',valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',20, ...
    'ExecutionEnvironment', 'gpu');

% Train network on the GPU!
trainedNet = trainNetwork(XTrain, YTrain, layers, options);

% Get training and validation loss from trained model
YValPred = classify(trainedNet, XVal);
valError = mean(YValPred ~= YVal);
YTrainPred = classify(trainedNet, XTrain);
trainError = mean(YTrainPred ~= YTrain);
disp("Training Error: "+ trainError*100+"%")
disp("Validation Error: "+ valError*100+"%")

% # confusion matrix
figure('Units','normalized','Position',[0.2 0.2 0.5 0.5]);
cm = confusionchart(YVal,YValPred);
cm.Title = 'Confusion Matrix for Validation Data';
cm.ColumnSummary = 'column-normalized';
cm.RowSummary = 'row-normalized';
sortClasses(cm, [commands,"unknown","background"])


% ## Visualize Early Conv Layer features - nice and simple shapes
chans = 1:12
I = deepDreamImage(trainedNet, 2, chans, 'PyramidLevel', 1);
figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'])

% ## Visualize Late Conv Layer - very complicated shapes! textures really.
chans = 1:48
I = deepDreamImage(trainedNet, 17, chans, 'PyramidLevel', 1);
figure
I = imtile(I,'ThumbnailSize',[64 64]);
imshow(I)
title(['Layer ',name,' Features'])

%% Markdown Cell:
% # Pre-Trained Resnet
% Residual networks (Resnets) consist of a series of "residual blocks" of layers, and bypass connections to skip these layers. The idea of these types of networks is that each residual block can correct for the error in the calculation of the previous res block, allowing the network to operate on errors rather than full results. Resnets generally show improved performance over traditional convolutional networks.
% 
% This section also utilizes transfer learning, making use of a model that has already been trained to recognize images from the Image Net dataset. While spectrograms are very different from real-world images, the early layers of the network should be similar, capturing low-level features such as edges. In order to re-train part of this network, the final layer is replaced to get a 10 class classifier for our dataset. The network is then trained as if the weights had been randomly initialized. One way this process could be improved is to set a variable learning rate so that early layers in the network adjust slower than the final layers of the network whose weights are largely useless for our purposes.
% 
% In order to use the pre-trained resnets that Mathworks provides, input images must be 224x224 pixels. We must then resize our spectrograms to be fed into this network. The network also expects color images. We add color to our spectrogram representations using the `pcolor()` function. Research has shown (CITE THIS) that adding color can actually improve network performance. Reprocess spectrograms as images with same size as pretrained resnet (WHYYYYY doesn't matlab handle varying input sizes for you???)
% 
% This method achieves 95.86% accuracy on the same validation set as the previous conv net. This is very similar in terms of performance, however the pre-trained resnet is able to reach this accuracy in only 3 epochs.

%% Markdown Cell:
% ![performance](pres/resnet_3_epochs.png)

%% Code Cell[22]:

load("resnet")

%% Code Cell[23]:

%XValIms = specs2Ims(XVal, [224 224]);
XTrainIms = specs2Ims(XTrain, [224 224]);

%% Code Cell[24]:

save("resnet", '-v7.3')

%% Code Cell[55]:

net = resnet18;
inputSize = net.Layers(1).InputSize

%% Markdown Cell:
% Use Matlab [example](https://www.mathworks.com/help/deeplearning/examples/train-deep-learning-network-to-classify-new-images.html) supporting function to get last two layers of network

%% Code Cell[56]:

lgraph = layerGraph(net);
[learnableLayer,classLayer] = findLayersToReplace(lgraph);

%% Code Cell[57]:

numClasses = 10;
newLearnableLayer = fullyConnectedLayer(numClasses, ...
    'Name','new_fc', ...
    'WeightLearnRateFactor',10, ...
    'BiasLearnRateFactor',10);

%% Markdown Cell:
% Replace the last pre-trained layer and classification with untrained layers based on the number of classes we require

%% Code Cell[58]:

lgraph = replaceLayer(lgraph,learnableLayer.Name,newLearnableLayer);
newClassLayer = classificationLayer('Name','new_classoutput');
lgraph = replaceLayer(lgraph,classLayer.Name,newClassLayer);

%% Markdown Cell:
% Optionally, freeze initial layers of resnet here (kind of complicated compared to fastai)

%% Markdown Cell:
% Use same optimizer as before to train

%% Code Cell[59]:

miniBatchSize = 128;
valFrequency = floor(numel(YTrain)/miniBatchSize);
options = trainingOptions('adam',...
    'InitialLearnRate',3e-4, ...
    'MaxEpochs',3, ...
    'MiniBatchSize',miniBatchSize, ...
    'Shuffle','every-epoch', ...
    'Plots','training-progress', ...
    'Verbose',false, ...
    'ValidationData',{XValIms,YVal}, ...
    'ValidationFrequency',valFrequency, ...
    'LearnRateSchedule','piecewise', ...
    'LearnRateDropFactor',0.1, ...
    'LearnRateDropPeriod',2, ...
    'ExecutionEnvironment', 'gpu');

%% Code Cell[60]:

trainNetwork(XTrainIms, YTrain, lgraph, options)

%% Markdown Cell:
% Fin.
