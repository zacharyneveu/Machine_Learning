% First Trial
% Tianyi Zhou 04/03/2019

%% Pre-processing

% Clean workspace
clear
clc
close all

% Load data test
%[y,Fs] = audioread('0.wav');

% ---------------------------------
% Import data from folder 
% ---------------------------------
% Codes written on WINDOWS. Path should be changed under other OS.
trainDataDir = 'RAWDATAunzipped\train\Train';
trainLabelDir = 'RAWDATAunzipped\train\train.csv';
testDataDir = 'RAWDATAunzipped\test\Test';

trainds = audioDatastore(trainDataDir,'FileExtensions','.wav');
testds = audioDatastore(testDataDir,'FileExtensions','.wav');

trainLabel = readtable(trainLabelDir);
trainds.Labels = categorical(trainLabel.Class);

% --------------------------------------------------
% To sort files in natural order, sorted result 
%    saved in trainds and testds
% --------------------------------------------------
% Sort training data
trainFiles = cell2table(trainds.Files,'VariableNames',{'FilePath'});
NUMtrain = length(trainFiles.FilePath);
ID = zeros(NUMtrain,1);

for i = 1:NUMtrain
    [~,fn,~] = fileparts(char(trainFiles.FilePath(i)));
    ID(i) = str2num(fn);
end

trainFiles = [table(ID) trainFiles];
[trainFiles,changedIdx] = sortrows(trainFiles);
trainds.Files = trainds.Files(changedIdx);

% Sort test data
testFiles = cell2table(testds.Files,'VariableNames',{'FilePath'});
NUMtest = length(testFiles.FilePath);
ID = zeros(NUMtest,1);

for i = 1:NUMtest
    [~,fn,~] = fileparts(char(testFiles.FilePath(i)));
    ID(i) = str2num(fn);
end

testFiles = [table(ID) testFiles];
[testFiles,changedIdx] = sortrows(testFiles);
testds.Files = testds.Files(changedIdx);
