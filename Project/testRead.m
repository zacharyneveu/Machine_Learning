% Test reading audio files
% Tianyi Zhou 04/04/2019

%% Pre-processing

% Clean workspace
clear
clc
close all

% ---------------------------------
% Import data from folder 
% ---------------------------------
% Codes written on WINDOWS. Path should be changed under other OS.
trainDataDir = 'SampleData\train\Train';
testDataDir = 'SampleData\test\Test';

trainDataStore = audioDatastore(trainDataDir,'FileExtensions','.wav');
testDataStore = audioDatastore(testDataDir,'FileExtensions','.wav');



% --------------------------------------------------
% To sort files in natural order, sorted result 
%    saved in trainDataStore and testDataStore
% --------------------------------------------------
% Sort training data
trainFiles = cell2table(trainDataStore.Files,'VariableNames',{'FilePath'});
NUMtrain = length(trainFiles.FilePath);
ID = zeros(NUMtrain,1);

for i = 1:NUMtrain
    [~,fn,~] = fileparts(char(trainFiles.FilePath(i)));
    ID(i) = str2num(fn);
end

trainFiles = [table(ID) trainFiles];
[trainFiles,changedIdx] = sortrows(trainFiles);
trainDataStore.Files = trainDataStore.Files(changedIdx); % This step clears the imported Labels in DS...

% Sort test data
testFiles = cell2table(testDataStore.Files,'VariableNames',{'FilePath'});
NUMtest = length(testFiles.FilePath);
ID = zeros(NUMtest,1);

for i = 1:NUMtest
    [~,fn,~] = fileparts(char(testFiles.FilePath(i)));
    ID(i) = str2num(fn);
end

testFiles = [table(ID) testFiles];
[testFiles,changedIdx] = sortrows(testFiles);
testDataStore.Files = testDataStore.Files(changedIdx);

% Import labels after sorting
trainLabelDir = 'SampleData\train\train.csv';
trainLabel = readtable(trainLabelDir);
trainDataStore.Labels = categorical(trainLabel.Class);


%% Read audio files + Feature Extraction

%Approach 1: Involving transpose operation
% tic
% 
% SampleRate = [];
% AudioFiles = {};
% Loudness = {};
% Pitch = {};
% HarmonicRatio = {};
% MFCC = {};
% 
% while hasdata(trainDataStore)
%     [audioIn, Info] = read(trainDataStore);
%     
%     SR = Info.SampleRate;
%     SampleRate(end+1) = SR;
%     AudioFiles(end+1) = {audioIn};
%     fprintf('Fraction of files read: %.2f\n',progress(trainDataStore))
%     
%     Loudness(end+1) = {integratedLoudness(audioIn,SR)};
%     Pitch(end+1) = {pitch(audioIn,SR)};
%     HarmonicRatio(end+1) = {harmonicRatio(audioIn,SR)};
%     MFCC(end+1) = {mfcc(audioIn,SR)};
% end 
% 
% SampleRate = SampleRate';
% AudioFiles = AudioFiles';
% Loudness = Loudness';
% Pitch = Pitch';
% HarmonicRatio = HarmonicRatio';
% MFCC = MFCC';
% 
% trainFiles = [trainFiles table(AudioFiles) table(SampleRate) ...
%     table(Loudness) table(Pitch) table(HarmonicRatio) table(MFCC)];
% 
% toc

% Approach 2: Without transpose
tic

SampleRate = zeros(NUMtrain,1);
AudioFiles = cell(NUMtrain,1);
Loudness = cell(NUMtrain,1);
Pitch = cell(NUMtrain,1);
HarmonicRatio = cell(NUMtrain,1);
MFCC = cell(NUMtrain,1);

for i=1:NUMtrain
    [audioIn, Info] = read(trainDataStore);
    
    SR = Info.SampleRate;
    SampleRate(i) = SR;
    AudioFiles(i) = {audioIn};
    fprintf('Fraction of files read: %.2f\n',progress(trainDataStore))
    
    Loudness(i) = {integratedLoudness(audioIn,SR)};
    Pitch(i) = {pitch(audioIn,SR)};
    HarmonicRatio(i) = {harmonicRatio(audioIn,SR)};
    MFCC(i) = {mfcc(audioIn,SR)};
end 
trainFiles = [trainFiles table(AudioFiles) table(SampleRate) ...
    table(Loudness) table(Pitch) table(HarmonicRatio) table(MFCC)];

toc

