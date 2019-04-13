function X = spectrograms(ads,segmentDuration,frameDuration,hopDuration,numBands)

disp("Computing speech spectrograms...");

numHops = ceil((segmentDuration - frameDuration)/hopDuration);
numFiles = length(ads.Files);
X = zeros([numBands,numHops+1,1,numFiles],'single');

for i = 1:numFiles
    
    [x,info] = read(ads);
    
    fs = info.SampleRate;
    frameLength = round(frameDuration*fs);
    hopLength = round(hopDuration*fs);
    
	try
		spec = melSpectrogram(x,fs, ...
			'WindowLength',frameLength, ...
			'OverlapLength',frameLength - hopLength, ...
			'NumBands',numBands, ...
			'FrequencyRange',[50,20000]);

		% If the spectrogram is less wide than numHops, then put spectrogram in
		% the middle of X.
		w = size(spec,2);
		left = floor((numHops-w)/2)+1;
		ind = left+1:left+w;
		%ind
		% Take only 1st channel if multiple
		%size(spec)
		%size(X)
		X(:,ind,1,i) = spec(:,:,1);

	catch e
		disp("Failed for file "+i)
		disp(info)
		disp(e)
		disp(size(x))
		continue
	end

    
    if mod(i,100) == 0
        disp("Processed " + i + " files out of " + numFiles)
    end
    
end

end
