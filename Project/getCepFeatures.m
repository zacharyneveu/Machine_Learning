function normedFeats = getCepFeatures(data, nFeats)
    specFeats = zeros(size(data,1),nFeats);
    for i=1:size(data,1)
        item = data{i}(:,1);
            specFeats(i,:) = gtcc(item, 48000, 'WindowLength', size(item,1), 'NumCoeffs', nFeats-1, 'LogEnergy', 'Append');
            if mod(i,100) == 0
                display(strcat("Progress: ", num2str(i)))
            end
    end
    normedFeats = zscore(specFeats)
end