function bandEnergy = computeEnergy(margSpect,startFreq,endFreq,resolution)
        %Calculate frequency start and end index
        hftIndexStart = round(startFreq/resolution)+1;
        hftIndexEnd = round(endFreq/resolution)+1;
        %Calculate Band Energy
        bandEnergy = log(sum(margSpect(hftIndexStart:hftIndexEnd,:).^2));
end