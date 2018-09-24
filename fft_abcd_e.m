clear all;
close all;

%System Requirements
%MATLAB 2018a - Signal Processing Toolbox
%Download the dataset from the link [3]
for lp=1:10
driveletter = {'Z/Z','O/O','F/F','N/N'};

for subset = 1:4
    %Iterate through the datasets
    for dataset = 1:100
        
        %Load the dataset for Non-Seizure Data
        data = load(strcat(driveletter{subset},sprintf('%03d', dataset),'.txt'));
        
        sampleRate = 173.61;
        len = 512;
        NFFT1 =(2^(nextpow2(len)));
        
        %Computing Fourier Spectrum
        signalFft = fft(data,NFFT1)/len;
        signalFftMagClass2(1:NFFT1/2,dataset)= 2*abs(signalFft(1:NFFT1/2));
        
        %Computing the Probability distribution
        signalFftprob(1:NFFT1/2)=(signalFftMagClass2(1:NFFT1/2,dataset).^2)./sum((signalFftMagClass2(1:NFFT1/2,dataset).^2));
        
        fft_axis =(0:NFFT1/2-1)*(sampleRate/NFFT1);
        
        %Shannon-Entropy
        SEN_non(subset,dataset) = -1*sum((signalFftprob).*log(signalFftprob));
        
        %Renyi-Entropy
        alpha=2;
        REN_non(subset,dataset) = (1/(1-alpha))*log(sum(signalFftprob.^alpha));
        
        %Tsallis-Entropy
        TEN_non(subset,dataset) = (1/(alpha-1))*(1-(sum(signalFftprob.^alpha)));
        
        %Compute Energy Features
        resolution = sampleRate/len;
        
        e1_non(subset,dataset) = computeEnergy(signalFftMagClass2(:,dataset),0,4,resolution);
        e2_non(subset,dataset) = computeEnergy(signalFftMagClass2(:,dataset),4,8,resolution);
        e3_non(subset,dataset) = computeEnergy(signalFftMagClass2(:,dataset),8,12,resolution);
        e4_non(subset,dataset) = computeEnergy(signalFftMagClass2(:,dataset),12,30,resolution);
        e5_non(subset,dataset) = computeEnergy(signalFftMagClass2(:,dataset),30,50,resolution);
    end
end


for dataset = 1:100
    %Load the dataset for Seizure Data
    data = load(strcat('S/S',sprintf('%03d', dataset),'.txt'));
    
    sampleRate = 173.61;
    len = 512;
    NFFT1 =(2^(nextpow2(len)));
    
    %Computing Fourier Spectrum
    signalFft = fft(data,NFFT1)/len;
    signalFftMagClass2(1:NFFT1/2,dataset)= 2*abs(signalFft(1:NFFT1/2));
    
    %Computing the Probability distribution
    signalFftprob(1:NFFT1/2)=(signalFftMagClass2(1:NFFT1/2,dataset).^2)./sum((signalFftMagClass2(1:NFFT1/2,dataset).^2));
    
    fft_axis =(0:NFFT1/2-1)*(sampleRate/NFFT1);
    
    %Shannon-Entropy
    SEN_sez(dataset) = -1*sum((signalFftprob).*log(signalFftprob));
    
    %Renyi-Entropy
    alpha = 2;
    REN_sez(dataset) = (1/(1-alpha))*log(sum(signalFftprob.^alpha));
    
    %Tsallis-Entropy
    TEN_sez(dataset) = (1/(alpha-1))*(1-(sum(signalFftprob.^alpha)));
    
    %Compute Energy Features
    resolution = sampleRate/len;
    e1_sez(dataset) = computeEnergy(signalFftMagClass2(:,dataset),0,4,resolution);
    e2_sez(dataset) = computeEnergy(signalFftMagClass2(:,dataset),4,8,resolution);
    e3_sez(dataset) = computeEnergy(signalFftMagClass2(:,dataset),8,12,resolution);
    e4_sez(dataset) = computeEnergy(signalFftMagClass2(:,dataset),12,30,resolution);
    e5_sez(dataset) = computeEnergy(signalFftMagClass2(:,dataset),30,50,resolution);
end

%Feature Aggregation
features_non = full([SEN_non(:),REN_non(:),TEN_non(:),e1_non(:),e2_non(:),e3_non(:),e4_non(:)]);
features_sez = [SEN_sez;REN_sez;TEN_sez;e1_sez;e2_sez;e3_sez;e4_sez]';

features = [features_non;features_sez];
labels = [ones(1,length(features_non)),-1*ones(1,length(features_sez))]';

%SVM Classification
cl_svm = fitcsvm(features,labels,'KernelFunction','rbf','ClassNames',[1,-1],'BoxConstraint',71.31,'KernelScale',13.72);

%10-fold Cross-validation
svm_models = crossval(cl_svm);
accuracy(lp) = (1-kfoldLoss(svm_models))*100;
end
accuracyFFT = mean(accuracy)
