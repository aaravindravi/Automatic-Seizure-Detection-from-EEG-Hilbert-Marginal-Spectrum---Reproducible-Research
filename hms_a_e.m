%System Requirements
%MATLAB 2018a - Signal Processing Toolbox
%Download the dataset from the link [3]

clear all;
close all;

%Number of IMFs to use
for comp = 1:5
    %Iterate through the datasets
    for dataset = 1:100
    
        %Load the dataset for Non-Seizure Data
        data = load(strcat('Z/Z',sprintf('%03d', dataset),'.txt'));
        sampleRate = 173.61; %Hz

        %Resolution
        len=512;
        resolution=sampleRate/512;
        
        %Empirical Mode Decomposition
        imf = emd(data); 
        
        %Hilbert-Huang Transform
        [HSpect,F(:,dataset),T] = hht(imf(:,1:comp),sampleRate,'FrequencyResolution',resolution); 
        
        %Computing the Marginal Spectrum
        margSpect(:,dataset) = sum(HSpect,2); 
        
        %Computing the Probability distribution
        margSpectEner(:,dataset) = sum(HSpect.^2,2);
        marSpectProb(:,dataset) = margSpectEner(:,dataset)./sum(margSpectEner(:,dataset));
        
        %Shannon-Entropy 
        SEN_non(dataset) = -1*sum((marSpectProb(:,dataset)).*log(0.0001+marSpectProb(:,dataset)));
        
        alpha=2;
        %Renyi Entropy
        REN_non(dataset) = (1/(1-alpha))*log(sum(marSpectProb(:,dataset).^alpha));
        
        %Tsallis Entropy
        TEN_non(dataset) = (1/(alpha-1))*(1-(sum(marSpectProb(:,dataset).^alpha)));
        
        %Compute Energy Features
        e1_non(dataset) = computeEnergy(margSpect(:,dataset),0,4,resolution);
        e2_non(dataset) = computeEnergy(margSpect(:,dataset),4,8,resolution);
        e3_non(dataset) = computeEnergy(margSpect(:,dataset),8,12,resolution);
        e4_non(dataset) = computeEnergy(margSpect(:,dataset),12,30,resolution);
        e5_non(dataset) = computeEnergy(margSpect(:,dataset),30,50,resolution);
        
    end


    for dataset = 1:100
        %Load the dataset for Seizure Data
        data = load(strcat('S/S',sprintf('%03d', dataset),'.txt'));
        sampleRate = 173.61;

        %Empirical Mode Decomposition
        imf = emd(data);
        
        %Hilbert-Huang Transform
        [HSpect,F(:,dataset),T] = hht(imf(:,1:comp),sampleRate,'FrequencyResolution',resolution);
        
        %Computing the Marginal Spectrum
        margSpectSez(:,dataset) = sum(HSpect,2);
        
        %Computing the Probability distribution
        margSpectEnerSez(:,dataset) = sum(HSpect.^2,2);
        marSpectProbSez(:,dataset) = margSpectEnerSez(:,dataset)./sum(margSpectEnerSez(:,dataset));
        
        %Shannon-Entropy 
        SEN_sez(dataset) = -1*sum((marSpectProbSez(:,dataset)).*log(0.0001+marSpectProbSez(:,dataset)));
        
        alpha=2;
        %Renyi Entropy
        REN_sez(dataset) = (1/(1-alpha))*log(sum(marSpectProbSez(:,dataset).^alpha));
        
        %Tsallis Entropy
        TEN_sez(dataset) = (1/(alpha-1))*(1-(sum(marSpectProbSez(:,dataset).^alpha)));
        
        e1_sez(dataset) = computeEnergy(margSpectSez(:,dataset),0,4,resolution);
        e2_sez(dataset) = computeEnergy(margSpectSez(:,dataset),4,8,resolution);
        e3_sez(dataset) = computeEnergy(margSpectSez(:,dataset),8,12,resolution);
        e4_sez(dataset) = computeEnergy(margSpectSez(:,dataset),12,30,resolution);
        e5_sez(dataset) = computeEnergy(margSpectSez(:,dataset),30,50,resolution);

    end

    %Feature Aggregation
    features_non = full([SEN_non;REN_non;TEN_non;e1_non;e2_non;e3_non;e4_non])';
    features_sez = full([SEN_sez;REN_sez;TEN_sez;e1_sez;e2_sez;e3_sez;e4_sez])';

    features = [features_non;features_sez];
    labels = [ones(1,length(features_non)),-1*ones(1,length(features_sez))]';

    %SVM Classification
    cl_svm = fitcsvm(features,labels,'KernelFunction','rbf','ClassNames',[1,-1],'BoxConstraint',5.7,'KernelScale',85.36);
    
    %10-fold Cross-validation
    svm_models = crossval(cl_svm);
    accuracy(comp) = (1-kfoldLoss(svm_models))*100;
end
accuracyHMS = mean(accuracy)


figure;
boxplot([e1_non',e1_sez',e2_non',e2_sez',e3_non',e3_sez',e4_non',e4_sez',e5_non',e5_sez']);
box on
set(gca, 'XTick',[1:10],'xticklabel',{'NS';'S';'NS';'S';'NS';'S';'NS';'S';'NS';'S';});xlabel('  Delta             Theta               Alpha               Beta            Gamma');ylabel('log(EN)');