function [XCNN, TCNN, classWeights] = prepareCNNInputs(X, T, numFeatures)
    interictalInd = find(T == "Interictal");
    preictalInd = find(T == "Preictal");
    ictalInd = find(T == "Ictal");

    interictal = length(interictalInd);
    preictal = length(preictalInd);
    ictal = length(ictalInd);
    total = interictal + preictal + ictal;

    X = mat2gray(cell2mat(X));

    XCNN = zeros(numFeatures, numFeatures, 1, floor(total/29));
    TCNN = strings(floor(total/29));

    i = 1;

    %INTERICTAL
    for j = 1:numFeatures:interictal-28
        XCNN(:,:,1,i) = X(:,interictalInd(j:j+28));
        TCNN(i) = "Interictal";
        i = i + 1;
    end
    
    %PREICTAL
    for j = 1:numFeatures:preictal-28
        XCNN(:,:,1,i) = X(:,preictalInd(j:j+28));
        TCNN(i) = "Preictal";
        i = i + 1;
    end

    %ICTAL
    for j = 1:numFeatures:ictal-28
        XCNN(:,:,1,i) = X(:,ictalInd(j:j+28));
        TCNN(i) = "Ictal";
        i = i + 1;
    end

    XCNN = XCNN(:,:,1,1:i-1);
    TCNN = categorical(TCNN(1:i-1));
    
    classWeights = [interictal/total, preictal/total, ictal/total]
    % classWeights = [total/interictal, total/preictal, total/ictal];
    % classWeights = classWeights./sum(classWeights)
end
    