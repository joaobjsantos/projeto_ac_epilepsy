function [XCNN, TCNN] = prepare_cnn(X, T, num_features)
    interictalInd = find(T == "Interictal");
    preictalInd = find(T == "Preictal");
    ictalInd = find(T == "Ictal");

    interictal = length(interictalInd);
    preictal = length(preictalInd);
    ictal = length(ictalInd);
    total = interictal + preictal + ictal;

    X = mat2gray(cell2mat(X));

    XCNN = zeros(num_features, num_features, 1, floor(total/29));
    TCNN = strings(floor(total/29));

    i = 1;

    %INTERICTAL
    for j = 1:num_features:interictal-num_features-1
        XCNN(:,:,1,i) = X(:,interictalInd(j:j+28));
        TCNN(i) = "Interictal";
        i = i + 1;
    end
    
    %PREICTAL
    for j = 1:num_features:preictal-num_features-1
        XCNN(:,:,1,i) = X(:,preictalInd(j:j+28));
        TCNN(i) = "Preictal";
        i = i + 1;
    end

    %ICTAL
    for j = 1:num_features:ictal-num_features-1
        XCNN(:,:,1,i) = X(:,ictalInd(j:j+28));
        TCNN(i) = "Ictal";
        i = i + 1;
    end

    XCNN = XCNN(:,:,1,1:i-1);
    TCNN = categorical(TCNN(1:i-1));
end
    