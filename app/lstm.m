function [lstm_net, testIdx] = lstm(X, T, type, balance_seizure_num, useCPUGPU)
    numClasses = 3;
    numFeatures = size(X,1);
    classes = categories(T);
    input_number = length(T);

    if balance_seizure_num
        [trainIdx, valIdx, testIdx] = balance_seizures(T, 0.7, 0.15, 0.15, true);
    else
        %[trainIdx,testIdx,~] = splitIdx(input_number,0.7, 0.15, 0.15);
        [trainIdx, valIdx, testIdx] = trainingPartitions(input_number, [0.7, 0.15, 0.15]);
    end
    

    if type == "detect"
        weights = [1 1 5];
    elseif type == "predict"
        weights = [1 5 1];
    else
        weights = 0;
        disp("Wrong type");
    end

    XTrain = X(:, trainIdx);
    TTrain = T(trainIdx);

    [trainBalancedIdx, class_weights, ~] = class_balancing(T, trainIdx, true, weights, true);

    XTrain = num2cell(XTrain,1);
    XTrain = XTrain(trainBalancedIdx);
    TTrain = TTrain(trainBalancedIdx);

    layers = [ ...
     sequenceInputLayer(numFeatures)
     lstmLayer(100,OutputMode="last")
     fullyConnectedLayer(numClasses)
     softmaxLayer
     classificationLayer(Classes=classes, ClassWeights=class_weights)];

    if useCPUGPU
        options = trainingOptions("adam", ...
         ExecutionEnvironment="gpu",...
         ValidationData={num2cell(X(:,valIdx),1), T(valIdx)}, ...
         Plots="training-progress");
    else
        options = trainingOptions("adam", ...
         ValidationData={num2cell(X(:,valIdx),1), T(valIdx)}, ...
         Plots="training-progress");
    end 
    
    lstm_net = trainNetwork(XTrain,TTrain,layers,options);
end