% André Filipe de Oliveira Moreira Nº 2020239416, PL3
% João Bernardo de Jesus Santos  Nº 2020218995, PL3
% Eliseu António Domingos Nº 2023174914, PL3  
% 
% This function trains a FeedForward Neural Network for detecting or
% predicting seizures(depending on type)


function [ffn_net, testIdx] = ffn(X, T, type, balance_seizure_num, useCPUGPU)
    ffn_net = feedforwardnet([50 20], 'trainscg');
    %ffn_net = feedforwardnet([20 10], 'trainscg');

    input_number = length(X);

    if balance_seizure_num
        [trainIdx,valIdx,testIdx] = balance_seizures(T, 0.70, 0.15, 0.15, false);
    else
        [trainIdx,valIdx,testIdx] = splitIdx(input_number, 0.70, 0.15, 0.15);
    end
    

    if type == "detect"
        weights = [1 1 10];
    elseif type == "predict"
        weights = [1 10 1];
    else
        weights = 0;
        disp("Wrong type");
    end

    [trainInd, ~, ew] = class_balancing(T, trainIdx, true, weights, false);
    
    XTrain = [X(:,trainIdx(trainInd)), X(:, valIdx), X(:, testIdx)];
    TTrain = [T(:,trainIdx(trainInd)), T(:, valIdx), T(:, testIdx)];

    ffn_net.divideFcn = 'divideind';
    ffn_net.divideParam.trainInd = trainIdx(trainInd);
    ffn_net.divideParam.valInd = valIdx;
    ffn_net.divideParam.testInd = testIdx;

    if useCPUGPU
        ffn_net = train(ffn_net, XTrain, TTrain, [], [], ew, 'useParallel', 'yes', 'useGpu', 'yes');
    else
        ffn_net = train(ffn_net, XTrain, TTrain, [], [], ew);
    end
end