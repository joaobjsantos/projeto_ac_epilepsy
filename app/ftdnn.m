function [ftdnn_net, testIdx] = ftdnn(X, T, type, balance_seizure_num, useCPUGPU)
    ftdnn_net = timedelaynet(1:2, 10);

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

    X = num2cell(X, 1);
    T = num2cell(T, 1);

    XTrain = [X(trainIdx(trainInd)), X(valIdx), X(testIdx)];
    TTrain = [T(trainIdx(trainInd)), T(valIdx), T(testIdx)];

    ffn_net.divideFcn = 'divideind';
    ffn_net.divideParam.trainInd = trainIdx(trainInd);
    ffn_net.divideParam.valInd = valIdx;
    ffn_net.divideParam.testInd = testIdx;
    
    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net, XTrain, TTrain, {}, ew);

    if useCPUGPU
        ftdnn_net = train(ftdnn_net, Xs, Ts, Xi, Ai, ew, 'useParallel', 'yes');
    else
        ftdnn_net = train(ftdnn_net, Xs, Ts, Xi, Ai, ew);
    end
end