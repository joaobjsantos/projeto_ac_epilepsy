function train_lstm_weighted()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T_categorical.mat");
    X = num2cell(X,1); 
    input_number = length(X);
    [idxTrain,idxVal, idxTest] = trainingPartitions(input_number, [0.7 0.15 0.15]);
    XTrain = X(idxTrain);
    TTrain = T(idxTrain);
    XVal = X(idxVal);
    TVal = T(idxVal);
    XTest = X(idxTest);
    TTest = T(idxTest);
    numHiddenUnits = 100;
    numClasses = 3;
    numFeatures = 29;
    minibatchsize = 2048; 
    classes = unique(T', "rows")';

    trainInterictal = sum(TTrain == "Interictal");
    trainPreictal = sum(TTrain == "Preictal");
    trainIctal = sum(TTrain == "Ictal");
    trainTotal = trainInterictal + trainPreictal + trainIctal;

    classWeights = [trainInterictal/trainTotal
                    trainPreictal/trainTotal
                    trainIctal/trainTotal];

    layers = [ ...
     sequenceInputLayer(numFeatures)
     lstmLayer(numHiddenUnits,OutputMode="last")
     fullyConnectedLayer(numClasses)
     softmaxLayer
     classificationLayer(Classes=classes, classWeights=classWeights)];

    options = trainingOptions("adam", ...
     InitialLearnRate=0.002,...
     ExecutionEnvironment="gpu",...
     MaxEpochs=150, ...
     ValidationData={XVal, TVal}, ...
     MiniBatchSize=minibatchsize, ...
     Shuffle="never", ...
     GradientThreshold=1, ...
     Verbose=false, ...
     Plots="training-progress");
    lstm_weighted_net = trainNetwork(XTrain,TTrain,layers,options);
    YTest = classify(lstm_weighted_net,XTest,MiniBatchSize=minibatchsize, ExecutionEnvironment="gpu");
    acc = mean(mean(YTest == TTest))
    save lstm_weighted_net.mat lstm_weighted_net
    save XTest.mat XTest
    save TTest.mat TTest
end