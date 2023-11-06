function train_lstm()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T_categorical.mat");
    X = num2cell(X,1); 
    input_number = length(X);
    [idxTrain,idxTest] = trainingPartitions(input_number, [0.9 0.1]);
    XTrain = X(idxTrain);
    TTrain = T(idxTrain);
    XTest = X(idxTest);
    TTest = T(idxTest);
    numHiddenUnits = 100;
    numClasses = 3;
    numFeatures = 29;
    minibatchsize = 2048; 
    layers = [ ...
     sequenceInputLayer(numFeatures)
     bilstmLayer(numHiddenUnits,OutputMode="last")
     fullyConnectedLayer(numClasses)
     softmaxLayer
     classificationLayer];
    options = trainingOptions("adam", ...
     InitialLearnRate=0.002,...
     ExecutionEnvironment="gpu",...
     MaxEpochs=150, ...
     MiniBatchSize=minibatchsize, ...
     Shuffle="never", ...
     GradientThreshold=1, ...
     Verbose=false, ...
     Plots="training-progress");
    lstm_net = trainNetwork(XTrain,TTrain,layers,options);
    YTest = classify(lstm_net,XTest,MiniBatchSize=minibatchsize, ExecutionEnvironment="gpu");
    acc = mean(mean(YTest == TTest))
    save lstm_net.mat lstm_net
    save XTest.mat XTest
    save TTest.mat TTest
end