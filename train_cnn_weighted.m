function train_cnn_weighted()
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
    numClasses = 3;
    numFeatures = 29;
    minibatchsize = 2048; 
    classes = unique(T', "rows")';
    
    [XTrainCNN, TTrainCNN, classWeights] = prepareCNNInputs(XTrain, TTrain, numFeatures);
    [XValCNN, TValCNN, ~] = prepareCNNInputs(XVal, TVal, numFeatures);
    
    layers = [ ...
     imageInputLayer([numFeatures, numFeatures, 1])
     convolution2dLayer(5, 20)
     reluLayer
     fullyConnectedLayer(numClasses)
     softmaxLayer
     classificationLayer(Classes=classes, classWeights=classWeights)];

    options = trainingOptions("adam", ...
     InitialLearnRate=0.002,...
     ExecutionEnvironment="gpu",...
     MaxEpochs=100, ...
     ValidationData={XValCNN, TValCNN}, ...
     MiniBatchSize=minibatchsize, ...
     Shuffle="never", ...
     GradientThreshold=1, ...
     Verbose=false, ...
     Plots="training-progress");

    cnn_weighted_net = trainNetwork(XTrainCNN,TTrainCNN,layers,options);

    [XTestCNN, TTestCNN, ~] = prepareCNNInputs(XTest, TTest, numFeatures);
    YTestCNN = classify(cnn_weighted_net,XTestCNN,MiniBatchSize=minibatchsize, ExecutionEnvironment="gpu");
    acc = mean(mean(YTestCNN == TTestCNN))

    save cnn_weighted_net.mat cnn_weighted_net
    save XTest.mat XTestCNN
    save TTest.mat TTestCNN
end