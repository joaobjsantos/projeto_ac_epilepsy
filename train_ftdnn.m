function train_ftdnn()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    % enc = importdata("autoenc.mat");
    X = num2cell(X,1);
    T = num2cell(T,1);

    ftdnn_net = timedelaynet(1:2, 10);
    input_number = length(X);
    [trainInd, valInd, testInd] = divideblock(input_number, 0.70, 0.15, 0.15);

    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net,X(trainInd),T(trainInd));
    
    ftdnn_net.divideFcn = 'divideind';
    ftdnn_net.divideParam.trainInd = trainInd;
    ftdnn_net.divideParam.valInd = valInd;
    ftdnn_net.divideParam.testInd = testInd;

    ftdnn_net.trainParam.epochs = 100;

    ftdnn_net = train(ftdnn_net, Xs, Ts, Xi, Ai);

    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net,X(testInd),T(testInd));
    Y = ftdnn_net(Xs, Xi);
    e = gsubtract(Y,Ts);
    rmse = sqrt(mse(e))
    save ftdnn_net.mat ftdnn_net
    save testInd.mat testInd
end