function train_ftdnn_ew_enc()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    %enc = importdata("autoenc.mat");
    X = num2cell(X,1);
    T = num2cell(T,1);

    ftdnn_net_ew_enc = timedelaynet(1:2, 10);
    input_number = length(X);
    [trainInd, valInd, testInd] = divideblock(input_number, 0.70, 0.15, 0.15);

    TTrain = T(trainInd);

    TTrain_matrix = cell2mat(TTrain);
    
    trainInterictalInd = find(all(TTrain_matrix == [1; 0; 0]));
    trainPreictalInd = find(all(TTrain_matrix == [0; 1; 0]));
    trainIctalInd = find(all(TTrain_matrix == [0; 0; 1]));

    ictal_ratio = (length(trainPreictalInd) + length(trainIctalInd))/length(trainInterictalInd);
    
    disp(ictal_ratio);

    ew = ones(1,length(trainInd));
    ew(trainInterictalInd) = ictal_ratio;

    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net_ew_enc, ...
        X(trainInd),T(trainInd),{}, ew);
    
    ftdnn_net_ew_enc.divideFcn = 'divideind';
    ftdnn_net_ew_enc.divideParam.trainInd = trainInd;
    ftdnn_net_ew_enc.divideParam.valInd = valInd;
    ftdnn_net_ew_enc.divideParam.testInd = testInd;

    ftdnn_net_ew_enc.trainParam.epochs = 100;

    ftdnn_net_ew_enc = train(ftdnn_net_ew_enc, Xs, Ts, Xi, Ai);

    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net_ew_enc,X(testInd),T(testInd));
    Y = ftdnn_net_ew_enc(Xs, Xi);
    e = gsubtract(Y,Ts);
    rmse = sqrt(mse(e))
    save ftdnn_net_ew_enc.mat ftdnn_net_ew_enc
    save testInd.mat testInd
end