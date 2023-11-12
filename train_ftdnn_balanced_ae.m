function train_ftdnn_balanced_ae()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    enc = importdata("autoenc.mat");
    X = num2cell(X,1);
    T = num2cell(T,1);
    
    ftdnn_net_balanced_ae = timedelaynet(1:2, 10);
    input_number = length(X);
    [trainInd,valInd,testInd] = divideblock(input_number, 0.90, 0.05, 0.05);

    TTrain = T(trainInd);

    TTrain_matrix = cell2mat(TTrain);
    
    trainInterictalInd = find(all(TTrain_matrix == [1; 0; 0]));
    trainPreictalInd = find(all(TTrain_matrix == [0; 1; 0]));
    trainIctalInd = find(all(TTrain_matrix == [0; 0; 1]));

    trainInd_balanced = sort( ...
        [randperm(length(trainInterictalInd), length(trainPreictalInd) + length(trainIctalInd)) ...
        trainPreictalInd, trainIctalInd]);

    X_balanced = [X(trainInd_balanced), X(valInd), X(testInd)];
    T_balanced = [T(trainInd_balanced), T(valInd), T(testInd)];

    X_balanced_enc = num2cell(encode(enc, cell2mat(X_balanced)),1);
    
    disp(size(X_balanced_enc));
    disp(size(trainInd_balanced));

    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net_balanced_ae,X_balanced_enc,T_balanced);

    ftdnn_net_balanced_ae.divideFcn = 'divideind';
    ftdnn_net_balanced_ae.divideParam.trainInd = 1:length(trainInd_balanced);
    ftdnn_net_balanced_ae.divideParam.valInd = length(trainInd_balanced)+1:length(trainInd_balanced)+length(valInd);
    ftdnn_net_balanced_ae.divideParam.testInd = length(trainInd_balanced)+length(valInd)+1:length(X_balanced);
    ftdnn_net_balanced_ae = train(ftdnn_net_balanced_ae, Xs, Ts, Xi, Ai);
    [Xs,Xi,Ai,Ts] = preparets(ftdnn_net_balanced_ae,num2cell(encode(enc, cell2mat(X(testInd))),1),T(testInd));
    Y = ftdnn_net_balanced_ae(Xs, Xi);
    e = gsubtract(Y,Ts);
    rmse = sqrt(mse(e))

    save ftdnn_net_balanced_ae.mat ftdnn_net_balanced_ae
    save testInd.mat testInd
end