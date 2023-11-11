function train_lrn_balanced()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    X = num2cell(X,1);
    T = num2cell(T,1);
    lrn_net_balanced = layrecnet(1:2, 10, "trainscg");
    input_number = length(X);
    [trainInd,valInd,testInd] = divideblock(input_number, 0.80, 0.1, 0.1);

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

    [Xs,Xi,Ai,Ts] = preparets(lrn_net_balanced,X_balanced,T_balanced);

    lrn_net_balanced.divideFcn = 'divideind';
    lrn_net_balanced.divideParam.trainInd = 1:length(trainInd_balanced);
    lrn_net_balanced.divideParam.valInd = length(trainInd_balanced)+1:length(trainInd_balanced)+length(valInd);
    lrn_net_balanced.divideParam.testInd = length(trainInd_balanced)+length(valInd)+1:length(X_balanced);
    [lrn_net_balanced, tr] = train(lrn_net_balanced, Xs, Ts, Xi, Ai, ...
        'useParallel', 'yes', 'useGpu', 'yes');
    save lrn_net_balanced.mat lrn_net_balanced
    save tr.mat tr
end