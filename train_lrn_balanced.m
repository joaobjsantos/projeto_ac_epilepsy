function train_lrn_balanced()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    X = num2cell(X,1); 
    T = num2cell(T,1);
    lrn_net = layrecnet(1:2, 10, "trainscg");
    [Xs,Xi,Ai,Ts] = preparets(lrn_net,X,T);
    input_number = length(X);
    [trainInd,valInd,testInd] = divideblock(input_number, 0.70, 0.15, 0.15);

    TTrain = T(trainInd);

    classes = unique(cell2mat(T)', "rows")';

    TTrain_matrix = cell2mat(TTrain);

    trainInterictal = sum(all(TTrain_matrix == [1; 0; 0]));
    trainPreictal = sum(all(TTrain_matrix == [0; 1; 0]));
    trainIctal = sum(all(TTrain_matrix == [0; 0; 1]));
    trainTotal = trainInterictal + trainPreictal + trainIctal;

    trainInd_balanced = sort(randperm(length(trainInd), trainPreictal + trainIctal));

    lrn_net.divideFcn = 'divideind';
    lrn_net.divideParam.trainInd = trainInd_balanced(1:100);
    lrn_net.divideParam.valInd = valInd(1:100);
    lrn_net.divideParam.testInd = testInd(1:100);
    [lrn_net, tr] = train(lrn_net, Xs, Ts, Xi, Ai, ...
        'useParallel', 'yes', 'useGpu', 'yes');
    save lrn_net.mat lrn_net
    save tr.mat tr
end