function train_lrn()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    X = num2cell(X,1); 
    T = num2cell(T,1);
    lrn_net = layrecnet(1, 10, "trainscg");
    [Xs,Xi,Ai,Ts] = preparets(lrn_net,X,T);
    input_number = length(X);
    [trainInd,valInd,testInd] = divideblock(input_number, 0.70, 0.15, 0.15);
    lrn_net.divideFcn = 'divideind';
    lrn_net.divideParam.trainInd = trainInd;
    lrn_net.divideParam.valInd = valInd;
    lrn_net.divideParam.testInd = testInd;
    [lrn_net, tr] = train(lrn_net, Xs, Ts, Xi, Ai, ...
        'useParallel', 'yes', 'useGpu', 'yes');
    save lrn_net.mat lrn_net
    save tr.mat tr
end