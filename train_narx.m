function train_narx()
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    X = num2cell(X,1); 
    T = num2cell(T,1);
    input_number = length(X);
    [trainInd,valInd,testInd] = divideblock(input_number, 0.70, 0.15, 0.15);
    narx_net = narxnet(1:2, 1:2, 10);
    [Xs,Xi,Ai,Ts] = preparets(narx_net,X(trainInd), {}, T(trainInd));
    % narx_net.divideFcn = 'divideind';
    % narx_net.divideParam.trainInd = trainInd;
    % narx_net.divideParam.valInd = valInd;
    % narx_net.divideParam.testInd = testInd;
    [narx_net, tr] = train(narx_net, Xs, Ts, Xi, Ai, 'useParallel', 'yes');
    [Y,Xf,Af] = net(Xs,Xi,Ai);
    perf = perform(net,Ts,Y)
    [netc,Xic,Aic] = closeloop(narx_net,Xf,Af);
    Yc = netc(XPredict,Xic,Aic)
    save Yc.mat Yc
    save narx_net.mat narx_net
    save tr.mat tr
end