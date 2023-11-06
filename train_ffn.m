function train_ffn()
    load("FeatVectSelT.mat");
    load("T.mat");
    ffn_net = feedforwardnet([100], 'trainscg');
    input_number = length(FeatVectSelT);
    [trainInd,valInd,testInd] = divideblock(input_number, 0.70, 0.15, 0.15);
    ffn_net.divideFcn = 'divideind';
    ffn_net.divideParam.trainInd = trainInd;
    ffn_net.divideParam.valInd = valInd;
    ffn_net.divideParam.testInd = testInd;
    [ffn_net, tr] = train(ffn_net, FeatVectSelT, T, 'useParallel', 'yes', 'useGpu', 'yes');
    save ffn_net.mat ffn_net
    save tr.mat tr
end