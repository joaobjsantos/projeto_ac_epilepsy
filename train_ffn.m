function train_ffn()
    load("FeatVectSelT.mat");
    load("T.mat");
    ffn_net = feedforwardnet(100, 'trainscg');
    [ffn_net, tr] = train(ffn_net, FeatVectSelT, T, 'useGpu', 'yes');
    save ffn_net.mat ffn_net
    save tr.mat tr
end