function test_deep_net_se_sp(netname, XTest, TTest)
    net = importdata(netname + ".mat");
    true_positives = 0;
    true_negatives = 0;
    false_positives = 0;
    false_negatives = 0;
    minibatchsize = 2048;
    YTest = classify(net,XTest,MiniBatchSize=minibatchsize, ExecutionEnvironment="gpu");
    for i = 1:length(TTest)
        if TTest(i) == "Interictal"
            if YTest(i) == TTest(i)
                true_negatives = true_negatives + 1;
            else
                false_positives = false_positives + 1;
            end
        else
            if YTest(i) == TTest(i)
                true_positives = true_positives + 1;
            else
                false_negatives = false_negatives + 1;
            end
        end
    end
    disp("Sensitivity = " + true_positives/(true_positives + false_negatives));
    disp("Specificity = " + true_negatives/(true_negatives + false_positives));
end