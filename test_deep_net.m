function test_deep_net(netname, XTest, TTest)
    net = importdata(netname + ".mat");
    total_correct = 0;
    total_correct_ictal = 0;
    total_ictal = 0;
    total = 0;
    minibatchsize = 2048;
    YTest = classify(net,XTest,MiniBatchSize=minibatchsize, ExecutionEnvironment="gpu");
    for i = 1:length(TTest)
        if TTest(i) ~= "Interictal"
            total_ictal = total_ictal + 1;
        end
        if YTest(i) == TTest(i)
            total_correct = total_correct + 1;
            if TTest(i) ~= "Interictal"
                total_correct_ictal = total_correct_ictal + 1;
            end
        end
        total = total + 1;
        disp(total/length(TTest));
    end
    disp("Total Accuracy/Ictal Accuracy: " + round(total_correct/length(TTest),4) ...
        + "/" + round(total_correct_ictal/total_ictal,4));
    disp("Preictal+Ictal/Total Ictal: " + sum(YTest == "Preictal") + "+" ...
        + sum(YTest == "Ictal") + "/" + total_ictal);
end