function test_predict_net_se_sp(netname)
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    % enc = importdata("autoenc.mat");
    % X = num2cell(encode(enc, X),1);
    X = num2cell(X,1);
    T = num2cell(T,1);
    net = importdata(netname + ".mat");
    testInd = importdata("testInd.mat");
    true_positives = 0;
    true_negatives = 0;
    false_positives = 0;
    false_negatives = 0;
    guesses = net(X(testInd));
    guesses_max = zeros(1, length(guesses));
    for i = 1:length(testInd)
        correct = find(T{testInd(i)}==1);
        guess = guesses{i};
        guess_max = find(guess == max(guess));
        guesses_max(i) = guess_max;
        if correct == 1
            if guess_max == correct
                true_negatives = true_negatives + 1;
            else
                false_positives = false_positives + 1;
            end
        else
            if guess_max == correct
                true_positives = true_positives + 1;
            else
                false_negatives = false_negatives + 1;
            end
        end
    end
    disp("Sensitivity = " + true_positives/(true_positives + false_negatives));
    disp("Specificity = " + true_negatives/(true_negatives + false_positives));
    save guesses_max.mat guesses_max
end