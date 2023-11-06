function test_net(netname)
    X = importdata("FeatVectSelT.mat");
    T = importdata("T.mat");
    net = importdata(netname + ".mat");
    tr = importdata("tr.mat");
    total_correct = 0;
    total_correct_ictal = 0;
    total_ictal = 0;
    total = 0;
    guesses = net(X(:, tr.testInd), 'useGPU', 'yes');
    guesses_max = zeros(1, length(guesses));
    for i = 1:length(tr.testInd)
        correct = find(T(:,tr.testInd(i))==1);
        guess = guesses(:,i);
        guess_max = find(guess == max(guess));
        guesses_max(i) = guess_max;
        if correct ~= 1
            total_ictal = total_ictal + 1;
        end

        if guess_max == correct
            total_correct = total_correct + 1;
            if correct ~= 1
                total_correct_ictal = total_correct_ictal + 1;
            end
        end
        total = total + 1;
        disp(total/length(tr.testInd));
    end
    disp("Total Accuracy/Ictal Accuracy: " + round(total_correct/length(tr.testInd),4) ...
        + "/" + round(total_correct_ictal/total_ictal,4));
    disp("Preictal+Ictal/Total Ictal: " + sum(guesses_max == 2) + "+" ...
        + sum(guesses_max == 3) + "/" + total_ictal);
    save guesses_max.mat guesses_max
end