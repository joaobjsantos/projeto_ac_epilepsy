function test_ffn()
    load("FeatVectSelT.mat");
    load("T.mat");
    load("ffn_net.mat");
    load("tr.mat");
    total_correct = 0;
    total_correct_ictal = 0;
    total_ictal = 0;
    total = 0;
    guesses = ffn_net(FeatVectSelT, 'useGPU', 'yes');
    for ind = tr.testInd
        correct = find(T(:,ind)==1);
        guess = guesses(:,ind);
        guess_max = find(guess == max(guess));
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
    disp(total_correct/length(tr.testInd));
    disp(total_correct_ictal/total_ictal);
end