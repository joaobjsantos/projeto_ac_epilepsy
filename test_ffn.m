function test_ffn()
    load("FeatVectSelT.mat");
    load("T.mat");
    load("ffn_net.mat");
    load("tr.mat");
    total_correct = 0;
    total_correct_ictal = 0;
    total_ictal = 0;
    total = 0;
    guesses = ffn_net(FeatVectSelT(:, tr.testInd), 'useGPU', 'yes');
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
    disp(total_correct/length(tr.testInd));
    disp(total_correct_ictal/total_ictal);
    disp(sum(guesses_max == 2));
    disp(sum(guesses_max == 3));
    disp(total_ictal);

    save guesses_max.mat guesses_max
end