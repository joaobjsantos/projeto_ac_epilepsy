function [se, sp, se_5_10, sp_5_10] = test_se_sp(T, Y, type, categorical)
    true_positives = 0;
    true_negatives = 0;
    false_positives = 0;
    false_negatives = 0;
    true_positives_5_10 = 0;
    true_negatives_5_10 = 0;
    false_positives_5_10 = 0;
    false_negatives_5_10 = 0;
    
    if type == "detect"
        if categorical
            positive = "Ictal";
        else
            positive = 3;
        end
    else
        if categorical
            positive = "Preictal";
        else
            positive = 2;
        end
    end

    if categorical
        test_size = length(Y);
    else
        test_size = size(Y, 2);
    end

    disp("test size = " + test_size); 
    
    last_10_correct = [];
    last_10_guesses = [];

    for i = 1:test_size
        if categorical
            guess = Y(i);
            correct = T(i);
        else
            guess = Y(:, i);
            correct = T(:, i);
            guess = find(guess == max(guess));
            correct = find(correct == max(correct));
        end
        

        % SE SP

        if correct == positive
            if guess == positive
                true_positives = true_positives + 1;
            else
                false_negatives = false_negatives + 1;
            end
        else
            if guess == positive
                false_positives = false_positives + 1;
            else
                true_negatives = true_negatives + 1;
            end
        end


        % SE_5_10 SP_5_10

        last_10_correct = [last_10_correct, correct];
        last_10_guesses = [last_10_guesses, guess];
        
        if rem(i, 10) == 0
            correct_5_10 = sum(last_10_correct==positive)>=5;
            guess_5_10 = sum(last_10_guesses==positive)>=5;
            last_10_correct = [];
            last_10_guesses = [];
            if correct_5_10
                if guess_5_10
                    true_positives_5_10 = true_positives_5_10 + 1;
                else
                    false_negatives_5_10 = false_negatives_5_10 + 1;
                end
            else
                if guess_5_10
                    false_positives_5_10 = false_positives_5_10 + 1;
                else
                    true_negatives_5_10 = true_negatives_5_10 + 1;
                end
            end
        end
    end

    if true_positives + false_negatives > 0
        se = round(true_positives/(true_positives + false_negatives)*100,2);
    else
        se = 0;
    end
    if true_negatives + false_positives > 0
        sp = round(true_negatives/(true_negatives + false_positives)*100,2);
    else
        sp = 0;
    end

    if true_positives_5_10 + false_negatives_5_10 > 0
        se_5_10 = round(true_positives_5_10/(true_positives_5_10 + false_negatives_5_10)*100,2);
    else
        se_5_10 = 0;
    end
    if true_negatives_5_10 + false_positives_5_10 > 0
        sp_5_10 = round(true_negatives_5_10/(true_negatives_5_10 + false_positives_5_10)*100,2);
    else
        sp_5_10 = 0;
    end

    disp("Total positives = " + (true_positives + false_negatives));
    disp("Sensitivity = " + se);
    disp("Specificity = " + sp);

    disp("Total positives 5/10 = " + (true_positives_5_10 + false_negatives_5_10));
    disp("Sensitivity 5/10 = " + se_5_10);
    disp("Specificity 5/10 = " + sp_5_10);
end