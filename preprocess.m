function [T, newTrg] = preprocess(patient)
    load(patient);

    % some constants
    PREICTAL_SIZE = 300;
    POSTICTAL_SIZE = 60;

    INTERICTAL_LABEL = 1;
    PREICTAL_LABEL = 2;
    ICTAL_LABEL = 3;
    POSTICTAL_LABEL = ICTAL_LABEL;

    INTERICTAL_CLASS = [1 0 0]';
    PREICTAL_CLASS = [0 1 0]';
    ICTAL_CLASS = [0 0 1]';
    POSTICTAL_CLASS = ICTAL_CLASS;


    % transpose feature vector
    FeatVectSelT = FeatVectSel.';


    % create target vector
    map_newTrg_values = dictionary([0 1], [1 3]);
    newTrg = map_newTrg_values(Trg);

    
    % label preictal and postictal states
    i = 1;
    while i < length(newTrg)
        if newTrg(i) == ICTAL_LABEL
            % mark preictal states
            for j=i-1:-1:max(i-PREICTAL_SIZE, 1)
                if newTrg(j) ~= INTERICTAL_LABEL
                    break;
                end
                newTrg(j) = PREICTAL_LABEL;
            end
            % skip ictal states
            i = i + 1;
            while newTrg(i) == ICTAL_LABEL && i < length(newTrg) 
                i = i + 1;
            end
            
    
            % mark postictal states
            postictal_limit = i + POSTICTAL_SIZE - 1;
            
            for i=i:min(postictal_limit, length(newTrg))
                if newTrg(i) ~= INTERICTAL_LABEL
                    break;
                end
                newTrg(i) = POSTICTAL_LABEL;
            end
        end
        i = i + 1;
    end


    % create T
    map_T_values = [INTERICTAL_CLASS, PREICTAL_CLASS, ICTAL_CLASS];
    %disp(min(newTrg) + " " + max(newTrg));
    T = zeros(3, length(newTrg));
    for i=1:length(newTrg)
        T(:,i) = map_T_values(:, newTrg(i));
    end

    % save new features and target matrix
    save FeatVectSelT.mat FeatVectSelT
    save newTrg.mat newTrg
    save T.mat T
end