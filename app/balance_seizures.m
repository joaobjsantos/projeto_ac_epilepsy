% André Filipe de Oliveira Moreira Nº 2020239416, PL3
% João Bernardo de Jesus Santos  Nº 2020218995, PL3
% Eliseu António Domingos Nº 2023174914, PL3  
% 
% This function divides target T by indices, so that those indices have a
% percentage of the current patient's total seizures equal to per1, per2
% and per3

function [idx1, idx2, idx3] = balance_seizures(T, per1, per2, per3, categorical)
    seizure_start_idx = [];
    seizure_end_idx = [];
    i = 1;
    
    if categorical
        interictal = "Interictal";
    else
        interictal = [1; 0; 0];
    end


    while i < length(T)
        if categorical
            current_class = T(i);
        else
            current_class = T(:,i);            
        end

        if current_class == interictal
            i = i + 1;
        else
            seizure_start_idx = [seizure_start_idx, i];
            while i < length(T) 
                if categorical
                    current_class = T(i);
                else
                    current_class = T(:,i);
                end

                if current_class == interictal
                    seizure_end_idx = [seizure_end_idx, i-1];
                    break;
                else
                    i = i + 1;
                end
            end
        end
    end

    % disp(seizure_start_idx)
    % disp(seizure_end_idx)
    num_seizures = length(seizure_start_idx);
    
    valCumPercentage = per1 + per2;

    % disp(seizure_end_idx(floor(num_seizures*per1)))
    
    disp("Total seizures: " + num_seizures);
    disp("Total indices: " + length(T));

    idx1_start = 1;
    idx1_end = seizure_end_idx(floor(num_seizures*per1));
    idx1 = idx1_start:idx1_end;
    disp("per1: " + floor(num_seizures*per1) + " seizures, " + (idx1_end - idx1_start) + " indices.");

    if per3 > 0
        idx2_start = seizure_end_idx(floor(num_seizures*per1))+1;
        idx2_end = seizure_end_idx(floor(num_seizures*valCumPercentage));
        idx2 = idx2_start:idx2_end;
        disp("per2: " + (floor(num_seizures*valCumPercentage)-floor(num_seizures*per1)) + " seizures, " + (idx2_end - idx2_start) + " indices.");
        
        idx3_start = seizure_end_idx(floor(num_seizures*valCumPercentage))+1;
        idx3_end = length(T);
        idx3 = idx3_start:idx3_end;
        disp("per3: " + (num_seizures-floor(num_seizures*valCumPercentage)) + " seizures, " + (idx3_end - idx3_start) + " indices.");
    else
        idx2_start = seizure_end_idx(floor(num_seizures*per1))+1;
        idx2_end = length(T);
        idx2 = idx2_start:idx2_end;
        disp("per2: " + (num_seizures-floor(num_seizures*per1)) + " seizures, " + (idx2_end - idx2_start) + " indices.");

        idx3 = [];
    end  
end