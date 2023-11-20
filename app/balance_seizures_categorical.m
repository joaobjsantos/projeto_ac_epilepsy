function [idxTrain, idxVal, idxTest] = balance_seizures_categorical(T, trainPercentage, valPercentage)
    seizure_start_idx = [];
    seizure_end_idx = [];
    i = 1;
    while i < length(T)
        if T(i) ~= "Ictal"
            i = i + 1;
        else
            seizure_start_idx = [seizure_start_idx, i];
            while i < length(T) 
                if T(i) ~= "Ictal"
                    seizure_end_idx = [seizure_end_idx, i-1];
                    break;
                else
                    i = i + 1;
                end
            end
        end
    end

    disp(seizure_start_idx)
    disp(seizure_end_idx)
    num_seizures = length(seizure_start_idx);
    
    valCumPercentage = trainPercentage + valPercentage;
    
    disp(seizure_end_idx(floor(num_seizures*trainPercentage)))
    disp(seizure_end_idx(floor(num_seizures*valCumPercentage)))

    idxTrain = 1:seizure_end_idx(floor(num_seizures*trainPercentage));
    if(floor(num_seizures*valCumPercentage) > seizure_end_idx(floor(num_seizures*trainPercentage)))
        idxVal = seizure_end_idx(floor(num_seizures*trainPercentage))+1:seizure_end_idx(floor(num_seizures*valCumPercentage));
    else
        idxVal = [];
    end
    idxTest = seizure_end_idx(floor(num_seizures*valCumPercentage))+1:length(T);
end