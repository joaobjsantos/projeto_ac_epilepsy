% André Filipe de Oliveira Moreira Nº 2020239416, PL3
% João Bernardo de Jesus Santos  Nº 2020218995, PL3
% Eliseu António Domingos Nº 2023174914, PL3  
% 
% This function balances the number of interictal instances on the training
% set, and returns the adequate weights for training the networks, along
% with the new training indices

function [idx, class_weights, ew] = class_balancing(T, idx, calc_weights, custom_weights, categorical)
    ew = [];

    if categorical
        interIdx = find(T(idx) == "Interictal");
        preIdx = find(T(idx) == "Preictal");
        ictalIdx = find(T(idx) == "Ictal");
    else
        interIdx = find(all(T(:, idx) == [1; 0; 0]));
        preIdx = find(all(T(:, idx) == [0; 1; 0]));
        ictalIdx = find(all(T(:, idx) == [0; 0; 1]));
    end

    n_inter = length(interIdx);
    n_pre = length(preIdx);
    n_ictal = length(ictalIdx);
    total = n_inter + n_pre + n_ictal;

    interBalancedIdx = interIdx(randperm(n_inter, n_pre + n_ictal));

    disp("Balancing: " + length(interBalancedIdx) + " " + n_pre + " " + n_ictal);

    idx = sort([interBalancedIdx, preIdx, ictalIdx]);
    
    if calc_weights
        class_weights = [total/length(interBalancedIdx), total/n_pre, total/n_ictal];
        class_weights = class_weights/mean(class_weights) + custom_weights
    else
        class_weights = custom_weights
    end

    if ~categorical
        ew = zeros(1, length(T));
        ew(interBalancedIdx) = class_weights(1);
        ew(preIdx) = class_weights(2);
        ew(ictalIdx) = class_weights(3);
end