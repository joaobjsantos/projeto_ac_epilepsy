% André Filipe de Oliveira Moreira Nº 2020239416, PL3
% João Bernardo de Jesus Santos  Nº 2020218995, PL3
% Eliseu António Domingos Nº 2023174914, PL3  
% 
% This function splits indices into train, validation and test continuous
% indices


function [idx1, idx2, idx3] = splitIdx(n, per1, per2, per3)
    idx1 = 1:floor(per1*n);
    cumPer2 = per1+per2;
    if per3 == 0
        idx2 = floor(per1*n)+1:n;
        idx3 = [];
    else
        idx2 = floor(per1*n)+1:floor(cumPer2*n);
        idx3 = floor(cumPer2*n)+1:n;
    end
    disp("#trainIdx = " + length(idx1));
    disp("#valIdx = " + length(idx2));
    disp("#testIdx = " + length(idx3));
end