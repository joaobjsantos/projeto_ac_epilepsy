% André Filipe de Oliveira Moreira Nº 2020239416, PL3
% João Bernardo de Jesus Santos  Nº 2020218995, PL3
% Eliseu António Domingos Nº 2023174914, PL3  
% 
% This function trains a simple autoencoder with the given input data and
% middle layer size

function enc = autoencoder(X, size, patient)
    if isfile("enc_"+size+"_"+patient+".mat")
        enc = importdata("enc_"+size+"_"+patient+".mat");
    else
        enc = trainAutoencoder(X, size, UseGPU=true);
        save("enc_"+size+"_"+patient+".mat", "enc");
    end
end
    