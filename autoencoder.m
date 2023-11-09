function autoencoder()
    X = importdata("FeatVectSelT.mat");
    autoenc = trainAutoencoder(X, 15);
    save autoenc.mat autoenc
end