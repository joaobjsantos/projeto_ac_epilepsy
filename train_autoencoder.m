function train_autoencoder()
    X = importdata("FeatVectSelT.mat");
    autoenc = trainAutoencoder(X, 5, UseGPU=true);
    save autoenc.mat autoenc
end