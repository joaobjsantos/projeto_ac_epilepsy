% André Filipe de Oliveira Moreira Nº 2020239416, PL3
% João Bernardo de Jesus Santos  Nº 2020218995, PL3
% Eliseu António Domingos Nº 2023174914, PL3  
% 
% This function receives data from the GUI and executes the actions
% selected


function [se, sp, se_5_10, sp_5_10, error_msg] = main(type, netname, action, patient, encoder_size, balance_seizure_num, useCPUGPU)
    error_msg = "";
    se = 0; 
    sp = 0; 
    se_5_10 = 0;  
    sp_5_10 = 0; 
    
    if ~isfile("FeatVectSelT_"+patient+".mat")
        preprocess(patient);
    end
    
    X = importdata("FeatVectSelT_"+patient+".mat");
    
    if encoder_size > 0
        enc = autoencoder(X, encoder_size, patient);
        X = encode(enc, X);
    elseif netname == "FTDNN"
        enc = autoencoder(X, 5, patient);
        X = encode(enc, X);
    end

    type_id = dictionary( ...
        "detect", 1, ...
        "predict", 2 ...
        );

    net_id = dictionary( ...
        "FFN", 1, ...
        "FTDNN", 2, ...
        "CNN", 3, ...
        "LSTM", 4 ...
        );
    
    best_net_54802 = [
        ["ffn_net_54802_d_39_98", "ftdnn_net_54802_d_63_91", "cnn_net_54802_d_74_100", "lstm_net_54802_d_64_98"];
        ["", "ftdnn_net_112502_d_38_90", "", ""]
        ];
    
    if netname == "FFN"
        T = importdata("T_"+patient+".mat");

        if action == "train"
            [ffn_net, testIdx] = ffn(X, T, type, balance_seizure_num, useCPUGPU);
        elseif action == "test"
            testIdx = 1:length(T);
            if patient == "54802"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    ffn_net = importdata(best_net_name + ".mat");
                end
            elseif patient == "112502"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    ffn_net = importdata(best_net_name + ".mat");
                end
            else
                error_msg = "There are no available pretrained models of this patient";
            end
        else
            disp("Invalid action");
        end
        testIdx = 1:length(T); 
        XTest = X(:, testIdx);
        TTest = T(:, testIdx);

        if useCPUGPU
            Y = ffn_net(XTest, 'useParallel', 'yes', 'useGpu', 'yes');
        else
            Y = ffn_net(XTest);
        end

        save ffn_net.mat ffn_net

        [se, sp, se_5_10, sp_5_10] = test_se_sp(TTest, Y, type, false);
    elseif netname == "FTDNN"
        T = importdata("T_"+patient+".mat");
        

        if action == "train"
            [ftdnn_net, testIdx] = ftdnn(X, T, type, balance_seizure_num, useCPUGPU);
        elseif action == "test"
            testIdx = 1:length(T);
            if patient == "54802"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    ftdnn_net = importdata(best_net_name + ".mat");
                end
            elseif patient == "112502"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    ftdnn_net = importdata(best_net_name + ".mat");
                end
            else
                error_msg = "There are no available pretrained models of this patient";
            end
        else
            disp("Invalid action");
        end
        testIdx = 1:length(T); 
        XTest = X(:, testIdx);
        TTest = T(:, testIdx);
        
        if useCPUGPU
            Y = cell2mat(ftdnn_net(num2cell(XTest,1), 'useParallel', 'yes'));
        else
            Y = cell2mat(ftdnn_net(num2cell(XTest,1)));
        end
        
        save ftdnn_net.mat ftdnn_net

        [se, sp, se_5_10, sp_5_10] = test_se_sp(TTest, Y, type, false);
    elseif netname == "CNN"
        T = importdata("T_categorical_"+patient+".mat");

        if action == "train"
            [cnn_net, testIdx] = cnn(X, T, type, balance_seizure_num, useCPUGPU);
        elseif action == "test"
            testIdx = 1:length(T);
            if patient == "54802"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    cnn_net = importdata(best_net_name + ".mat");
                end
            elseif patient == "112502"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    cnn_net = importdata(best_net_name + ".mat");
                end
            else
                error_msg = "There are no available pretrained models of this patient";
            end
        else
            disp("Invalid action");
        end
        testIdx = 1:length(T); 
        XTest = num2cell(X,1);
        
        XTest = XTest(testIdx);
        TTest = T(testIdx);

        if useCPUGPU
            [XTestCNN, TTestCNN] = prepare_cnn(XTest, TTest, size(X,1));
            YTestCNN = classify(cnn_net,XTestCNN, ExecutionEnvironment="gpu");
        else
            [XTestCNN, TTestCNN] = prepare_cnn(XTest, TTest, size(X,1));
            YTestCNN = classify(cnn_net,XTestCNN);
        end
        
        save cnn_net.mat cnn_net

        [se, sp, se_5_10, sp_5_10] = test_se_sp(TTestCNN, YTestCNN, type, true);
    elseif netname == "LSTM"
        T = importdata("T_categorical_"+patient+".mat");

        

        if action == "train"
            [lstm_net, testIdx] = lstm(X, T, type, balance_seizure_num, useCPUGPU);
        elseif action == "test"
            testIdx = 1:length(T);
            if patient == "54802"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    lstm_net = importdata(best_net_name + ".mat");
                end
            elseif patient == "112502"
                best_net_name = best_net_54802(type_id(type), net_id(netname))
                if best_net_name == ""
                    error_msg = "Combination of patient, detect/predict and network type not available as a pretrained model"
                    return
                else
                    lstm_net = importdata(best_net_name + ".mat");
                end
            else
                error_msg = "There are no available pretrained models of this patient";
            end
        else
            disp("Invalid action");
        end
        testIdx = 1:length(T); 
        XTest = num2cell(X,1);
        
        XTest = XTest(testIdx);
        TTest = T(testIdx);

        if useCPUGPU
            YTest = classify(lstm_net,XTest, ExecutionEnvironment="gpu");
        else
            YTest = classify(lstm_net,XTest);
        end
        
        save lstm_net.mat lstm_net

        [se, sp, se_5_10, sp_5_10] = test_se_sp(TTest, YTest, type, true);
    else
        disp("Wrong net name");
    end
end