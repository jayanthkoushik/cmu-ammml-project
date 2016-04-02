% manfeatsvm.m: svm on manually extracted features.
addpath('~/Documents/libsvm-3.21/matlab');

DATA_ROOT = fullfile(pwd, '..', 'data', 'man_feats', 'mattext');
SAVE_ROOT = fullfile(pwd, '..', 'data', 'saves');
MODALITIES = {'visual', 'audio', 'text'};
SIGS = {'all', 'sig'};
CVALUES = linspace(10^-7, 10^7, 100);
GVALUES = linspace(10^-7, 10^-1, 100);

for m = 1:length(MODALITIES)
    modality = MODALITIES{m};
    for s = 1:length(SIGS)
        sig = SIGS{s};

        X_train = dlmread(fullfile(DATA_ROOT, modality, strcat('X_', sig, '_train.txt')));
        X_val = dlmread(fullfile(DATA_ROOT, modality, strcat('X_', sig, '_val.txt')));
        X_test = dlmread(fullfile(DATA_ROOT, modality, strcat('X_', sig, '_test.txt')));

        y_train = dlmread(fullfile(DATA_ROOT, modality, strcat('y_train.txt')));
        y_val = dlmread(fullfile(DATA_ROOT, modality, strcat('y_val.txt')));
        y_test = dlmread(fullfile(DATA_ROOT, modality, strcat('y_test.txt')));

        best_acc = 0;
        for i = 1:length(CVALUES)
            for j = 1:length(GVALUES)
                model = svmtrain(y_train, X_train, strcat(['-s 0 -t 2 -m 1000 -c ' num2str(CVALUES(i)) ' -g ' num2str(GVALUES(j))]));
                [pred, acc, dec] = svmpredict(y_val, X_val, model);
                if acc(1) > best_acc
                    best_acc = acc(1);
                    best_c = CVALUES(i);
                    best_g = GVALUES(j);
                end
            end
        end

        model = svmtrain(cat(1, y_train, y_val), cat(1, X_train, X_val), strcat(['-s 0 -t 2 -m 1000 -c ' num2str(best_c) ' -g ' num2str(best_g)]));
        [pred, acc, dec] = svmpredict(y_test, X_test, model);
        summary = struct('accuracy', acc(1), 'best_c', best_c, 'best_g', best_g);

        save_dir = fullfile(SAVE_ROOT, strcat('mansvm_', modality, '_', sig));
        mkdir(save_dir);
        save(fullfile(save_dir, 'model.mat'), 'model');
        save(fullfile(save_dir, 'summary.mat'), 'summary');
    end
end