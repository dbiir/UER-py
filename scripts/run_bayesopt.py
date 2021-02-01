import warnings
import lightgbm as lgb
import numpy as np
from bayes_opt import BayesianOptimization
import argparse


def read_labels(dataset_path):
    with open(dataset_path, mode="r", encoding="utf-8") as f:
        columns, labels = {}, []
        for line_id, line in enumerate(f):
            if line_id == 0:
                for i, column_name in enumerate(line.strip().split("\t")):
                    columns[column_name] = i
                continue
            line = line[:-1].split("\t")
            labels.append(int(line[columns["label"]]))
        return labels


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--train_features_path", type=str, required=True,
                        help="Path of the train features for stacking.")

    # Model options.
    parser.add_argument("--models_num", type=int, default=64,
                        help="Number of models for ensemble.")
    parser.add_argument("--folds_num", type=int, default=5,
                        help="Number of folds for cross validation.")    
    parser.add_argument("--labels_num", type=int, default=2,
                        help="Number of labels.")

    # Bayesian optimization options.
    parser.add_argument("--epochs_num", type=int, default=300,
                        help="Number of epochs.")

    args = parser.parse_args()

    labels = []

    labels = read_labels(args.train_path)

    train_features = []
    for i in range(args.models_num):
        train_features.append(np.load(args.train_features_path + 'train_features_'+str(i)+'.npy'))

    train_features = np.concatenate(train_features, axis=-1)
    print(train_features.shape)

    def lgb_cv(
            num_leaves, 
            min_data_in_leaf,  
            learning_rate,
            min_sum_hessian_in_leaf, 
            feature_fraction,
            lambda_l1,
            lambda_l2,
            min_gain_to_split,
            max_depth):

        num_leaves = int(num_leaves)
        min_data_in_leaf = int(min_data_in_leaf)
        max_depth = int(max_depth)
     
        param = {
            'num_leaves': num_leaves,
            'min_data_in_leaf': min_data_in_leaf,
            'learning_rate': learning_rate,
            'min_sum_hessian_in_leaf': min_sum_hessian_in_leaf,
            'feature_fraction': feature_fraction,
            'lambda_l1': lambda_l1,
            'lambda_l2': lambda_l2,
            'min_gain_to_split': min_gain_to_split,
            'max_depth': max_depth,
            'save_binary': True,
            'max_bin': 63,
            'bagging_fraction': 0.4,
            'bagging_freq': 5,
            'seed': 2017,
            'objective': 'multiclass',
            'num_class': args.labels_num,  
            'boosting_type': 'dart',
            'verbose': -1,
            'metric': 'multi_error',
        }
        scores = []

        instances_num_per_fold = len(labels) // args.folds_num + 1

        for fold_id in range(args.folds_num):

            x_train = np.concatenate((train_features[0:fold_id*instances_num_per_fold], train_features[(fold_id+1)*instances_num_per_fold:]),axis = 0)
            x_val = train_features[fold_id*instances_num_per_fold: (fold_id+1)*instances_num_per_fold]
            y_train = labels[0:fold_id*instances_num_per_fold] + labels[(fold_id+1)*instances_num_per_fold:]
            y_val = labels[fold_id*instances_num_per_fold: (fold_id+1)*instances_num_per_fold]

            lgb_train = lgb.Dataset(x_train, y_train) 
            lgb_eval = lgb.Dataset(x_val, y_val, reference=lgb_train)  

            model = lgb.train(param,lgb_train,num_boost_round=150,valid_sets=lgb_eval, early_stopping_rounds=30, verbose_eval=0)

            pred = model.predict(x_val)
            val_pred = np.argmax(pred,axis=1)

            confusion = np.zeros((args.labels_num,args.labels_num))

            for i in range(len(pred)):
                confusion[val_pred[i],y_val[i]] += 1
            correct = np.sum(val_pred == y_val)

            marco_f1 = []
            for i in range(args.labels_num):
                try:
                    p = confusion[i,i].item()/confusion[i,:].sum().item()
                    r = confusion[i,i].item()/confusion[:,i].sum().item()
                    f1 = 2*p*r / (p+r)
                except ZeroDivisionError:
                    f1 = 0
                marco_f1.append(f1)
            scores.append(np.mean(marco_f1))
        
        return np.mean(scores)
        
    bounds = {
        'num_leaves': (20, 90),
        'min_data_in_leaf': (5, 100),
        'learning_rate': (0.005, 0.5),
        'min_sum_hessian_in_leaf': (0.00001, 20),
        'feature_fraction': (0.001, 0.5),
        'lambda_l1': (0, 10),
        'lambda_l2': (0, 10),
        'min_gain_to_split': (0, 1.0),
        'max_depth':(3,200),
    }

    lgb_bo = BayesianOptimization(lgb_cv, bounds)
     
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore')
        lgb_bo.maximize(n_iter=args.epochs_num)


if __name__ == "__main__":
    main()

