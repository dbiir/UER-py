import lightgbm as lgb
import numpy as np
import json
import argparse
from run_lgb_cv_bayesopt import read_labels


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Path options.
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path of the trainset.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path of the testset.")    
    parser.add_argument("--train_features_path", type=str, required=True,
                        help="Path of the train features for stacking.")
    parser.add_argument("--test_features_path", type=str, required=True,
                        help="Path of the test features for stacking.")

    # Model options.
    parser.add_argument("--models_num", type=int, default=64,
                        help="Number of models for ensemble.")
    parser.add_argument("--labels_num", type=int, default=6,
                        help="Number of label.")

    args = parser.parse_args()

    train_features = []
    for i in range(args.models_num):
        train_features.append(np.load(args.train_features_path + "train_features_" + str(i) + ".npy"))
    train_features = np.concatenate(train_features, axis=-1)
    train_labels = read_labels(args.train_path)

    test_features = []
    for i in range(args.models_num):
        test_features.append(np.load(args.test_features_path + "test_features_" + str(i) + ".npy"))
    test_features = np.concatenate(test_features, axis=-1)
    test_labels = read_labels(args.test_path)

    params = {
        "task": "train",
        "objective": "multiclass",
        "num_class": args.labels_num,  
        "metric": "multi_error",
        "feature_fraction": 0.25,
        "lambda_l1": 5.0,
        "lambda_l2": 5.0,

        "learning_rate": 0.02,
        "max_depth": 100,
        "min_data_in_leaf": 50,
        "num_leaves": 10
    }

    lgb_train = lgb.Dataset(train_features, train_labels) 
    lgb_eval = lgb.Dataset(test_features, test_labels, reference=lgb_train)  

    model = lgb.train(params, lgb_train, valid_sets=lgb_eval, verbose_eval=50)

    test_pred = model.predict(test_features)
    test_pred = np.argmax(test_pred, axis=1)

    confusion = np.zeros((args.labels_num, args.labels_num))

    for i in range(len(test_pred)):
        confusion[test_pred[i], test_labels[i]] += 1
    correct = np.sum(test_pred == test_labels)

    macro_f1 = []
    print("Confusion matrix:")
    print(confusion)
    print("Report precision, recall, and f1:")
    eps = 1e-9
    for i in range(args.labels_num):
        p = confusion[i, i].item() / (confusion[i, :].sum().item() + eps)
        r = confusion[i, i].item() / (confusion[:, i].sum().item() + eps)
        f1 = 2 * p * r / (p + r + eps)
        print("Label {}: {:.3f}, {:.3f}, {:.3f}".format(i, p, r, f1))
        macro_f1.append(f1)

    print("Macro F1: {:.4f}".format(np.mean(macro_f1)))
    print("Acc. (Correct/Total): {:.4f} ({}/{})".format(correct/len(test_pred), correct, len(test_pred)))


if __name__ == "__main__":
    main()
