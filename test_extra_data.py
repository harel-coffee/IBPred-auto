from auto_pred import *
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    matthews_corrcoef,
    recall_score,
    balanced_accuracy_score,
)

models_ = base_grid_models + base_bayes_models + clf_models
set_num_ = base_grid_d + base_bayes_d + clf_d
file_ = os.path.join(
    data_path, 'extra_data', 'extra_data_ibp14101_non-ibp12924.fasta'
)

feature_ = get_features(file_)
metrics = ['model', 'dimension', 'AUC', 'OA', 'MCC', 'Sn', 'Sp', 'AA']
test_result_ = pd.DataFrame([], columns=metrics)
y_true = np.array([1] * 14101 + [0] * 12924)
for i, m in enumerate(models_):
    print('test model: %s; set_num:%d' % (m, set_num_[i]))
    result = auto_pred(feature_, m, set_num_[i])
    metrics_ = [
        m,
        set_num_[i],
        roc_auc_score(y_true, result.iloc[:, 2].values),
        accuracy_score(y_true, result.iloc[:, 0].values),
        matthews_corrcoef(y_true, result.iloc[:, 0].values),
        recall_score(y_true, result.iloc[:, 0].values, pos_label=1),  # sn
        recall_score(y_true, result.iloc[:, 0].values, pos_label=0),  # sp
        balanced_accuracy_score(y_true, result.iloc[:, 0].values),
    ]
    test_result_ = test_result_.append(
        dict(zip(metrics, metrics_,)), ignore_index=True
    )
    result.to_csv(os.path.join(data_path, 'extra_data', m + '_results.csv'))

test_result_.to_csv(
    os.path.join(data_path, 'extra_data', 'extra_data_metrics.csv'),
    index=False,
)
