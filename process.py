import os
import time
import joblib
import random
import warnings
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import f_classif
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import matthews_corrcoef, accuracy_score
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from ClfSearchCV import RF_GridCV, SVM_BayesCV, DT_BayesCV
from ClfSearchCV import RF_BayesCV, NB_BayesCV, ABC_BayesCV
from PseCKSAAP import PseCKSAAP
from DDE import DDE
from CTD import CTD
from QSOrder import QSOrder


# Initial settings
random.seed(42)
np.random.seed(42)
random_state = 42
np.seterr(divide='ignore', invalid='ignore')
warnings.filterwarnings('ignore')
data_path = os.path.join(os.path.abspath('.'), 'datas')
feature_path = os.path.join(data_path, 'feature')
ifs_path = os.path.join(data_path, 'ifs')
bin_path = os.path.join(data_path, 'bin')
pos_num = 114
neg_num = 207
label = np.array([1] * pos_num + [0] * neg_num)
raw_fasta = os.path.join(data_path, 'data.fasta')


# Feature extraction
# parameters of feature extraction methods
k, delta, nlag, w = 9, 10, 5, 0.5

pseck = PseCKSAAP(gap=k, delta=delta)
pseck.fit(raw_fasta)
pseck.encodings.to_csv(os.path.join(feature_path, 'feature_psecksaap.csv'))
print('feature psecksaap √')

dde = DDE()
dde.fit(raw_fasta)
dde.encodings.to_csv(os.path.join(feature_path, 'feature_dde.csv'))
print('feature dde √')

ctd = CTD()
ctd.fit(raw_fasta)
ctd.encodings.to_csv(os.path.join(feature_path, 'feature_ctd.csv'))
print('feature ctd √')

qso = QSOrder(nlag=nlag, w=w)
qso.fit(raw_fasta)
qso.encodings.to_csv(os.path.join(feature_path, 'feature_qsorder.csv'))
print('feature qsorder √')

feature_all = pd.concat(
    (
        pseck.encodings,
        dde.encodings.iloc[:, :],
        ctd.encodings.iloc[:, :],
        qso.encodings.iloc[:, :],
    ),
    axis=1,
    sort=False,
)
feature_all.to_csv(os.path.join(feature_path, 'feature_all.csv'))

# in this process:
# psecksaap + dde + ctd + qsorder
# 4140 + 400 + 273 + 50
names = ['P', 'PD', 'PDC', 'PDCQ']
lens = [
    len(pseck.headers),
    len(pseck.headers + dde.headers),
    len(pseck.headers + dde.headers + ctd.headers),
    len(pseck.headers + dde.headers + ctd.headers + qso.headers),
]
name_lens = dict(zip(names, lens))
del (raw_fasta, pseck, dde, ctd, qso)


# Training and testing
# split dataset
X_train_pre, X_test_pre, y_train, y_test = train_test_split(
    feature_all,
    label,
    test_size=0.2,
    random_state=random_state,
    shuffle=True,
    stratify=label,
)

X_train_pre = pd.DataFrame(X_train_pre, columns=feature_all.columns)
pd.DataFrame(
    np.c_[y_train, X_train_pre],
    columns=['label'] + feature_all.columns.to_list(),
).to_csv(os.path.join(feature_path, 'feature_train.csv'), index=False)

X_test_pre = pd.DataFrame(X_test_pre, columns=feature_all.columns)
pd.DataFrame(
    np.c_[y_test, X_test_pre],
    columns=['label'] + feature_all.columns.to_list(),
).to_csv(os.path.join(feature_path, 'feature_test.csv'), index=False)

del feature_all


# Feature selection
# 1. choose significant features
# ANOVA
Fscore, pvalue = f_classif(X_train_pre, y_train)
Fscore = pd.Series(Fscore, index=X_train_pre.columns).sort_values(
    ascending=False
)
Fscore.to_csv(os.path.join(data_path, "F_score.csv"), header=None)

Pvalue = pd.Series(pvalue, index=X_train_pre.columns).sort_values(
    ascending=True
)
Pvalue.to_csv(os.path.join(data_path, "p_value.csv"), header=None)

# feature significant // p < 0.05
feature_sig_columns = {
    name: X_train_pre.iloc[:, :lens]
    .reindex(columns=Pvalue[Pvalue < 0.05].dropna().index)
    .dropna(axis=1)
    .columns
    for name, lens in name_lens.items()
}
pd.DataFrame(
    dict([(k, pd.Series(v)) for k, v in feature_sig_columns.items()])
).to_csv(
    os.path.join(data_path, 'f_score_sig_features_by_methods.csv'), index=False
)

X_train_pre = X_train_pre[feature_sig_columns['PDCQ']]
X_train_pre.to_csv(os.path.join(feature_path, 'feature_train_sig.csv'))
X_test_pre = X_test_pre[feature_sig_columns['PDCQ']]
X_test_pre.to_csv(os.path.join(feature_path, 'feature_test_sig.csv'))


# 2. scaling
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train_pre)

# save scaler
# input features of this scaler must be significant fetures
joblib.dump(scaler, os.path.join(bin_path, 'scaler_sig'))

X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train_pre.columns)
X_test_scaled = pd.DataFrame(
    scaler.transform(X_test_pre), columns=X_train_pre.columns
)

pd.DataFrame(
    np.c_[y_train, X_train_scaled],
    columns=['label'] + X_train_pre.columns.to_list(),
).to_csv(os.path.join(feature_path, 'train_sig.csv'), index=False)
pd.DataFrame(
    np.c_[y_test, X_test_scaled],
    columns=['label'] + X_train_pre.columns.to_list(),
).to_csv(os.path.join(feature_path, 'test_sig.csv'), index=False)

del X_train_pre, X_test_pre


# 3. settings of training
cv = StratifiedKFold(n_splits=10)

# grid search cv --sklearn
base_grid_params = {
    'criterion': ['gini', 'entropy'],  # 2
    'max_depth': np.arange(5, 150, 35),  # 5
    'min_samples_split': np.arange(2, 30, 5),  # 6
    # 'min_samples_leaf': np.arange(1, 10, 2),  # 5
    'min_samples_leaf': [5],
    # 'max_leaf_nodes': np.arange(50, 200, 30),  # 5
    'max_leaf_nodes': [100],
    # 'ccp_alpha': np.logspace(-10, 0, num=6, base=10),  # 6
    'ccp_alpha': [1e-3],
    'n_estimators': np.logspace(1, 3, num=6, base=10, dtype=int),  # 6
}  # 360

# bayes search cv --skopt
base_bayes_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': (5, 150),
    'min_samples_split': (2, 30),
    'min_samples_leaf': (1, 10),
    'max_leaf_nodes': (50, 200),
    'ccp_alpha': (1e-10, 1.0, 'log-uniform'),
    'n_estimators': (10, 1000, 'log-uniform'),
}  # 64  ## dt + 1*(12-1) = 66 --> 64/68 --> 64


base_grid_cv = RF_GridCV(base_grid_params)
base_bayes_cv = RF_BayesCV(base_bayes_params)


# 4. main IFS process
# test metrics: using for test results DataFrame columns
metrics = ['method', 'dimension', 'OA', 'AUC', 'MCC', 'Sn', 'Sp', 'AA']
# train columns
column_ = ['dimension', 'auc_mean', 'auc_std', 'estimator']

test_result_grid = pd.DataFrame([], columns=metrics)
test_roc_grid = pd.DataFrame(
    [], index=['fpr', 'tpr', 'thresholds'], columns=name_lens.keys()
)
test_result_bayes = pd.DataFrame([], columns=metrics)
test_roc_bayes = pd.DataFrame(
    [], index=['fpr', 'tpr', 'thresholds'], columns=name_lens.keys()
)

# P, PD, PDC, and PDCQ
for name, lens in name_lens.items():
    print('------------feature approach %s begin------------' % name)

    X_train = X_train_scaled[feature_sig_columns[name]]
    X_test = X_test_scaled[feature_sig_columns[name]]

    train_result_grid = pd.DataFrame([], columns=column_)
    # IFS, use no more than `num of training samples` features
    for i in range(1, X_train.shape[0] + 1):
        print('------------%d feature(s)------------' % i)

        try:
            base_clf_grid = base_grid_cv.fit(X_train.iloc[:, :i], y_train)
            cv_metric = [
                base_clf_grid.best_score_,
                base_clf_grid.cv_results_['std_test_score'][
                    base_clf_grid.best_index_
                ],
            ]

            train_result_grid = train_result_grid.append(
                dict(
                    zip(
                        column_,
                        [i] + cv_metric + [base_clf_grid.best_estimator_],
                    )
                ),
                ignore_index=True,
            )

        except Exception as e:
            print('Retry Failed! ', repr(e))
            train_result_grid = train_result_grid.append(
                dict(zip(column_, [i] + [np.NaN] * (len(column_) - 1))),
                ignore_index=True,
            )
        time.sleep(1)

    train_result_grid.to_csv(
        os.path.join(ifs_path, 'base_grid_ifs_' + name + '.csv'), index=False,
    )

    train_result_bayes = pd.DataFrame([], columns=column_)
    for i in range(1, X_train.shape[0] + 1):
        print('------------%d feature(s)------------' % i)

        try:
            base_clf_bayes = base_bayes_cv.fit(X_train.iloc[:, :i], y_train)
            cv_metric = [
                base_clf_bayes.best_score_,
                base_clf_bayes.cv_results_['std_test_score'][
                    base_clf_bayes.best_index_
                ],
            ]

            train_result_bayes = train_result_bayes.append(
                dict(
                    zip(
                        column_,
                        [i] + cv_metric + [base_clf_bayes.best_estimator_],
                    )
                ),
                ignore_index=True,
            )

        except Exception as e:
            print('Retry Failed! ', repr(e))
            train_result_bayes = train_result_bayes.append(
                dict(zip(column_, [i] + [np.NaN] * (len(column_) - 1))),
                ignore_index=True,
            )
        time.sleep(1)

    # all ifs of one method finished
    train_result_bayes.to_csv(
        os.path.join(ifs_path, 'base_bayes_ifs_' + name + '.csv'), index=False,
    )

    # test grid search
    d_grid, clf_grid = (
        train_result_grid.sort_values(
            by=['auc_mean', 'dimension'], ascending=[True, False]
        )
        .dropna()
        .iloc[-1, [0, -1]]
    )
    d_grid = int(d_grid)
    clf_grid.fit(X_train.iloc[:, :d_grid], y_train)

    joblib.dump(clf_grid, os.path.join(bin_path, 'base_grid_clf_' + name))

    # determine the thresholds
    y_pred_pro = (
        clf_grid.predict_proba(X_test.iloc[:, :d_grid])[:, 1]
        if hasattr(clf_grid, 'predict_proba')
        else clf_grid.decision_function(X_test.iloc[:, :d_grid])
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_pro)
    test_roc_grid.loc['fpr', name] = fpr.tolist()
    test_roc_grid.loc['tpr', name] = tpr.tolist()
    test_roc_grid.loc['thresholds', name] = thresholds.tolist()
    # tolist: for reading file, using eval()

    Youden_index = np.argmax(tpr - fpr)
    y_pred = y_pred_pro >= thresholds[Youden_index]

    # grid results
    test_result_grid = test_result_grid.append(
        dict(
            zip(
                metrics,
                [
                    name,
                    d_grid,
                    accuracy_score(y_test, y_pred),
                    roc_auc_score(y_test, y_pred_pro),
                    matthews_corrcoef(y_test, y_pred),
                    tpr[Youden_index],  # sn
                    1 - fpr[Youden_index],  # sp
                    balanced_accuracy_score(y_test, y_pred),
                ],
            )
        ),
        ignore_index=True,
    )

    # redundantly save test data
    test_result_grid.to_csv(
        os.path.join(data_path, 'base_grid_test_results.csv'), index=False
    )

    # test grid search
    d_bayes, clf_bayes = (
        train_result_bayes.sort_values(
            by=['auc_mean', 'dimension'], ascending=[True, False]
        )
        .dropna()
        .iloc[-1, [0, -1]]
    )
    d_bayes = int(d_bayes)
    clf_bayes.fit(X_train.iloc[:, :d_bayes], y_train)

    joblib.dump(clf_bayes, os.path.join(bin_path, 'base_bayes_clf_' + name))

    # determine thresholds
    y_pred_pro = (
        clf_bayes.predict_proba(X_test.iloc[:, :d_bayes])[:, 1]
        if hasattr(clf_bayes, 'predict_proba')
        else clf_bayes.decision_function(X_test.iloc[:, :d_bayes])
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_pro)
    test_roc_bayes.loc['fpr', name] = fpr.tolist()
    test_roc_bayes.loc['tpr', name] = tpr.tolist()
    test_roc_bayes.loc['thresholds', name] = thresholds.tolist()

    Youden_index = np.argmax(tpr - fpr)
    y_pred = y_pred_pro >= thresholds[Youden_index]

    # bayes results
    test_result_bayes = test_result_bayes.append(
        dict(
            zip(
                metrics,
                [
                    name,
                    d_bayes,
                    accuracy_score(y_test, y_pred),
                    roc_auc_score(y_test, y_pred_pro),
                    matthews_corrcoef(y_test, y_pred),
                    tpr[Youden_index],  # sn
                    1 - fpr[Youden_index],  # sp
                    balanced_accuracy_score(y_test, y_pred),
                ],
            )
        ),
        ignore_index=True,
    )

    test_result_bayes.to_csv(
        os.path.join(data_path, 'base_bayes_test_results.csv'), index=False
    )

test_roc_grid.to_csv(os.path.join(data_path, 'base_grid_test_roc.csv'))
test_roc_bayes.to_csv(os.path.join(data_path, 'base_bayes_test_roc.csv'))


# Comparison between different algorithms
# svm cv
# # inspired by orthogonal experimental design, but for ref only
# n=#choices >=2, if n>=12, then n=12, else n=n  # 12=n_jobs in CLfSearchCV defined
svm_params = [
    # 1*(12-1) + 1 = 12 --> 12
    ({'kernel': ['linear'], 'C': (0.5, 131072.0, 'log-uniform')}, 12),
    (
        {
            'kernel': ['sigmoid', 'rbf'],
            'C': (0.5, 131072.0, 'log-uniform'),  # 2^17
            'gamma': (0.0000152587890625, 4.0, 'log-uniform'),  # 2^(-16)
        },
        24,  # 1*(2-1) + 2*(12-1) + 1 = 24 --> 24
    ),
    (
        {
            'kernel': ['poly'],
            'C': (0.5, 131072.0, 'log-uniform'),
            'gamma': (0.0000152587890625, 4.0, 'log-uniform'),
            'degree': (1, 5),
        },
        28,  # 2*(12-1) + 1*(5-1) + 1 = 27 --> 24/28 --> 28
    ),
]

# decision tree cv
dt_params = {
    'criterion': ['gini', 'entropy'],
    'max_depth': (5, 150),
    'min_samples_split': (2, 30),
    'min_samples_leaf': (1, 10),
    'max_leaf_nodes': (50, 200),
    'ccp_alpha': (1e-10, 1.0, 'log-uniform'),
}  # 1*(2-1) + 4*(12-1) + 1*(10-1) + 1 = 55 --> 52/56 --> 56

# naive bayes cv
nb_params = {'var_smoothing': (1e-12, 100.0, 'log-uniform')}
# 1*(12-1) + 1 = 12 --> 12

# AdaBoost
abc_params = {
    'n_estimators': (10, 1000, 'log-uniform'),
    'learning_rate': (1e-3, 2.0, 'log-uniform'),
}  # 2*(12-1) + 1 = 23 --> 20/24 --> 24


svm_bayes = SVM_BayesCV(svm_params)
dt_bayes = DT_BayesCV(dt_params)
nb_bayes = NB_BayesCV(nb_params)
abc_bayes = ABC_BayesCV(abc_params)

# all clfs IFS processes with d of P/D/C/Q (best features)
feature_method = (
    test_result_bayes.sort_values(by='AUC', ascending=True)
    .dropna()
    .iloc[-1, 0]
)

X_train = X_train_scaled[feature_sig_columns[feature_method]]
X_test = X_test_scaled[feature_sig_columns[feature_method]]

column_ = ['dimension', 'auc_mean', 'auc_std', 'estimator']
metrics = ['classifier', 'dimension', 'OA', 'AUC', 'MCC', 'Sn', 'Sp', 'AA']
clfs = ['SVM', 'DT', 'NB', 'ABC']
clfs_ = dict(zip(clfs, [svm_bayes, dt_bayes, nb_bayes, abc_bayes]))

test_results = pd.DataFrame([], columns=metrics)
test_roc = pd.DataFrame([], index=['fpr', 'tpr', 'thresholds'], columns=clfs)
for clf_ in clfs:
    print(
        '-------------%s using %s IFS process -------------'
        % (feature_method, clf_)
    )

    train_results = pd.DataFrame([], columns=column_)
    for i in range(1, X_train.shape[0] + 1):
        print('------------%d feature(s)------------' % i)

        try:
            clf_bayes = clfs_[clf_].fit(
                X_train=X_train.iloc[:, :i], y_train=y_train
            )
            cv_metric = [
                clf_bayes.best_score_,
                clf_bayes.cv_results_['std_test_score'][clf_bayes.best_index_],
            ]

            train_results = train_results.append(
                dict(
                    zip(
                        column_, [i] + cv_metric + [clf_bayes.best_estimator_],
                    )
                ),
                ignore_index=True,
            )

        except Exception as e:
            print('Retry Failed! ', repr(e))
            train_results = train_results.append(
                dict(zip(column_, [i] + [np.NaN] * (len(column_) - 1))),
                ignore_index=True,
            )
        time.sleep(1)

    train_results.to_csv(
        os.path.join(ifs_path, clf_ + '_bayes_ifs.csv'), index=False
    )

    # test
    d, clf = (
        train_results.sort_values(
            by=['auc_mean', 'dimension'], ascending=[True, False]
        )
        .dropna()
        .iloc[-1, [0, -1]]
    )
    d = int(d)
    clf.fit(X_train.iloc[:, :d], y_train)
    joblib.dump(clf, os.path.join(bin_path, clf_ + '_bayes'))

    # determine the thresholds
    y_pred_pro = (
        clf.predict_proba(X_test.iloc[:, :d])[:, 1]
        if hasattr(clf, 'predict_proba')
        else clf.decision_function(X_test.iloc[:, :d])
    )

    fpr, tpr, thresholds = roc_curve(y_test, y_pred_pro)
    test_roc.loc['fpr', clf_] = fpr.tolist()
    test_roc.loc['tpr', clf_] = tpr.tolist()
    test_roc.loc['thresholds', clf_] = thresholds.tolist()

    Youden_index = np.argmax(tpr - fpr)
    y_pred = y_pred_pro >= thresholds[Youden_index]

    # results
    test_results = test_results.append(
        dict(
            zip(
                metrics,
                [
                    clf_,
                    d,
                    accuracy_score(y_test, y_pred),
                    roc_auc_score(y_test, y_pred_pro),
                    matthews_corrcoef(y_test, y_pred),
                    tpr[Youden_index],  # sn
                    1 - fpr[Youden_index],  # sp
                    balanced_accuracy_score(y_test, y_pred),
                ],
            )
        ),
        ignore_index=True,
    )
    test_results.to_csv(
        os.path.join(data_path, 'clfs_bayes_test_results.csv'), index=False
    )

test_roc.to_csv(os.path.join(data_path, 'clfs_test_roc.csv'))
with open(os.path.join(data_path, 'clfs_test_roc.csv'), 'a') as f:
    f.write('\n' + feature_method)
