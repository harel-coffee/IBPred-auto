import os
import joblib
import argparse
import numpy as np
import pandas as pd
from PseCKSAAP import PseCKSAAP
from DDE import DDE
from CTD import CTD
from QSOrder import QSOrder

data_path = os.path.join(os.path.abspath('.'), 'datas')
bin_path = os.path.join(data_path, 'bin')
methods = ['P', 'PD', 'PDC', 'PDCQ']
clfs = ['SVM', 'DT', 'NB', 'ABC']
base_grid_models = ['base_grid_clf_' + i for i in methods]
base_grid_d = [122, 146, 173, 148]
base_bayes_models = ['base_bayes_clf_' + i for i in methods]
base_bayes_d = [113, 193, 242, 239]
clf_models = [i + '_bayes' for i in clfs]
clf_d = [229, 11, 243, 254]
models = {
    'base_grid': base_grid_models,
    'base_bayes': base_bayes_models,
    'clf': clf_models,
}
roc_file = {
    'base_grid': 'base_grid_test_roc.csv',
    'base_bayes': 'base_bayes_test_roc.csv',
    'clf': 'clfs_test_roc.csv',
}


def auto_pred(fastas, model='base_bayes_clf_PD', set_num=193):
    fastas_name = os.path.basename(fastas)

    if model in clf_models:
        x = 'clf'
    elif model in base_bayes_models:
        x = 'base_bayes'
    else:
        x = 'base_grid'

    roc_file_ = roc_file[x]
    roc_ = pd.read_csv(os.path.join(data_path, roc_file_), index_col=0)
    name = model.split('_')[-1]
    # check
    name = name if name in methods else model.split('_')[0]
    fpr = np.array(eval(roc_.loc['fpr', name]))
    tpr = np.array(eval(roc_.loc['tpr', name]))
    Youden_index = np.argmax(tpr - fpr)

    # get features
    k, delta, nlag, w = 9, 10, 5, 0.5
    pseck = PseCKSAAP(gap=k, delta=delta)
    pseck.fit(fastas)
    print(fastas_name + ' feature psecksaap √')
    dde = DDE()
    dde.fit(fastas)
    print(fastas_name + ' feature dde √')
    ctd = CTD()
    ctd.fit(fastas)
    print(fastas_name + ' feature ctd √'),
    qso = QSOrder(nlag=nlag, w=w)
    qso.fit(fastas)
    print(fastas_name + ' feature qsorder √')

    # if os.path.exists(os.path.join(os.path.abspath('.'), fastas_name)):
    #     os.remove(os.path.join(os.path.abspath('.'), fastas_name))

    feature = pd.concat(
        (
            pseck.encodings,
            dde.encodings.iloc[:, :],
            ctd.encodings.iloc[:, :],
            qso.encodings.iloc[:, :],
        ),
        axis=1,
        sort=False,
    )

    # import feature rank
    F_sig = pd.read_csv(
        os.path.join(data_path, 'f_score_sig_features_by_methods.csv')
    )['PDCQ']
    feature = feature.loc[:, F_sig.values]

    # scale features
    scaler = joblib.load(os.path.join(bin_path, 'scaler_sig'))
    feature = pd.DataFrame(
        scaler.transform(feature.values),
        index=feature.index,
        columns=F_sig.values,
    )
    PD_columns = pd.read_csv(
        os.path.join(data_path, 'f_score_sig_features_by_methods.csv')
    )['PD'][:set_num]
    feature = feature.loc[:, PD_columns.values]

    model = joblib.load(os.path.join(bin_path, model))
    try:
        label_proba = model.predict_proba(feature.values)
        pred_pro = (
            label_proba[:, 1]
            if hasattr(model, 'predict_proba')
            else model.decision_function(feature.values)
        )
        threshold = eval(roc_.loc['thresholds', name])[Youden_index]
        pred_label = pred_pro >= threshold
        result = pd.DataFrame(
            np.c_[pred_label, label_proba],
            index=feature.index,
            columns=['label(proba_1≥%f)' % threshold, 'proba_0', 'proba_1'],
        )
        return result
    except SystemError:
        print('Prediction happened something wrong')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        usage='USAGE:',
        description='Use the model to predict whether the protein is IBP',
    )
    parser.add_argument(
        '--file', required=True, help='input protein sequence fasta file'
    )
    parser.add_argument("--out", help="the generated descriptor file")
    parser.add_argument(
        "--model",
        default='base_bayes_clf_PD',
        help="the model chosed to identify IBP, default: 'base_bayes_clf_PD'",
    )
    parser.add_argument(
        "--set_num",
        default=193,
        help="the number of features as the model required "
        + "default: 193 (the default model required)"
        + "(check paper for more info)",
    )
    parser.add_argument(
        "--out",
        default='result.csv',
        help="the filename of output",
    )
    args = parser.parse_args()
    model = args.model if args.model is not None else 'base_bayes_clf_PD'
    if model not in clf_models + base_bayes_models + base_grid_models:
        print(
            'Error: the models must be in the allowed list,'
            + ' please see the code!'
        )
        exit()
    set_num = args.set_num if args.set_num is not None else 193
    if int(set_num) not in range(1, 257):
        print('Error: the basic allowed range of set_num is [1, 2, ..., 256]')
        exit()
    output = args.out if args.out is not None else 'result.csv'
    result = auto_pred(args.file, model, set_num)
    result.to_csv(output)
