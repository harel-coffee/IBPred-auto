import os
import time
import random
import numpy as np
from sklearn.svm import SVC
from skopt import BayesSearchCV
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_result
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier


random.seed(42)
np.random.seed(42)
random_state = 42
data_path = os.path.join(os.path.abspath('.'), 'datas')
n_points = 2  # no more than 9
n_jobs = 12
cv = StratifiedKFold(n_splits=10)


def write_error_log(clf, search_method, dimension, error_info):
    '''
    example:
    clf='SVM', search_method='Grid', dimension=X_train.shape[1]
    '''
    with open(os.path.join(data_path, 'error_log.txt'), 'a') as f:
        f.write(
            str(clf)
            + ' '
            + str(search_method)
            + ' SearchCV fitting failed at D:'
            + str(dimension)
            + ', and info ↓\n'
        )
        f.write(repr(error_info))
        f.write('\n----------\n')


def is_False(value):
    return value is False


class CLF_CV:
    def __init__(self, CLF, SearchCV, search_spaces):
        '''
        example:
        CLF=SVC(probability=True), CV=BayesSearchCV
        '''
        self.CLF = CLF
        self.SearchCV = SearchCV
        self.search_spaces = search_spaces
        self.cv = cv
        self.n_jobs = n_jobs


class CLF_GridCV(CLF_CV):
    def __init__(self, CLF, search_spaces, SearchCV=GridSearchCV):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.scoring = 'roc_auc'


class SVM_GridCV(CLF_GridCV):
    def __init__(
        self,
        search_spaces,
        CLF=SVC(probability=True, random_state=random_state),
        SearchCV=GridSearchCV,
    ):
        super().__init__(CLF, search_spaces, SearchCV=SearchCV)

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            param_grid=self.search_spaces,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        try:
            clf.fit(X_train, y_train)
            print('SVM GridSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in SVM GridSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='SVM',
                search_method='Grid',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class DT_GridCV(CLF_GridCV):
    def __init__(
        self,
        search_spaces,
        CLF=DecisionTreeClassifier(random_state=random_state),
        SearchCV=GridSearchCV,
    ):
        super().__init__(CLF, search_spaces, SearchCV=SearchCV)

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            param_grid=self.search_spaces,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        try:
            clf.fit(X_train, y_train)
            print('DT GridSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in DT GridSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='DT',
                search_method='Grid',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class NB_GridCV(CLF_GridCV):
    def __init__(self, search_spaces, CLF=GaussianNB(), SearchCV=GridSearchCV):
        super().__init__(CLF, search_spaces, SearchCV=SearchCV)

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            param_grid=self.search_spaces,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        try:
            clf.fit(X_train, y_train)
            print('NB GridSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in NB GridSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='NB',
                search_method='Grid',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class RF_GridCV(CLF_GridCV):
    def __init__(
        self,
        search_spaces,
        CLF=RandomForestClassifier(random_state=random_state),
        SearchCV=GridSearchCV,
    ):
        super().__init__(CLF, search_spaces, SearchCV=SearchCV)

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            param_grid=self.search_spaces,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        try:
            clf.fit(X_train, y_train)
            print('RF GridSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in RF GridSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='RF',
                search_method='Grid',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class ABC_GridCV(CLF_GridCV):
    def __init__(
        self,
        search_spaces,
        CLF=AdaBoostClassifier(random_state=random_state),
        SearchCV=GridSearchCV,
    ):
        super().__init__(CLF, search_spaces, SearchCV=SearchCV)

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            param_grid=self.search_spaces,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            cv=self.cv,
        )
        try:
            clf.fit(X_train, y_train)
            print('ABC GridSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in ABC GridSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='ABC',
                search_method='Grid',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class CLF_BayesCV(CLF_CV):
    def __init__(self, CLF, search_spaces, SearchCV=BayesSearchCV):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.scoring = 'roc_auc'
        self.n_points = n_points
        self.random_state = random_state


class SVM_BayesCV(CLF_BayesCV):
    def __init__(
        self,
        search_spaces,
        CLF=SVC(probability=True, random_state=random_state),
        SearchCV=BayesSearchCV,
    ):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.n_iter = 24

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            search_spaces=self.search_spaces,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_points=self.n_points,
            cv=self.cv,
            random_state=self.random_state,
        )
        try:
            clf.fit(X_train, y_train)
            print('SVM BayesSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in SVM BayesSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='SVM',
                search_method='Bayes',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class DT_BayesCV(CLF_BayesCV):
    def __init__(
        self,
        search_spaces,
        CLF=DecisionTreeClassifier(random_state=random_state),
        SearchCV=BayesSearchCV,
    ):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.n_iter = 56

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            search_spaces=self.search_spaces,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_points=self.n_points,
            cv=self.cv,
            random_state=self.random_state,
        )
        try:
            clf.fit(X_train, y_train)
            print('DT BayesSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in DT BayesSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='DT',
                search_method='Bayes',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class NB_BayesCV(CLF_BayesCV):
    def __init__(
        self, search_spaces, CLF=GaussianNB(), SearchCV=BayesSearchCV
    ):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.n_iter = 12

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            search_spaces=self.search_spaces,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_points=self.n_points,
            cv=self.cv,
            random_state=self.random_state,
        )
        try:
            clf.fit(X_train, y_train)
            print('NB BayesSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in NB BayesSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='NB',
                search_method='Bayes',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class RF_BayesCV(CLF_BayesCV):
    def __init__(
        self,
        search_spaces,
        CLF=RandomForestClassifier(random_state=random_state),
        SearchCV=BayesSearchCV,
    ):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.n_iter = 64

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            search_spaces=self.search_spaces,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_points=self.n_points,
            cv=self.cv,
            random_state=self.random_state,
        )
        try:
            clf.fit(X_train, y_train)
            print('RF BayesSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in RF BayesSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='RF',
                search_method='Bayes',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False


class ABC_BayesCV(CLF_BayesCV):
    def __init__(
        self,
        search_spaces,
        CLF=AdaBoostClassifier(random_state=random_state),
        SearchCV=BayesSearchCV,
    ):
        super().__init__(
            CLF=CLF, SearchCV=SearchCV, search_spaces=search_spaces
        )
        self.n_iter = 24

    @retry(
        retry=retry_if_result(is_False),
        stop=stop_after_attempt(5),
        wait=wait_fixed(20),
    )
    def fit(self, X_train, y_train):
        time_start = time.time()
        clf = self.SearchCV(
            estimator=self.CLF,
            search_spaces=self.search_spaces,
            n_iter=self.n_iter,
            scoring=self.scoring,
            n_jobs=self.n_jobs,
            n_points=self.n_points,
            cv=self.cv,
            random_state=self.random_state,
        )
        try:
            clf.fit(X_train, y_train)
            print('ABC BayesSearchCV time: %.1fs' % (time.time() - time_start))
            print('Estimator:', clf.best_estimator_)
            print(
                'Best score:%.4f±%.4f'
                % (
                    clf.best_score_,
                    clf.cv_results_['std_test_score'][clf.best_index_],
                )
            )
            return clf
        except Exception as e:
            print('There\'s something wrong in ABC BayesSearchCV fitting!')
            retry_number = self.fit.retry.statistics['attempt_number']
            print('Retry fit: ' + str(retry_number))
            write_error_log(
                clf='ABC',
                search_method='Bayes',
                dimension=X_train.shape[1],
                error_info=e,
            )
            return False
