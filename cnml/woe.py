# Tree classifier based on WoE
# woe = \log\frac{1-p}{p}

import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target


class WoETransformer(TransformerMixin, BaseEstimator):
    """Weight of evidence transformer"""
    def __init__(self,
                 criterion="gini",
                 splitter="best",
                 max_depth=None,
                 min_samples_split=2,
                 min_samples_leaf=1,
                 min_weight_fraction_leaf=0.,
                 max_features=None,
                 random_state=None,
                 max_leaf_nodes=None,
                 min_impurity_decrease=0.,
                 min_impurity_split=None,
                 class_weight=None,
                 presort='deprecated',
                 ccp_alpha=0.0):
        self.criterion = criterion
        self.splitter = splitter
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_weight_fraction_leaf = min_weight_fraction_leaf
        self.max_features = max_features
        self.random_state = random_state
        self.max_leaf_nodes = max_leaf_nodes
        self.min_impurity_decrease = min_impurity_decrease
        self.min_impurity_split = min_impurity_split
        self.class_weight = class_weight
        self.presort = presort
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, processes=1):

        n_samples, self.n_features_ = X.shape

        if type_of_target(y) != 'binary':
            raise ValueError('Expected binary target')

        y_arr = np.array(y)
        neg_class, pos_class = np.unique(y_arr)
        neg = np.sum(y_arr == neg_class)
        pos = n_samples - neg
        mean_woe = np.log(neg) - np.log(pos)

        def fit_var_i(i):
            # For each variable build a tree
            is_numeric = True
            try:
                x = np.array(X)[:, [i]].astype(float)
            except ValueError:
                # Treat as string
                is_numeric = False
                le = preprocessing.LabelEncoder()
                x = np.array(X)[:, [i]]
                le.fit(x)
                x = np.array([le.transform(x)]).T

            nan = np.isnan(x)[:, 0]
            n_nan = sum(nan)
            tree = DecisionTreeClassifier(**self.get_params())
            tree.fit(x[~nan], y_arr[~nan])

            if not is_numeric:
                tree = Pipeline(steps=[('labeller', le), ('tree', tree)])

            if n_nan > 0:
                neg = np.sum(y_arr == neg_class)
                pos = n_samples - neg
                return tree, np.log(neg + 0.001) - np.log(pos + 0.001)
            else:
                return tree, mean_woe

        if processes > 1:
            from multiprocessing import Pool
            pool = Pool(processes=processes)
            self._trees = pool.map(fit_var_i, range(self.n_features_))
        else:
            self._trees = list(map(fit_var_i, range(self.n_features_)))

        return self

    def transform(self, X):
        result = None
        for i in range(self.n_features_):
            tree, nan_woe = self._trees[i]
            xi = np.array(X)[:, [i]]
            try:
                xi = xi.astype(float)
                nan = np.isnan(xi)[:, 0]
                log_proba = tree.predict_proba(xi[~nan])
                woe = log_proba[:, 0] - log_proba[:, 1]  # the woe
                woe = np.array([woe]).T
                xi[~nan] = woe
                xi[nan] = nan_woe
            except ValueError:
                (_, le) , (_, tree) = tree.steps
                log_proba = tree.predict_proba(np.array([le.transform(xi)]).T)
                woe = log_proba[:, 0] - log_proba[:, 1]  # the woe
                woe = np.array([woe]).T
                xi = woe

            if i == 0:
                result = xi
            else:
                result = np.concatenate((result, xi), axis=1)

        return result
