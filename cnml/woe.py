# Tree classifier based on WoE
# woe = \log\frac{1-p}{p}

import numpy as np
from sklearn import preprocessing
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.multiclass import type_of_target


class WoETransformer(TransformerMixin, BaseEstimator):
    """A weight of evidence transformer.

    It uses DecisionTreeClassiffier under the hood (hence the parameters)
    and handles missing values as well.
    It applies the weight of evidence transformation to categorical
    variables or numerical variables. The weight of
    evidence is defined as:

    .. math::
        \\text{WoE} = \\log\\frac{1-p}{p}

    This transformer deals with new categories in the test data set. See
    :meth:`WoETransformer.transform`

    Parameters
    ----------
    criterion : {"gini", "entropy"}, default="gini"
        The function to measure the quality of a split. Supported criteria are
        "gini" for the Gini impurity and "entropy" for the information gain.
    splitter : {"best", "random"}, default="best"
        The strategy used to choose the split at each node. Supported
        strategies are "best" to choose the best split
        and "random" to choose the best random split.
    max_depth : int, default=None
        The maximum depth of the tree. If None, then nodes are expanded
        until all leaves are pure or until all leaves
         contain less than min_samples_split samples.
    min_samples_split : int or float, default=2
        The minimum number of samples required to split an internal node:
        - If int, then consider `min_samples_split` as the minimum number.
        - If float, then `min_samples_split` is a fraction and `ceil(
        min_samples_split * n_samples)` are the minimum
          number of samples for each split.
    min_samples_leaf : int or float, default=1
        The minimum number of samples required to be at a leaf node.
        A split point at any depth will only be considered if it leaves at
        least ``min_samples_leaf`` training samples in each of the left and
        right branches.  This may have the effect of smoothing the model,
        especially in regression.
        - If int, then consider `min_samples_leaf` as the minimum number.
        - If float, then `min_samples_leaf` is a fraction and `ceil(
        min_samples_leaf * n_samples)` are the minimum
          number of samples for each node.
    min_weight_fraction_leaf : float, default=0.0
        The minimum weighted fraction of the sum total of weights (of all
        the input samples) required to be at a leaf node. Samples have
        equal weight when sample_weight is not provided.
    max_features : int, float or {"auto", "sqrt", "log2"}, default=None
        The number of features to consider when looking for the best split:
            - If int, then consider `max_features` features at each split.
            - If float, then `max_features` is a fraction and
              `int(max_features * n_features)` features are considered at each
              split.
            - If "auto", then `max_features=sqrt(n_features)`.
            - If "sqrt", then `max_features=sqrt(n_features)`.
            - If "log2", then `max_features=log2(n_features)`.
            - If None, then `max_features=n_features`.
        Note: the search for a split does not stop until at least one
        valid partition of the node samples is found, even if it requires to
        effectively inspect more than ``max_features`` features.
    random_state : int or RandomState, default=None
        If int, random_state is the seed used by the random number generator;
        If RandomState instance, random_state is the random number generator;
        If None, the random number generator is the RandomState instance used
        by `np.random`.
    max_leaf_nodes : int, default=None
        Grow a tree with ``max_leaf_nodes`` in best-first fashion.
        Best nodes are defined as relative reduction in impurity.
        If None then unlimited number of leaf nodes.
    min_impurity_decrease : float, default=0.0
        A node will be split if this split induces a decrease of the impurity
        greater than or equal to this value.
        The weighted impurity decrease equation is the following::
            N_t / N * (impurity - N_t_R / N_t * right_impurity
                                - N_t_L / N_t * left_impurity)
        where ``N`` is the total number of samples, ``N_t`` is the number of
        samples at the current node, ``N_t_L`` is the number of samples in the
        left child, and ``N_t_R`` is the number of samples in the right child.
        ``N``, ``N_t``, ``N_t_R`` and ``N_t_L`` all refer to the weighted sum,
        if ``sample_weight`` is passed.
    class_weight : dict, list of dict or "balanced", default=None
        Weights associated with classes in the form ``{class_label: weight}``.
        If None, all classes are supposed to have weight one. For
        multi-output problems, a list of dicts can be provided in the same
        order as the columns of y.
        Note that for multioutput (including multilabel) weights should be
        defined for each class of every column in its own dict. For example,
        for four-class multilabel classification weights should be
        [{0: 1, 1: 1}, {0: 1, 1: 5}, {0: 1, 1: 1}, {0: 1, 1: 1}] instead of
        [{1:1}, {2:5}, {3:1}, {4:1}].
        The "balanced" mode uses the values of y to automatically adjust
        weights inversely proportional to class frequencies in the input data
        as ``n_samples / (n_classes * np.bincount(y))``
        For multi-output, the weights of each column of y will be multiplied.
        Note that these weights will be multiplied with sample_weight (passed
        through the fit method) if sample_weight is specified.
    ccp_alpha : non-negative float, default=0.0
        Complexity parameter used for Minimal Cost-Complexity Pruning. The
        subtree with the largest cost complexity that is smaller than
        ``ccp_alpha`` will be chosen. By default, no pruning is performed. 
    """

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
                 class_weight=None,
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
        self.class_weight = class_weight
        self.ccp_alpha = ccp_alpha

    def fit(self, X, y, num_processes: int = 1):
        """Fits each of the columns in X independently

        Args:
            X ({array-like, sparse matrix} of shape (n_samples,
            n_features)): The training input samples. Internally, it will be
            converted to
                ``dtype=np.float32`` and if a sparse matrix is provided to a
                sparse ``csc_matrix``.
            y (array-like of shape (n_samples,)): The target values (class
                labels) as integers or strings. This only works for binary
                classes.
            num_processes (int): Number of processes to use

        Returns:
            WoETransformer: **self** -- Fitted transformer.
        """

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
            le = preprocessing.LabelEncoder()
            try:
                x = np.array(X)[:, [i]].astype(float)
            except ValueError:
                # Treat as string
                is_numeric = False

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

        if num_processes > 1:
            from multiprocessing import Pool
            pool = Pool(processes=num_processes)
            self._trees = pool.map(fit_var_i, range(self.n_features_))
        else:
            self._trees = list(map(fit_var_i, range(self.n_features_)))

        return self

    def transform(self, X):
        """Applies the WoE transformation

        For categorical variables if new categories appear the mean WoE is
        applied.

        Args:
            X (array like): input data

        Returns: (array like of shape X.shape)
            transformed array
        """
        result = None
        for i in range(self.n_features_):
            tree, nan_woe = self._trees[i]
            xi = np.array(X)[:, [i]]
            try:
                xi = xi.astype(float)
            except ValueError:
                # Variable is not numeric
                (_, le), (_, tree) = tree.steps
                # ... but new categories might have appeared so the
                # transformer cannot be applied directly
                le_dict = dict(zip(le.classes_, le.transform(le.classes_)))
                vf = np.vectorize(lambda x: le_dict.get(x, np.nan))
                xi = vf(xi)
            # Score woe
            nan = np.isnan(xi)[:, 0]
            if np.sum(~nan) > 0:
                # Can't predict probability with empty arrays
                log_proba = tree.predict_proba(xi[~nan])
                woe = log_proba[:, 0] - log_proba[:, 1]  # the woe
                woe = np.array([woe]).T
                xi[~nan] = woe

            xi[nan] = nan_woe

            if i == 0:
                result = xi
            else:
                result = np.concatenate((result, xi), axis=1)

        return result
