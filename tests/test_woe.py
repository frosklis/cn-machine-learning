"""Test the woe transformer"""
import pickle

import numpy as np

from cnml.woe import WoETransformer


def test_woe_numeric_no_missing(titanic):
    """Numeric data, no missings"""
    # This shouldn't fail
    data = titanic.dropna()
    target_var = 'Survived'
    numeric_vars = ['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']
    X = data[numeric_vars]
    y = data[target_var]

    woe_transformer = WoETransformer()
    woe_transformer.fit(X, y)

    woe_transformer.transform(X)
    assert True


def test_woe_numeric_with_missing(titanic):
    """Numeric data with missings"""
    data = titanic
    target_var = 'Survived'
    numeric_vars = ['Age', 'Pclass', 'SibSp', 'Parch', 'Fare']
    X = data[numeric_vars]
    y = data[target_var]

    woe_transformer = WoETransformer()
    woe_transformer.fit(X, y)

    woe_transformer.transform(X)
    assert True


def test_woe_mixed_vars(titanic):
    """Numeric and string data, with missings"""
    data = titanic
    target_var = 'Survived'
    explanatory = ['Age', 'Sex']
    X = data[explanatory]
    y = data[target_var]

    woe_transformer = WoETransformer()
    woe_transformer.fit(X, y)

    woe_transformer.transform(X)
    assert True


def test_woe_pickle(titanic, tmpdir):
    """Train, save, load and check the transformer still works"""
    data = titanic
    target_var = 'Survived'
    explanatory = ['Age', 'Sex']
    X = data[explanatory]
    y = data[target_var]

    woe_transformer = WoETransformer()
    woe_transformer.fit(X, y)

    saved = pickle.dumps(woe_transformer)
    loaded = pickle.loads(saved)

    assert np.allclose(woe_transformer.transform(X),
                       loaded.transform(X))

    path = tmpdir.join("model.pkl")
    with open(path, 'wb') as f:
        pickle.dump(woe_transformer, f)
    with open(path, 'rb') as f:
        loaded = pickle.load(f)

    assert np.allclose(woe_transformer.transform(X),
                       loaded.transform(X))
