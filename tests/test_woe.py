from cnml.woe import WoETransformer


def test_woe_numeric_no_missing(titanic):
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
    data = titanic
    target_var = 'Survived'
    explanatory = ['Age', 'Sex']
    X = data[explanatory]
    y = data[target_var]

    woe_transformer = WoETransformer()
    woe_transformer.fit(X, y)

    woe_transformer.transform(X)
    assert True
