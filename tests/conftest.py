import os

import pytest
import pandas as pd
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

@pytest.fixture
def titanic():
    train = os.path.join(THIS_DIR, 'test-files/titanic/train.csv')
    return pd.read_csv(train)
    