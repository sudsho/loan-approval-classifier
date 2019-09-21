import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from src.model import get_model


def test_get_logreg():
    m = get_model('logreg')
    assert isinstance(m, LogisticRegression)


def test_get_rf_with_params():
    m = get_model('random_forest', {'n_estimators': 10, 'random_state': 0})
    assert isinstance(m, RandomForestClassifier)
    assert m.n_estimators == 10


def test_get_gb_alias():
    m = get_model('gb')
    assert isinstance(m, GradientBoostingClassifier)


def test_get_unknown_raises():
    with pytest.raises(ValueError):
        get_model('xgboost')
