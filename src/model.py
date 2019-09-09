"""Model factory for loan approval classifier."""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


def get_model(name, params=None):
    params = params or {}
    name = name.lower()
    if name == 'logreg' or name == 'logistic_regression':
        return LogisticRegression(**params)
    if name == 'random_forest' or name == 'rf':
        return RandomForestClassifier(**params)
    if name == 'gradient_boosting' or name == 'gb':
        return GradientBoostingClassifier(**params)
    raise ValueError("unknown model: %s" % name)
