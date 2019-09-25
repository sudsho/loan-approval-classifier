from src.evaluate import report


def test_report_basic():
    y_true = [1, 0, 1, 1, 0, 0]
    y_pred = [1, 0, 1, 0, 0, 1]
    rep = report(y_true, y_pred)
    assert 0.0 <= rep['accuracy'] <= 1.0
    assert len(rep['confusion_matrix']) == 2
    assert 'classification_report' in rep


def test_report_with_proba_has_auc():
    y_true = [1, 0, 1, 0]
    y_pred = [1, 0, 1, 0]
    y_proba = [0.9, 0.1, 0.8, 0.2]
    rep = report(y_true, y_pred, y_proba=y_proba)
    assert 0.0 <= rep['roc_auc'] <= 1.0
