"""Evaluation helpers."""
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)


def report(y_true, y_pred, y_proba=None):
    out = {}
    out['accuracy'] = float(accuracy_score(y_true, y_pred))
    out['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    out['classification_report'] = classification_report(
        y_true, y_pred, target_names=['N', 'Y'], output_dict=True,
    )
    if y_proba is not None:
        out['roc_auc'] = float(roc_auc_score(y_true, y_proba))
    return out


def pretty_print(rep):
    print('accuracy: %.4f' % rep['accuracy'])
    if 'roc_auc' in rep:
        print('roc auc:  %.4f' % rep['roc_auc'])
    print('confusion matrix:')
    for row in rep['confusion_matrix']:
        print(' ', row)
