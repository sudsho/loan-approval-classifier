"""Train a loan approval classifier."""
import os
import argparse
import yaml
import joblib

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from src.data import load_dataset, split
from src.preprocess import (
    impute_missing,
    encode_categoricals,
    encode_target,
    TARGET,
)
from src.model import get_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='configs/default.yaml')
    return p.parse_args()


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    df = load_dataset(cfg['data']['train_path'])

    X_train, X_test, y_train, y_test = split(
        df,
        test_size=cfg['split']['test_size'],
        random_state=cfg['split']['random_state'],
        stratify=cfg['split'].get('stratify', True),
    )

    X_train = impute_missing(X_train)
    X_test = impute_missing(X_test)

    X_train_enc, encoders = encode_categoricals(X_train)
    X_test_enc, _ = encode_categoricals(X_test, encoders=encoders)

    y_train_enc = encode_target(y_train)
    y_test_enc = encode_target(y_test)

    model = get_model(cfg['model']['type'], cfg['model'].get('params'))
    model.fit(X_train_enc, y_train_enc)

    preds = model.predict(X_test_enc)
    acc = accuracy_score(y_test_enc, preds)
    print('test accuracy: %.4f' % acc)
    print('confusion matrix:')
    print(confusion_matrix(y_test_enc, preds))
    print('classification report:')
    print(classification_report(y_test_enc, preds, target_names=['N', 'Y']))

    out_dir = os.path.dirname(cfg['artifacts']['model_path'])
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)
    joblib.dump(model, cfg['artifacts']['model_path'])
    joblib.dump(encoders, cfg['artifacts']['preproc_path'])
    print('saved model to', cfg['artifacts']['model_path'])


if __name__ == '__main__':
    main()
