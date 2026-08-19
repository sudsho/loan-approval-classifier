.PHONY: smoke test train

# Offline end-to-end check: train, print accuracy + roc auc, exercise the
# predict path (direct call + Flask test client). No network, no downloads.
smoke:
	python scripts/smoke.py

test:
	python -m pytest -q

train:
	python -m src.train --config configs/default.yaml
