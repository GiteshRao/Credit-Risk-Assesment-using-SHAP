"""One-off migration: re-save credit_risk_model.pkl in a layout current xgboost can read.

The committed pickle stores every booster in the serialized container produced by
an older xgboost build.  xgboost 3.4.1's ``Booster.__setstate__`` (backed by
``XGBoosterUnserializeFromBuffer``) rejects that container, so a plain
``joblib.load`` fails with "input stream corrupted" / "Invalid cast, from Null to
Object".

The ``Model`` member of the container already holds the current
``{"learner": ...}`` document, which ``Booster.load_model`` reads without any
trouble.  So we rebuild each booster through ``load_model``, then re-dump the
pipeline.  The rewritten pickle uses 3.4.1's own container format and therefore
loads with a plain ``joblib.load`` -- no shim needed at runtime.
"""

import shutil
import sys
import tempfile
from pathlib import Path

import joblib
import pandas as pd
import xgboost.core as xc

LEGACY_PREFIX = b"{L\x00\x00\x00\x00\x00\x00\x00\x06Config"
FEATURES = [
    "person_age", "person_income", "person_home_ownership", "person_emp_length",
    "loan_intent", "loan_grade", "loan_amnt", "loan_int_rate",
    "loan_percent_income", "cb_person_default_on_file", "cb_person_cred_hist_length",
]

_rebuilt = 0


def _setstate_via_load_model(self, state):
    """Rebuild the booster handle with load_model() instead of the broken C unserializer."""
    global _rebuilt
    state = dict(state)
    handle = state.get("handle")
    if handle is not None:
        raw = bytes(handle)
        if raw.startswith(LEGACY_PREFIX):
            start = raw.find(b"Model") + len(b"Model")
            unwrapped = raw[start:-1]
            if unwrapped[:1] != b"{" or unwrapped[-1:] != b"}":
                raise RuntimeError("unexpected legacy booster layout")
            raw = unwrapped
        booster = xc.Booster()
        booster.load_model(bytearray(raw))
        state["handle"] = booster.handle
        booster.handle = None  # ownership moves to the object being restored
        _rebuilt += 1
    self.__dict__.update(state)


def main():
    model_path = Path("credit_risk_model.pkl")
    threshold = joblib.load("best_threshold.pkl")
    print(f"threshold = {threshold:.6f}")

    xc.Booster.__setstate__ = _setstate_via_load_model
    model = joblib.load(model_path)
    print(f"loaded {type(model).__name__} | rebuilt boosters: {_rebuilt}")

    sample = pd.read_csv("credit_risk_dataset.csv").head(3)[FEATURES]
    print(f"predict_proba on 3 labelled rows: {model.predict_proba(sample)[:, 1]}")

    backup = Path(tempfile.gettempdir()) / "credit_risk_model.pkl.legacy.bak"
    shutil.copy2(model_path, backup)
    print(f"legacy pickle backed up to: {backup}")

    # _setstate_via_load_model is still in place, but __getstate__ is untouched, so the
    # dump below writes 3.4.1's native container format.
    staged = Path("credit_risk_model.migrated.pkl")
    joblib.dump(model, staged)
    print(f"staged migrated pickle: {staged} ({staged.stat().st_size} bytes)")


if __name__ == "__main__":
    sys.exit(main())
