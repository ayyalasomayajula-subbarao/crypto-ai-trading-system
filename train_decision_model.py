import pandas as pd
import numpy as np
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import accuracy_score, roc_auc_score, brier_score_loss
import joblib
import os
import warnings
warnings.filterwarnings('ignore')


class DecisionModelTrainer:
    """
    Production Model: 24h Direction Predictor (UP / DOWN / SIDEWAYS)

    Predicts whether the next 24h will be UP (>+3%), DOWN (<-3%), or SIDEWAYS.
    P(UP) is used as the entry signal — simpler and more learnable from
    technical indicators than TP/SL race timing.

    Trade execution still uses TP=5% / SL=3% / 48h limit on real price data.

    LightGBM 3-class with class_weight='balanced' + isotonic calibration.
    Saved as: decision_model_v2.pkl, decision_features_v2.txt
    """

    def __init__(self, coin):
        self.coin = coin
        self.tp_pct = 0.05    # used for EV reporting only
        self.sl_pct = 0.03

    def load_and_prepare_data(self):
        """Load feature CSV with pre-computed direction labels."""
        filepath = f"data/{self.coin}_multi_tf_features.csv"

        if not os.path.exists(filepath):
            print(f"  ❌ File not found: {filepath}")
            return None

        df = pd.read_csv(filepath)
        print(f"  Loaded: {len(df):,} rows")

        # Direction labels already computed during feature engineering
        df['decision_label'] = df['target_direction']
        before = len(df)
        df = df.dropna(subset=['decision_label']).copy()
        dropped = before - len(df)

        print(f"  Dropped {dropped:,} rows (NaN tail) → {len(df):,} training rows")

        dist = df['decision_label'].value_counts(normalize=True) * 100
        print(f"\n  Direction Label Distribution:")
        for label in ['UP', 'SIDEWAYS', 'DOWN']:
            if label in dist.index:
                print(f"    {label}: {dist[label]:.1f}%")

        return df

    def train(self, df):
        """
        Train 3-class direction LightGBM, calibrate with isotonic regression.

        Chronological split: 70% train | 15% calibration | 15% test
        class_weight='balanced' handles the heavy SIDEWAYS imbalance (~65-75%).
        """
        print(f"\n  Training Direction Model (LightGBM UP/DOWN/SIDEWAYS)...")

        drop_cols = ['timestamp', 'target_return', 'target_direction',
                     'open', 'high', 'low', 'close', 'volume', 'decision_label']
        feature_cols = [col for col in df.columns if col not in drop_cols]

        X = df[feature_cols]
        y = df['decision_label']

        # Chronological 70/15/15 split
        n = len(X)
        split_train = int(n * 0.70)
        split_cal   = int(n * 0.85)

        X_train, y_train = X.iloc[:split_train],          y.iloc[:split_train]
        X_cal,   y_cal   = X.iloc[split_train:split_cal], y.iloc[split_train:split_cal]
        X_test,  y_test  = X.iloc[split_cal:],            y.iloc[split_cal:]

        print(f"  Train: {len(X_train):,} | Calibration: {len(X_cal):,} | Test: {len(X_test):,}")

        lgbm = LGBMClassifier(
            n_estimators=500,
            learning_rate=0.02,
            max_depth=6,
            num_leaves=31,
            min_child_samples=50,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        lgbm.fit(X_train, y_train)

        # Isotonic calibration on held-out set
        calibrated = CalibratedClassifierCV(lgbm, cv='prefit', method='isotonic')
        calibrated.fit(X_cal, y_cal)

        # Evaluate on test set
        test_pred  = calibrated.predict(X_test)
        test_proba = calibrated.predict_proba(X_test)
        classes    = list(calibrated.classes_)

        up_idx    = classes.index('UP')
        up_proba  = test_proba[:, up_idx]
        y_test_up = (y_test == 'UP').astype(int)

        train_acc = accuracy_score(y_train, calibrated.predict(X_train))
        test_acc  = accuracy_score(y_test, test_pred)
        auc       = roc_auc_score(y_test_up, up_proba)
        brier     = brier_score_loss(y_test_up, up_proba)

        print(f"\n  {'='*50}")
        print(f"  RESULTS")
        print(f"  {'='*50}")
        print(f"  Train Accuracy : {train_acc*100:.1f}%")
        print(f"  Test Accuracy  : {test_acc*100:.1f}%")
        print(f"  Overfit Gap    : {(train_acc - test_acc)*100:.1f}%")
        print(f"  ROC-AUC (UP)   : {auc:.3f}   (0.5=random, 1.0=perfect)")
        print(f"  Brier Score    : {brier:.3f}   (lower=better calibrated)")

        # Test set distribution
        test_dist = y_test.value_counts(normalize=True) * 100
        print(f"\n  Test label distribution:")
        for lbl in ['UP', 'SIDEWAYS', 'DOWN']:
            if lbl in test_dist.index:
                print(f"    {lbl}: {test_dist[lbl]:.1f}%")

        # Calibration: P(UP) bucket vs actual UP rate
        print(f"\n  CALIBRATION CHECK (Actual UP Rate per P(UP) Bucket):")
        print(f"  {'Predicted':>12}  {'Actual':>8}  Visual")
        print(f"  {'-'*48}")
        frac_pos, mean_pred = calibration_curve(y_test_up, up_proba, n_bins=8)
        for actual, predicted in zip(frac_pos, mean_pred):
            bar = '█' * int(actual * 20)
            gap = abs(actual - predicted)
            quality = '✅' if gap < 0.05 else ('⚠️' if gap < 0.10 else '❌')
            print(f"  Pred {predicted*100:5.1f}%  →  Actual {actual*100:5.1f}%  {bar} {quality}")

        # Threshold analysis: P(UP) ≥ threshold → enter long with TP=5% SL=3%
        print(f"\n  THRESHOLD ANALYSIS (P(UP) as long entry signal):")
        print(f"  {'Threshold':>10}  {'Signals':>8}  {'Coverage':>9}  {'UP Rate':>8}  {'EV':>7}  Go?")
        print(f"  {'-'*60}")
        for threshold in [0.20, 0.25, 0.30, 0.35, 0.40]:
            mask = up_proba >= threshold
            if mask.sum() >= 10:
                actual_up = y_test_up.values[mask].mean()
                n_sig     = mask.sum()
                coverage  = mask.mean() * 100
                ev        = actual_up * self.tp_pct - (1 - actual_up) * self.sl_pct
                status    = '✅' if ev > 0.002 else ('⚠️' if ev > 0 else '❌')
                print(f"  P >= {threshold:.2f}  {n_sig:8,}  {coverage:8.1f}%  "
                      f"{actual_up*100:8.1f}%  {ev*100:+6.2f}%  {status}")

        return calibrated, feature_cols, y_test, up_proba

    def save_model(self, model, feature_cols):
        """Save calibrated binary model as decision_model_v2.pkl"""
        model_dir = f"models/{self.coin}"
        os.makedirs(model_dir, exist_ok=True)

        joblib.dump(model, f"{model_dir}/decision_model_v2.pkl")

        with open(f"{model_dir}/decision_features_v2.txt", 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")

        print(f"\n  Saved: {model_dir}/decision_model_v2.pkl")
        print(f"  Saved: {model_dir}/decision_features_v2.txt")


def train_all_decision_models():
    print("\n" + "#"*70)
    print("  DIRECTION MODELS — UP / DOWN / SIDEWAYS")
    print("  3-class | class_weight=balanced | Isotonic calibration")
    print("#"*70)

    coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
    results = {}

    for coin in coins:
        print(f"\n{'='*70}")
        print(f"  {coin}")
        print(f"{'='*70}")

        trainer = DecisionModelTrainer(coin)

        df = trainer.load_and_prepare_data()
        if df is None:
            continue

        calibrated, feature_cols, y_test, up_proba = trainer.train(df)
        trainer.save_model(calibrated, feature_cols)

        y_test_up = (y_test == 'UP').astype(int)
        results[coin] = {
            'auc':   roc_auc_score(y_test_up, up_proba),
            'brier': brier_score_loss(y_test_up, up_proba),
        }

    # Summary
    print(f"\n{'='*70}")
    print(f"  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Coin':<12}  {'ROC-AUC':>8}  {'Brier':>7}  Calibration")
    print(f"  {'-'*45}")
    for coin, data in results.items():
        cal_quality = 'good' if data['brier'] < 0.20 else ('ok' if data['brier'] < 0.25 else 'poor')
        print(f"  {coin:<12}  {data['auc']:>8.3f}  {data['brier']:>7.3f}  {cal_quality}")

    print(f"\n  Models saved as decision_model_v2.pkl")
    print(f"  Next step: update walk_forward_validation.py for binary labels")
    print(f"{'='*70}")

    return results


if __name__ == "__main__":
    train_all_decision_models()
