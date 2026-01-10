import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class CoinModelTrainer:
    """Train separate models for each coin"""
    
    def __init__(self, coin):
        self.coin = coin
        self.rf_model = None
        self.xgb_model = None
        self.xgb_encoder = None
        self.feature_cols = None
    
    def load_data(self):
        """Load multi-timeframe features for this coin"""
        filepath = f"data/{self.coin}_multi_tf_features.csv"
        
        if not os.path.exists(filepath):
            print(f"  ❌ File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"  Loaded: {len(df):,} rows, {len(df.columns)} columns")
        
        return df
    
    def prepare_data(self, df, test_size=0.2):
        """Prepare train/test split"""
        # Drop non-feature columns
        drop_cols = ['timestamp', 'target_return', 'target_direction', 
                     'open', 'high', 'low', 'close', 'volume']
        
        self.feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[self.feature_cols]
        y = df['target_direction']
        
        # Time-series split
        split_idx = int(len(X) * (1 - test_size))
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        print(f"  Features: {len(self.feature_cols)}")
        
        return X_train, X_test, y_train, y_test
    
    def train_random_forest(self, X_train, y_train):
        """Train Random Forest"""
        self.rf_model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=30,
            min_samples_split=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        self.rf_model.fit(X_train, y_train)
        return self.rf_model
    
    def train_xgboost(self, X_train, y_train):
        """Train XGBoost"""
        self.xgb_encoder = LabelEncoder()
        y_encoded = self.xgb_encoder.fit_transform(y_train)
        
        self.xgb_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=7,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=30,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1
        )
        self.xgb_model.fit(X_train, y_encoded)
        return self.xgb_model
    
    def evaluate(self, X_train, y_train, X_test, y_test):
        """Evaluate both models"""
        results = {}
        
        # Random Forest
        rf_train_pred = self.rf_model.predict(X_train)
        rf_test_pred = self.rf_model.predict(X_test)
        rf_train_acc = accuracy_score(y_train, rf_train_pred)
        rf_test_acc = accuracy_score(y_test, rf_test_pred)
        
        results['rf'] = {
            'train_acc': rf_train_acc,
            'test_acc': rf_test_acc,
            'overfit': rf_train_acc - rf_test_acc
        }
        
        # XGBoost
        y_test_encoded = self.xgb_encoder.transform(y_test)
        xgb_train_pred = self.xgb_model.predict(X_train)
        xgb_test_pred = self.xgb_model.predict(X_test)
        xgb_train_acc = accuracy_score(self.xgb_encoder.transform(y_train), xgb_train_pred)
        xgb_test_acc = accuracy_score(y_test_encoded, xgb_test_pred)
        
        results['xgb'] = {
            'train_acc': xgb_train_acc,
            'test_acc': xgb_test_acc,
            'overfit': xgb_train_acc - xgb_test_acc
        }
        
        # Ensemble
        rf_proba = self.rf_model.predict_proba(X_test)
        xgb_proba = self.xgb_model.predict_proba(X_test)
        
        # Align probabilities
        classes = ['DOWN', 'SIDEWAYS', 'UP']
        rf_classes = list(self.rf_model.classes_)
        xgb_classes = list(self.xgb_encoder.classes_)
        
        rf_ordered = np.zeros((len(X_test), 3))
        xgb_ordered = np.zeros((len(X_test), 3))
        
        for i, cls in enumerate(classes):
            if cls in rf_classes:
                rf_ordered[:, i] = rf_proba[:, rf_classes.index(cls)]
            if cls in xgb_classes:
                xgb_ordered[:, i] = xgb_proba[:, xgb_classes.index(cls)]
        
        ensemble_proba = rf_ordered * 0.5 + xgb_ordered * 0.5
        ensemble_pred = np.array([classes[i] for i in ensemble_proba.argmax(axis=1)])
        ensemble_acc = accuracy_score(y_test, ensemble_pred)
        
        results['ensemble'] = {
            'test_acc': ensemble_acc
        }
        
        return results, ensemble_pred, ensemble_proba
    
    def analyze_signals(self, y_test, predictions, probabilities):
        """Analyze trading signals"""
        classes = ['DOWN', 'SIDEWAYS', 'UP']
        confidence = probabilities.max(axis=1)
        y_test_arr = y_test.values if hasattr(y_test, 'values') else y_test
        
        analysis = {}
        
        # Per-class accuracy
        for cls in classes:
            mask = predictions == cls
            if mask.sum() > 0:
                correct = (y_test_arr[mask] == cls).sum()
                total = mask.sum()
                analysis[cls] = {
                    'count': total,
                    'correct': correct,
                    'accuracy': correct / total * 100
                }
        
        # High confidence signals
        for threshold in [0.45, 0.50, 0.55]:
            mask = confidence >= threshold
            if mask.sum() > 0:
                acc = (predictions[mask] == y_test_arr[mask]).mean() * 100
                analysis[f'conf_{int(threshold*100)}'] = {
                    'count': mask.sum(),
                    'accuracy': acc
                }
        
        return analysis
    
    def save_models(self):
        """Save models for this coin"""
        model_dir = f"models/{self.coin}"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(self.rf_model, f"{model_dir}/random_forest.pkl")
        joblib.dump(self.xgb_model, f"{model_dir}/xgboost.pkl")
        joblib.dump(self.xgb_encoder, f"{model_dir}/label_encoder.pkl")
        
        with open(f"{model_dir}/feature_list.txt", 'w') as f:
            for feat in self.feature_cols:
                f.write(f"{feat}\n")
        
        return model_dir


def train_all_coins():
    """Train models for all coins"""
    
    print("\n" + "#"*70)
    print("  TRAINING SEPARATE MODELS PER COIN")
    print("#"*70)
    
    coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
    all_results = {}
    
    for coin in coins:
        print(f"\n{'='*70}")
        print(f"  {coin}")
        print(f"{'='*70}")
        
        trainer = CoinModelTrainer(coin)
        
        # Load data
        df = trainer.load_data()
        if df is None:
            continue
        
        # Prepare data
        X_train, X_test, y_train, y_test = trainer.prepare_data(df)
        
        # Show target distribution
        dist = y_train.value_counts(normalize=True) * 100
        print(f"\n  Target distribution (train):")
        for label in ['UP', 'SIDEWAYS', 'DOWN']:
            if label in dist.index:
                print(f"    {label}: {dist[label]:.1f}%")
        
        # Train models
        print(f"\n  Training Random Forest...")
        trainer.train_random_forest(X_train, y_train)
        
        print(f"  Training XGBoost...")
        trainer.train_xgboost(X_train, y_train)
        
        # Evaluate
        print(f"\n  Evaluating...")
        results, predictions, probabilities = trainer.evaluate(X_train, y_train, X_test, y_test)
        
        # Display results
        print(f"\n  📊 RESULTS:")
        print(f"  {'-'*50}")
        print(f"  {'Model':<15} {'Train Acc':<12} {'Test Acc':<12} {'Overfit':<12}")
        print(f"  {'-'*50}")
        print(f"  {'Random Forest':<15} {results['rf']['train_acc']*100:>8.1f}%    {results['rf']['test_acc']*100:>8.1f}%    {results['rf']['overfit']*100:>8.1f}%")
        print(f"  {'XGBoost':<15} {results['xgb']['train_acc']*100:>8.1f}%    {results['xgb']['test_acc']*100:>8.1f}%    {results['xgb']['overfit']*100:>8.1f}%")
        print(f"  {'Ensemble':<15} {'-':>8}      {results['ensemble']['test_acc']*100:>8.1f}%")
        print(f"  {'-'*50}")
        
        # Analyze signals
        analysis = trainer.analyze_signals(y_test, predictions, probabilities)
        
        print(f"\n  📈 SIGNAL ANALYSIS:")
        print(f"  {'-'*50}")
        
        for cls in ['UP', 'SIDEWAYS', 'DOWN']:
            if cls in analysis:
                a = analysis[cls]
                edge = a['accuracy'] - 33.3
                status = "✅" if edge > 5 else ("👍" if edge > 0 else "⚠️")
                print(f"  {cls:<10} {a['count']:>5} signals, {a['accuracy']:>5.1f}% acc (edge: {edge:+.1f}%) {status}")
        
        print(f"\n  🎯 HIGH-CONFIDENCE SIGNALS:")
        for key in ['conf_45', 'conf_50', 'conf_55']:
            if key in analysis:
                a = analysis[key]
                threshold = key.split('_')[1]
                print(f"  >={threshold}% conf: {a['count']:>5} signals, {a['accuracy']:>5.1f}% accuracy")
        
        # Save models
        model_dir = trainer.save_models()
        print(f"\n  💾 Saved to: {model_dir}/")
        
        all_results[coin] = {
            'results': results,
            'analysis': analysis
        }
    
    # Summary comparison
    print(f"\n{'='*70}")
    print("  SUMMARY: ALL COINS COMPARISON")
    print(f"{'='*70}")
    print(f"\n  {'Coin':<12} {'Ensemble Acc':<15} {'BUY Acc':<12} {'SELL Acc':<12} {'HOLD Acc':<12}")
    print(f"  {'-'*60}")
    
    for coin, data in all_results.items():
        ens_acc = data['results']['ensemble']['test_acc'] * 100
        buy_acc = data['analysis'].get('UP', {}).get('accuracy', 0)
        sell_acc = data['analysis'].get('DOWN', {}).get('accuracy', 0)
        hold_acc = data['analysis'].get('SIDEWAYS', {}).get('accuracy', 0)
        
        print(f"  {coin:<12} {ens_acc:>8.1f}%       {buy_acc:>8.1f}%    {sell_acc:>8.1f}%    {hold_acc:>8.1f}%")
    
    print(f"\n{'='*70}")
    print("🎉 ALL MODELS TRAINED!")
    print(f"{'='*70}")
    
    return all_results


# Main
if __name__ == "__main__":
    results = train_all_coins()