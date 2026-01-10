import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

class DecisionModelTrainer:
    """
    Train Model B: Decision Model
    
    Labels based on TP/SL race:
    - WIN: Price hits +5% before -3% (within 48h)
    - LOSS: Price hits -3% before +5%
    - NO_TRADE: Neither hit within 48h
    """
    
    def __init__(self, coin):
        self.coin = coin
        self.tp_pct = 0.05  # 5% take profit
        self.sl_pct = 0.03  # 3% stop loss
        self.time_limit = 48  # 48 hours
    
    def create_decision_labels(self, df):
        """Create WIN/LOSS/NO_TRADE labels based on TP/SL race"""
        print(f"  Creating decision labels (TP={self.tp_pct*100}%, SL={self.sl_pct*100}%, Time={self.time_limit}h)...")
        
        labels = []
        
        for i in range(len(df) - self.time_limit):
            entry_price = df.iloc[i]['close']
            tp_price = entry_price * (1 + self.tp_pct)
            sl_price = entry_price * (1 - self.sl_pct)
            
            # Look at next 48 hours
            future_prices = df.iloc[i+1:i+1+self.time_limit]
            
            tp_hit = False
            sl_hit = False
            tp_time = self.time_limit + 1
            sl_time = self.time_limit + 1
            
            for j, (_, row) in enumerate(future_prices.iterrows()):
                high = row['high']
                low = row['low']
                
                # Check if TP hit
                if high >= tp_price and not tp_hit:
                    tp_hit = True
                    tp_time = j
                
                # Check if SL hit
                if low <= sl_price and not sl_hit:
                    sl_hit = True
                    sl_time = j
            
            # Determine outcome
            if tp_hit and sl_hit:
                # Both hit - which came first?
                if tp_time < sl_time:
                    labels.append('WIN')
                else:
                    labels.append('LOSS')
            elif tp_hit:
                labels.append('WIN')
            elif sl_hit:
                labels.append('LOSS')
            else:
                labels.append('NO_TRADE')
        
        # Pad remaining rows with NaN
        labels.extend([np.nan] * self.time_limit)
        
        return labels
    
    def load_and_prepare_data(self):
        """Load data and create decision labels"""
        filepath = f"data/{self.coin}_multi_tf_features.csv"
        
        if not os.path.exists(filepath):
            print(f"  ❌ File not found: {filepath}")
            return None
        
        df = pd.read_csv(filepath)
        print(f"  Loaded: {len(df):,} rows")
        
        # Create decision labels
        df['decision_label'] = self.create_decision_labels(df)
        
        # Drop rows without labels
        df = df.dropna(subset=['decision_label'])
        
        # Show distribution
        dist = df['decision_label'].value_counts(normalize=True) * 100
        print(f"\n  Decision Label Distribution:")
        for label in ['WIN', 'NO_TRADE', 'LOSS']:
            if label in dist.index:
                print(f"    {label}: {dist[label]:.1f}%")
        
        return df
    
    def train(self, df):
        """Train the decision model"""
        print(f"\n  Training Decision Model...")
        
        # Prepare features
        drop_cols = ['timestamp', 'target_return', 'target_direction', 
                     'open', 'high', 'low', 'close', 'volume', 'decision_label']
        
        feature_cols = [col for col in df.columns if col not in drop_cols]
        
        X = df[feature_cols]
        y = df['decision_label']
        
        # Time-series split
        split_idx = int(len(X) * 0.8)
        
        X_train = X.iloc[:split_idx]
        X_test = X.iloc[split_idx:]
        y_train = y.iloc[:split_idx]
        y_test = y.iloc[split_idx:]
        
        print(f"  Train: {len(X_train):,} | Test: {len(X_test):,}")
        
        # Train Random Forest
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=10,
            min_samples_leaf=30,
            min_samples_split=50,
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        
        train_acc = accuracy_score(y_train, train_pred)
        test_acc = accuracy_score(y_test, test_pred)
        
        print(f"\n  📊 RESULTS:")
        print(f"  {'-'*40}")
        print(f"  Training Accuracy: {train_acc*100:.1f}%")
        print(f"  Test Accuracy: {test_acc*100:.1f}%")
        print(f"  Overfit: {(train_acc-test_acc)*100:.1f}%")
        
        # Per-class analysis
        print(f"\n  Per-Class Performance:")
        print(f"  {'-'*40}")
        
        for label in ['WIN', 'NO_TRADE', 'LOSS']:
            mask = y_test == label
            if mask.sum() > 0:
                correct = (test_pred[mask] == label).sum()
                total = mask.sum()
                acc = correct / total * 100
                print(f"  {label:<10} {acc:>5.1f}% ({correct}/{total})")
        
        # WIN precision (most important!)
        win_predictions = test_pred == 'WIN'
        if win_predictions.sum() > 0:
            win_correct = (y_test[win_predictions] == 'WIN').sum()
            win_precision = win_correct / win_predictions.sum() * 100
            print(f"\n  🎯 WIN Precision: {win_precision:.1f}%")
            print(f"     (When model says WIN, it's right {win_precision:.1f}% of the time)")
        
        return model, feature_cols, X_test, y_test, test_pred
    
    def analyze_trading_signals(self, y_test, predictions, probabilities):
        """Analyze the quality of BUY signals"""
        print(f"\n  📈 TRADING SIGNAL ANALYSIS:")
        print(f"  {'-'*40}")
        
        # Get WIN probability for each prediction
        win_proba = probabilities[:, list(self.model.classes_).index('WIN')]
        
        # Test different thresholds
        for threshold in [0.35, 0.40, 0.45, 0.50]:
            mask = win_proba >= threshold
            if mask.sum() > 0:
                correct = (y_test.values[mask] == 'WIN').sum()
                total = mask.sum()
                precision = correct / total * 100
                
                status = "✅" if precision > 50 else ("👍" if precision > 40 else "⚠️")
                print(f"  WIN prob >= {threshold*100:.0f}%: {total:,} signals, {precision:.1f}% precision {status}")
    
    def save_model(self, model, feature_cols):
        """Save the decision model"""
        model_dir = f"models/{self.coin}"
        os.makedirs(model_dir, exist_ok=True)
        
        joblib.dump(model, f"{model_dir}/decision_model.pkl")
        
        with open(f"{model_dir}/decision_features.txt", 'w') as f:
            for feat in feature_cols:
                f.write(f"{feat}\n")
        
        print(f"\n  💾 Saved: {model_dir}/decision_model.pkl")
        
        # Store model reference for analysis
        self.model = model


def train_all_decision_models():
    """Train decision models for all coins"""
    
    print("\n" + "#"*70)
    print("  TRAINING DECISION MODELS (WIN/LOSS/NO_TRADE)")
    print("#"*70)
    
    coins = ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']
    results = {}
    
    for coin in coins:
        print(f"\n{'='*70}")
        print(f"  {coin}")
        print(f"{'='*70}")
        
        trainer = DecisionModelTrainer(coin)
        
        # Load and prepare data
        df = trainer.load_and_prepare_data()
        if df is None:
            continue
        
        # Train
        model, feature_cols, X_test, y_test, predictions = trainer.train(df)
        
        # Get probabilities for analysis
        probabilities = model.predict_proba(X_test)
        
        # Analyze trading signals
        trainer.model = model
        trainer.analyze_trading_signals(y_test, predictions, probabilities)
        
        # Save
        trainer.save_model(model, feature_cols)
        
        results[coin] = {
            'model': model,
            'accuracy': accuracy_score(y_test, predictions)
        }
    
    # Summary
    print(f"\n{'='*70}")
    print("  SUMMARY: DECISION MODELS")
    print(f"{'='*70}")
    print(f"\n  {'Coin':<12} {'Test Accuracy':<15}")
    print(f"  {'-'*30}")
    
    for coin, data in results.items():
        print(f"  {coin:<12} {data['accuracy']*100:>8.1f}%")
    
    print(f"\n{'='*70}")
    print("🎉 DECISION MODELS TRAINED!")
    print(f"{'='*70}")
    
    return results


if __name__ == "__main__":
    results = train_all_decision_models()