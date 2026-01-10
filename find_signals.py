import pandas as pd
import numpy as np
import joblib
import warnings
warnings.filterwarnings('ignore')

def find_high_confidence_signals(coin, threshold=0.45):
    """Find historical periods where model gave BUY signals"""
    
    print(f"\n{'='*60}")
    print(f"  Finding BUY Signals for {coin} (WIN prob >= {threshold*100:.0f}%)")
    print(f"{'='*60}")
    
    # Load model
    model = joblib.load(f"models/{coin}/decision_model.pkl")
    
    with open(f"models/{coin}/decision_features.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    # Load data
    df = pd.read_csv(f"data/{coin}_multi_tf_features.csv")
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Get features
    X = df[feature_cols]
    
    # Get probabilities
    probas = model.predict_proba(X)
    classes = list(model.classes_)
    
    win_idx = classes.index('WIN')
    loss_idx = classes.index('LOSS')
    
    df['win_prob'] = probas[:, win_idx]
    df['loss_prob'] = probas[:, loss_idx]
    
    # Find high WIN probability signals
    signals = df[df['win_prob'] >= threshold].copy()
    
    print(f"\n  Total data points: {len(df):,}")
    print(f"  BUY signals found: {len(signals):,} ({len(signals)/len(df)*100:.1f}%)")
    
    if len(signals) == 0:
        print("  No signals found at this threshold")
        return None
    
    # Analyze signals by time
    signals['year_month'] = signals['timestamp'].dt.to_period('M')
    monthly = signals.groupby('year_month').size()
    
    print(f"\n  📅 Signals by Month (last 12):")
    for period, count in monthly.tail(12).items():
        print(f"     {period}: {count} signals")
    
    # Show most recent signals
    print(f"\n  📊 Most Recent {min(10, len(signals))} BUY Signals:")
    print(f"  {'-'*70}")
    
    recent = signals.tail(10)
    for _, row in recent.iterrows():
        date = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        win_p = row['win_prob'] * 100
        loss_p = row['loss_prob'] * 100
        close = row['close']
        
        print(f"  {date} | WIN: {win_p:>5.1f}% | LOSS: {loss_p:>5.1f}% | Price: {close:.2f}")
    
    # Show the BEST signals (highest WIN prob)
    print(f"\n  🔥 Top 10 HIGHEST Confidence Signals:")
    print(f"  {'-'*70}")
    
    best = signals.nlargest(10, 'win_prob')
    for _, row in best.iterrows():
        date = row['timestamp'].strftime('%Y-%m-%d %H:%M')
        win_p = row['win_prob'] * 100
        loss_p = row['loss_prob'] * 100
        
        print(f"  {date} | WIN: {win_p:>5.1f}% | LOSS: {loss_p:>5.1f}%")
    
    # Current state
    latest = df.tail(1).iloc[0]
    print(f"\n  📍 CURRENT STATE ({latest['timestamp'].strftime('%Y-%m-%d %H:%M')}):")
    print(f"     WIN Probability: {latest['win_prob']*100:.1f}%")
    print(f"     LOSS Probability: {latest['loss_prob']*100:.1f}%")
    
    if latest['win_prob'] >= threshold:
        print(f"     ✅ CURRENTLY A BUY SIGNAL!")
    else:
        print(f"     ⏳ Not a buy signal right now")
        print(f"     Need WIN prob >= {threshold*100:.0f}% (currently {latest['win_prob']*100:.1f}%)")
    
    return signals


def analyze_signal_performance(coin, threshold=0.45):
    """Check how well the signals actually performed"""
    
    print(f"\n{'='*60}")
    print(f"  Signal Performance Analysis: {coin}")
    print(f"{'='*60}")
    
    # Load model and data
    model = joblib.load(f"models/{coin}/decision_model.pkl")
    
    with open(f"models/{coin}/decision_features.txt", 'r') as f:
        feature_cols = [line.strip() for line in f.readlines()]
    
    df = pd.read_csv(f"data/{coin}_multi_tf_features.csv")
    
    # Get probabilities
    X = df[feature_cols]
    probas = model.predict_proba(X)
    classes = list(model.classes_)
    
    win_idx = classes.index('WIN')
    df['win_prob'] = probas[:, win_idx]
    
    # Calculate actual outcomes (did price hit TP or SL first?)
    tp_pct = 0.05  # 5%
    sl_pct = 0.03  # 3%
    time_limit = 48
    
    outcomes = []
    for i in range(len(df) - time_limit):
        entry = df.iloc[i]['close']
        tp_price = entry * (1 + tp_pct)
        sl_price = entry * (1 - sl_pct)
        
        future = df.iloc[i+1:i+1+time_limit]
        
        tp_hit = (future['high'] >= tp_price).any()
        sl_hit = (future['low'] <= sl_price).any()
        
        if tp_hit and not sl_hit:
            outcomes.append('WIN')
        elif sl_hit and not tp_hit:
            outcomes.append('LOSS')
        elif tp_hit and sl_hit:
            # Check which came first
            tp_idx = future[future['high'] >= tp_price].index[0]
            sl_idx = future[future['low'] <= sl_price].index[0]
            outcomes.append('WIN' if tp_idx < sl_idx else 'LOSS')
        else:
            outcomes.append('NO_TRADE')
    
    outcomes.extend([None] * time_limit)
    df['actual_outcome'] = outcomes
    
    # Filter to signals only
    signals = df[(df['win_prob'] >= threshold) & (df['actual_outcome'].notna())]
    
    if len(signals) == 0:
        print("  No signals with outcomes to analyze")
        return
    
    # Calculate actual win rate
    wins = (signals['actual_outcome'] == 'WIN').sum()
    total = len(signals)
    win_rate = wins / total * 100
    
    print(f"\n  Signals at {threshold*100:.0f}% threshold: {total:,}")
    print(f"  Actual WINS: {wins:,}")
    print(f"  Actual WIN RATE: {win_rate:.1f}%")
    
    if win_rate > 50:
        print(f"  ✅ Profitable strategy!")
    else:
        print(f"  ⚠️ Below 50% win rate")
    
    # Test different thresholds
    print(f"\n  📊 Win Rate by Threshold:")
    print(f"  {'-'*40}")
    
    for t in [0.35, 0.40, 0.45, 0.50, 0.55]:
        mask = (df['win_prob'] >= t) & (df['actual_outcome'].notna())
        if mask.sum() > 0:
            wr = (df.loc[mask, 'actual_outcome'] == 'WIN').mean() * 100
            cnt = mask.sum()
            status = "✅" if wr > 50 else "👍" if wr > 40 else "⚠️"
            print(f"  >= {t*100:.0f}%: {cnt:>5} signals, {wr:>5.1f}% win rate {status}")


# Main
if __name__ == "__main__":
    print("\n" + "#"*60)
    print("  FINDING HISTORICAL BUY SIGNALS")
    print("#"*60)
    
    for coin in ['BTC_USDT', 'ETH_USDT', 'SOL_USDT', 'PEPE_USDT']:
        find_high_confidence_signals(coin, threshold=0.45)
        analyze_signal_performance(coin, threshold=0.45)
    
    print("\n" + "#"*60)
    print("  ANALYSIS COMPLETE!")
    print("#"*60)