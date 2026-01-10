import React from 'react';

interface Signal {
  coin: string;
  price: number;
  price_source: string;
  win_probability: number;
  loss_probability: number;
  verdict: string;
  confidence: string;
  signal_strength: string;
  warnings_count: number;
}

interface Props {
  signal: Signal;
  onClick: () => void;
  isSelected: boolean;
  verdictColor: string;
}

const SignalCard: React.FC<Props> = ({ signal, onClick, isSelected, verdictColor }) => {
  const formatPrice = (price: number): string => {
    if (price < 0.001) return `$${price.toFixed(8)}`;
    if (price < 1) return `$${price.toFixed(6)}`;
    if (price < 1000) return `$${price.toFixed(2)}`;
    return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  };

  const getCoinEmoji = (coin: string): string => {
    if (coin.includes('BTC')) return '₿';
    if (coin.includes('ETH')) return 'Ξ';
    if (coin.includes('SOL')) return '◎';
    if (coin.includes('PEPE')) return '🐸';
    return '🪙';
  };

  const getSignalStrengthIcon = (strength: string): string => {
    switch (strength) {
      case 'STRONG': return '🟢🟢🟢';
      case 'MODERATE': return '🟢🟢';
      case 'WEAK': return '🟡';
      case 'NEUTRAL': return '⚪';
      case 'DANGER': return '🔴🔴🔴';
      default: return '⚪';
    }
  };

  const edge = (signal.win_probability * 5) - (signal.loss_probability * 3);

  return (
    <div 
      className={`signal-card ${isSelected ? 'selected' : ''}`}
      onClick={onClick}
      style={{ borderColor: isSelected ? verdictColor : 'transparent' }}
    >
      {/* Header */}
      <div className="card-header">
        <span className="coin-emoji">{getCoinEmoji(signal.coin)}</span>
        <span className="coin-name">{signal.coin.replace('_', '/')}</span>
        <span 
          className="verdict-badge"
          style={{ backgroundColor: verdictColor }}
        >
          {signal.verdict}
        </span>
      </div>

      {/* Price */}
      <div className="card-price">
        <span className="price">{formatPrice(signal.price)}</span>
        <span className={`price-source ${signal.price_source.toLowerCase()}`}>
          {signal.price_source}
        </span>
      </div>

      {/* Probabilities */}
      <div className="card-probabilities">
        <div className="prob-bar">
          <div className="prob-label">
            <span>WIN</span>
            <span className="prob-value win">{(signal.win_probability * 100).toFixed(1)}%</span>
          </div>
          <div className="prob-track">
            <div 
              className="prob-fill win"
              style={{ width: `${signal.win_probability * 100}%` }}
            />
          </div>
        </div>
        <div className="prob-bar">
          <div className="prob-label">
            <span>LOSS</span>
            <span className="prob-value loss">{(signal.loss_probability * 100).toFixed(1)}%</span>
          </div>
          <div className="prob-track">
            <div 
              className="prob-fill loss"
              style={{ width: `${signal.loss_probability * 100}%` }}
            />
          </div>
        </div>
      </div>

      {/* Edge */}
      <div className="card-edge">
        <span className="edge-label">Edge:</span>
        <span className={`edge-value ${edge >= 0 ? 'positive' : 'negative'}`}>
          {edge >= 0 ? '+' : ''}{edge.toFixed(2)}%
        </span>
      </div>

      {/* Footer */}
      <div className="card-footer">
        <span className="signal-strength">
          {getSignalStrengthIcon(signal.signal_strength)}
        </span>
        <span className="confidence">{signal.confidence}</span>
        {signal.warnings_count > 0 && (
          <span className="warnings-badge">⚠️ {signal.warnings_count}</span>
        )}
      </div>
    </div>
  );
};

export default SignalCard;