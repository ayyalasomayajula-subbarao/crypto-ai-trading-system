import React from 'react';

interface BTCContext {
  trend_1h: string;
  trend_4h: string;
  trend_1d: string;
  overall_trend: string;
  strength: number;
  support_alts: boolean;
  price: number;
  change_24h: number;
}

interface Props {
  btc: BTCContext;
}

const MarketContext: React.FC<Props> = ({ btc }) => {
  const getTrendIcon = (trend: string): string => {
    switch (trend) {
      case 'UP': return '📈';
      case 'DOWN': return '📉';
      case 'SIDEWAYS': return '➡️';
      case 'BULLISH': return '🟢';
      case 'BEARISH': return '🔴';
      case 'NEUTRAL': return '🟡';
      default: return '❓';
    }
  };

  const getTrendColor = (trend: string): string => {
    switch (trend) {
      case 'UP':
      case 'BULLISH': return '#10b981';
      case 'DOWN':
      case 'BEARISH': return '#ef4444';
      default: return '#f59e0b';
    }
  };

  return (
    <div className="market-context">
      <h3>₿ BTC Context</h3>
      
      {/* BTC Price */}
      <div className="btc-price-section">
        <div className="btc-price">${btc.price.toLocaleString()}</div>
        <div className={`btc-change ${btc.change_24h >= 0 ? 'positive' : 'negative'}`}>
          {btc.change_24h >= 0 ? '+' : ''}{btc.change_24h.toFixed(2)}% (24h)
        </div>
      </div>

      {/* Trend Table */}
      <div className="trend-table">
        <div className="trend-row">
          <span className="trend-label">1H</span>
          <span 
            className="trend-value"
            style={{ color: getTrendColor(btc.trend_1h) }}
          >
            {getTrendIcon(btc.trend_1h)} {btc.trend_1h}
          </span>
        </div>
        <div className="trend-row">
          <span className="trend-label">4H</span>
          <span 
            className="trend-value"
            style={{ color: getTrendColor(btc.trend_4h) }}
          >
            {getTrendIcon(btc.trend_4h)} {btc.trend_4h}
          </span>
        </div>
        <div className="trend-row">
          <span className="trend-label">1D</span>
          <span 
            className="trend-value"
            style={{ color: getTrendColor(btc.trend_1d) }}
          >
            {getTrendIcon(btc.trend_1d)} {btc.trend_1d}
          </span>
        </div>
      </div>

      {/* Overall */}
      <div className="overall-trend">
        <span className="overall-label">Overall:</span>
        <span 
          className="overall-value"
          style={{ color: getTrendColor(btc.overall_trend) }}
        >
          {getTrendIcon(btc.overall_trend)} {btc.overall_trend}
        </span>
      </div>

      {/* Strength */}
      <div className="trend-strength">
        <span className="strength-label">Strength:</span>
        <div className="strength-bar">
          <div 
            className="strength-fill"
            style={{ width: `${btc.strength}%` }}
          />
        </div>
        <span className="strength-value">{btc.strength.toFixed(1)}%</span>
      </div>

      {/* Alts Support */}
      <div className={`alts-support ${btc.support_alts ? 'yes' : 'no'}`}>
        {btc.support_alts ? '✅ Supports Alt Trades' : '⚠️ Alts at Risk'}
      </div>
    </div>
  );
};

export default MarketContext;