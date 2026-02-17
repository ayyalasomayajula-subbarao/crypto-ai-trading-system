import React, { useState, useEffect } from 'react';
import axios from 'axios';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface DetailData {
  coin: string;
  timestamp: string;
  price: number;
  price_source: string;
  verdict: string;
  confidence: string;
  win_probability: number;
  loss_probability: number;
  signal_strength: string;
  reasoning: string[];
  warnings: string[];
  risk: {
    position_size_pct: number;
    position_size_usd: number;
    stop_loss_pct: number;
    stop_loss_price: number;
    take_profit_pct: number;
    take_profit_price: number;
    max_loss_usd: number;
    action: string;
  };
  forecast: {
    direction: string;
    current_price: number;
    bull_target: number;
    bear_target: number;
    probabilities: {
      up: number;
      sideways: number;
      down: number;
    };
  };
  market_context: {
    regime: {
      regime: string;
      adx: number;
      volatility: string;
      recommendation: string;
    };
    shock: {
      shock_detected: boolean;
      type: string | null;
      severity: string;
    };
  };
}

interface Props {
  coin: string;
}

const CoinDetail: React.FC<Props> = ({ coin }) => {
  const [data, setData] = useState<DetailData | null>(null);
  const [loading, setLoading] = useState(true);
  const [capital, setCapital] = useState(1000);

  useEffect(() => {
    const fetchDetail = async () => {
      try {
        setLoading(true);
        const response = await axios.get<DetailData>(
          `${API_BASE}/analyze/${coin}?capital=${capital}`
        );
        setData(response.data);
      } catch (err) {
        console.error(err);
      } finally {
        setLoading(false);
      }
    };

    fetchDetail();
  }, [coin, capital]);

  const formatPrice = (price: number): string => {
    if (price < 0.001) return `$${price.toFixed(8)}`;
    if (price < 1) return `$${price.toFixed(6)}`;
    return `$${price.toLocaleString(undefined, { maximumFractionDigits: 2 })}`;
  };

  if (loading) {
    return <div className="coin-detail loading">Loading details...</div>;
  }

  if (!data) {
    return <div className="coin-detail error">Failed to load details</div>;
  }

  return (
    <div className="coin-detail">
      <h3>📋 {coin.replace('_', '/')} Analysis</h3>

      {/* Capital Input */}
      <div className="capital-input">
        <label>Capital: $</label>
        <input
          type="number"
          value={capital}
          onChange={(e) => setCapital(Number(e.target.value))}
          min={100}
          step={100}
        />
      </div>

      {/* Verdict */}
      <div className={`detail-verdict ${data.verdict.toLowerCase()}`}>
        <span className="verdict-text">{data.verdict}</span>
        <span className="confidence-text">{data.confidence} confidence</span>
      </div>

      {/* Reasoning */}
      <div className="detail-section">
        <h4>💭 Reasoning</h4>
        <ul className="reasoning-list">
          {data.reasoning.map((reason, i) => (
            <li key={i}>{reason}</li>
          ))}
        </ul>
      </div>

      {/* Warnings */}
      {data.warnings.length > 0 && (
        <div className="detail-section warnings">
          <h4>⚠️ Warnings</h4>
          <ul className="warnings-list">
            {data.warnings.map((warning, i) => (
              <li key={i}>{warning}</li>
            ))}
          </ul>
        </div>
      )}

      {/* Forecast */}
      <div className="detail-section">
        <h4>🔮 Forecast ({data.forecast.direction})</h4>
        <div className="forecast-targets">
          <div className="target bull">
            <span className="target-label">Bull Target</span>
            <span className="target-price">{formatPrice(data.forecast.bull_target)}</span>
          </div>
          <div className="target current">
            <span className="target-label">Current</span>
            <span className="target-price">{formatPrice(data.forecast.current_price)}</span>
          </div>
          <div className="target bear">
            <span className="target-label">Bear Target</span>
            <span className="target-price">{formatPrice(data.forecast.bear_target)}</span>
          </div>
        </div>
      </div>

      {/* Risk Management */}
      {data.risk.action === 'OPEN_POSITION' && (
        <div className="detail-section">
          <h4>⚖️ Risk Management</h4>
          <div className="risk-grid">
            <div className="risk-item">
              <span className="risk-label">Position Size</span>
              <span className="risk-value">
                ${data.risk.position_size_usd.toFixed(0)} ({data.risk.position_size_pct}%)
              </span>
            </div>
            <div className="risk-item">
              <span className="risk-label">Stop Loss</span>
              <span className="risk-value loss">
                {formatPrice(data.risk.stop_loss_price)} (-{data.risk.stop_loss_pct}%)
              </span>
            </div>
            <div className="risk-item">
              <span className="risk-label">Take Profit</span>
              <span className="risk-value win">
                {formatPrice(data.risk.take_profit_price)} (+{data.risk.take_profit_pct}%)
              </span>
            </div>
            <div className="risk-item">
              <span className="risk-label">Max Loss</span>
              <span className="risk-value loss">${data.risk.max_loss_usd.toFixed(2)}</span>
            </div>
          </div>
        </div>
      )}

      {/* Market Regime */}
      <div className="detail-section">
        <h4>📊 Market Regime</h4>
        <div className="regime-info">
          <span className={`regime-badge ${data.market_context.regime.regime.toLowerCase()}`}>
            {data.market_context.regime.regime}
          </span>
          <span className="adx">ADX: {data.market_context.regime.adx}</span>
          <span className="volatility">{data.market_context.regime.volatility}</span>
        </div>
        <p className="regime-recommendation">{data.market_context.regime.recommendation}</p>
      </div>
    </div>
  );
};

export default CoinDetail;