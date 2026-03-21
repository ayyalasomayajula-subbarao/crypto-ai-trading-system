import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './VerdictTab.css';

// ─── Types ───────────────────────────────────────────────────────────────────

interface SignalItem {
  name: string;
  category: string;
  value: string;
  score: number;
  emoji: string;
  weight: number;
  reason: string;
  w_score: number;
}

interface TradeSetup {
  direction: 'LONG' | 'SHORT';
  entry_price: number;
  stop_loss: number;
  take_profit: number;
  sl_pct: number;
  tp_pct: number;
  position_usd: number;
  position_pct: number;
  max_loss_usd: number;
  risk_reward: number;
  max_hold_h: number;
}

interface VerdictData {
  coin: string;
  timestamp: string;
  price: number;
  verdict: string;
  confidence: number;
  consensus_score: number;
  max_score: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  agreement_pct: number;
  agreement_label: string;
  model_status: string;
  signals: SignalItem[];
  summary: string;
  reasons: string[];
  risks: string[];
  watch_for: string[];
  trade_setup: TradeSetup | null;
  cached: boolean;
}

interface AccuracyStats {
  total: number;
  correct: number;
  accuracy_pct: number;
  by_verdict: Record<string, { total: number; correct: number }>;
}

interface Props {
  coin: string;
  capital: number | '';
  tradeType: string;
}

// ─── Constants ───────────────────────────────────────────────────────────────

const VERDICT_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  STRONG_BUY:  { color: '#00d4aa', bg: 'rgba(0,212,170,0.12)',  label: '⬆⬆ Strong Buy'  },
  BUY:         { color: '#10b981', bg: 'rgba(16,185,129,0.12)', label: '⬆ Buy'          },
  LEAN_BUY:    { color: '#6ee7b7', bg: 'rgba(110,231,183,0.10)',label: '↗ Lean Buy'     },
  HOLD:        { color: '#94a3b8', bg: 'rgba(148,163,184,0.10)',label: '→ Hold'          },
  AVOID:       { color: '#f59e0b', bg: 'rgba(245,158,11,0.12)', label: '⚠ Avoid'        },
  LEAN_SELL:   { color: '#fbbf24', bg: 'rgba(251,191,36,0.10)', label: '↘ Lean Sell'    },
  SELL:        { color: '#ef4444', bg: 'rgba(239,68,68,0.12)',  label: '⬇ Sell'         },
  STRONG_SELL: { color: '#dc2626', bg: 'rgba(220,38,38,0.12)',  label: '⬇⬇ Strong Sell' },
};

const CATEGORY_ORDER = ['MODEL', 'TREND', 'MOMENTUM', 'VOLUME', 'FLOW', 'SENTIMENT', 'MACRO'];
const CATEGORY_LABELS: Record<string, string> = {
  MODEL:     '🤖 ML Model',
  TREND:     '📈 Trend',
  MOMENTUM:  '⚡ Momentum',
  VOLUME:    '📊 Volume',
  FLOW:      '🌊 Derivatives Flow',
  SENTIMENT: '😨 Sentiment',
  MACRO:     '🌍 Macro',
};

const MODEL_STATUS_CONFIG: Record<string, { color: string; label: string }> = {
  MARGINAL:    { color: '#10b981', label: 'MARGINAL ✓' },
  NOT_VIABLE:  { color: '#f59e0b', label: 'NOT VIABLE' },
  NO_MODEL:    { color: '#64748b', label: 'NO MODEL'   },
};

function formatPrice(p: number): string {
  if (!p) return '—';
  if (p < 0.001) return p.toFixed(8);
  if (p < 1)     return p.toFixed(5);
  if (p < 100)   return p.toFixed(3);
  return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
}

// ─── Sub-components ──────────────────────────────────────────────────────────

function VerdictBadge({ data }: { data: VerdictData }) {
  const cfg = VERDICT_CONFIG[data.verdict] || VERDICT_CONFIG['HOLD'];
  const modelCfg = MODEL_STATUS_CONFIG[data.model_status] || MODEL_STATUS_CONFIG['NO_MODEL'];

  const scoreBarWidth = Math.min(100, Math.abs(data.consensus_score / data.max_score) * 100);
  const isPositive = data.consensus_score >= 0;

  return (
    <div className="vt-badge-card" style={{ borderColor: cfg.color, background: cfg.bg }}>
      <div className="vt-badge-main">
        <div className="vt-verdict-label" style={{ color: cfg.color }}>
          {cfg.label}
        </div>
        <div className="vt-confidence">
          <span className="vt-conf-value">{data.confidence}%</span>
          <span className="vt-conf-label">confidence</span>
        </div>
      </div>

      <div className="vt-badge-meta">
        <div className="vt-score-row">
          <span className="vt-score-num">
            Score: {data.consensus_score > 0 ? '+' : ''}{data.consensus_score.toFixed(1)}
            <span className="vt-score-max"> / {data.max_score}</span>
          </span>
          <div className="vt-score-bar-wrap">
            <div
              className={`vt-score-bar ${isPositive ? 'bull' : 'bear'}`}
              style={{ width: `${scoreBarWidth}%` }}
            />
          </div>
        </div>

        <div className="vt-counts">
          <span className="vt-count bull">🟢 {data.bullish_count}</span>
          <span className="vt-count bear">🔴 {data.bearish_count}</span>
          <span className="vt-count neut">⚪ {data.neutral_count}</span>
          <span className="vt-agreement" data-level={(data.agreement_label || 'low').toLowerCase()}>
            {data.agreement_label || '—'} agreement ({data.agreement_pct ?? 0}%)
          </span>
        </div>

        <div className="vt-model-pill" style={{ color: modelCfg.color }}>
          Model: {modelCfg.label}
        </div>
      </div>
    </div>
  );
}

function SignalGrid({ signals }: { signals: SignalItem[] }) {
  const grouped: Record<string, SignalItem[]> = {};
  for (const s of (signals || [])) {
    if (!grouped[s.category]) grouped[s.category] = [];
    grouped[s.category].push(s);
  }

  return (
    <div className="vt-signal-grid">
      {CATEGORY_ORDER.filter(cat => grouped[cat]).map(cat => (
        <div key={cat} className="vt-signal-group">
          <div className="vt-signal-group-label">{CATEGORY_LABELS[cat] || cat}</div>
          {grouped[cat].map((s, i) => (
            <div key={i} className={`vt-signal-row score-${s.score}`}>
              <span className="vt-sig-emoji">{s.emoji}</span>
              <div className="vt-sig-content">
                <span className="vt-sig-name">{s.name}</span>
                <span className="vt-sig-reason">{s.reason}</span>
              </div>
              <span className="vt-sig-weight">×{s.weight}</span>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

function TradeSetupCard({ setup, coin }: { setup: TradeSetup; coin: string }) {
  const isLong = setup.direction === 'LONG';
  const dirColor = isLong ? '#10b981' : '#ef4444';

  return (
    <div className="vt-trade-card">
      <div className="vt-trade-header">
        <span className="vt-trade-direction" style={{ color: dirColor }}>
          {isLong ? '⬆ LONG' : '⬇ SHORT'}
        </span>
        <span className="vt-trade-rr">R:R {setup.risk_reward}:1</span>
      </div>

      <div className="vt-trade-grid">
        <div className="vt-trade-item">
          <span className="vt-ti-label">Entry</span>
          <span className="vt-ti-value">${formatPrice(setup.entry_price)}</span>
        </div>
        <div className="vt-trade-item sl">
          <span className="vt-ti-label">Stop Loss</span>
          <span className="vt-ti-value">
            ${formatPrice(setup.stop_loss)}{' '}
            <span className="vt-ti-pct">(-{setup.sl_pct}%)</span>
          </span>
        </div>
        <div className="vt-trade-item tp">
          <span className="vt-ti-label">Take Profit</span>
          <span className="vt-ti-value">
            ${formatPrice(setup.take_profit)}{' '}
            <span className="vt-ti-pct">(+{setup.tp_pct}%)</span>
          </span>
        </div>
        <div className="vt-trade-item">
          <span className="vt-ti-label">Size</span>
          <span className="vt-ti-value">
            ${setup.position_usd.toLocaleString()} ({setup.position_pct}%)
          </span>
        </div>
        <div className="vt-trade-item">
          <span className="vt-ti-label">Max Risk</span>
          <span className="vt-ti-value loss">${setup.max_loss_usd.toLocaleString()}</span>
        </div>
        <div className="vt-trade-item">
          <span className="vt-ti-label">Max Hold</span>
          <span className="vt-ti-value">{setup.max_hold_h}h</span>
        </div>
      </div>
    </div>
  );
}

function AIReasoning({ data }: { data: VerdictData }) {
  return (
    <div className="vt-ai-card">
      <div className="vt-ai-header">🧠 AI Analysis</div>

      {data.summary && (
        <p className="vt-ai-summary">{data.summary}</p>
      )}

      {(data.reasons?.length ?? 0) > 0 && (
        <div className="vt-ai-section">
          <div className="vt-ai-section-title">Why this verdict</div>
          <ul className="vt-ai-list bull">
            {(data.reasons || []).map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}

      {(data.risks?.length ?? 0) > 0 && (
        <div className="vt-ai-section">
          <div className="vt-ai-section-title">Key risks</div>
          <ul className="vt-ai-list bear">
            {(data.risks || []).map((r, i) => <li key={i}>{r}</li>)}
          </ul>
        </div>
      )}

      {(data.watch_for?.length ?? 0) > 0 && (
        <div className="vt-ai-section">
          <div className="vt-ai-section-title">Watch for (would change verdict)</div>
          <ul className="vt-ai-list neut">
            {(data.watch_for || []).map((w, i) => <li key={i}>{w}</li>)}
          </ul>
        </div>
      )}
    </div>
  );
}

function AccuracyCard({ stats }: { stats: AccuracyStats }) {
  if (stats.total === 0) return null;
  return (
    <div className="vt-accuracy-card">
      <div className="vt-acc-header">📊 Past 30-Day Accuracy</div>
      <div className="vt-acc-main">
        <span className="vt-acc-pct" style={{ color: stats.accuracy_pct >= 60 ? '#10b981' : '#f59e0b' }}>
          {stats.accuracy_pct}%
        </span>
        <span className="vt-acc-sub">{stats.correct}/{stats.total} verdicts correct</span>
      </div>
    </div>
  );
}

// ─── Main Component ───────────────────────────────────────────────────────────

const VerdictTab: React.FC<Props> = ({ coin, capital, tradeType }) => {
  const [loading, setLoading]     = useState(false);
  const [data, setData]           = useState<VerdictData | null>(null);
  const [accuracy, setAccuracy]   = useState<AccuracyStats | null>(null);
  const [error, setError]         = useState<string | null>(null);
  const [lastFetch, setLastFetch] = useState<Date | null>(null);

  const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;

  const fetchVerdict = useCallback(async () => {
    if (!coin) return;
    setLoading(true);
    setError(null);

    const cap = capital && capital > 0 ? capital : 1000;
    const tt  = tradeType || 'SWING';

    try {
      const [verdictRes, accRes] = await Promise.allSettled([
        axios.get(`${API_BASE}/verdict/${coin}`, { params: { trade_type: tt, capital: cap } }),
        axios.get(`${API_BASE}/verdict-history/${coin}`, { params: { days: 30 } }),
      ]);

      if (verdictRes.status === 'fulfilled') {
        setData(verdictRes.value.data);
        setLastFetch(new Date());
      } else {
        setError('Failed to fetch verdict. Check API connection.');
      }

      if (accRes.status === 'fulfilled') {
        setAccuracy(accRes.value.data);
      }
    } catch (e: any) {
      setError(e?.response?.data?.detail || e.message || 'Unknown error');
    } finally {
      setLoading(false);
    }
  }, [coin, capital, tradeType, API_BASE]);

  // Auto-fetch on mount
  useEffect(() => {
    fetchVerdict();
  }, [fetchVerdict]);

  const timeSince = lastFetch
    ? Math.floor((Date.now() - lastFetch.getTime()) / 1000)
    : null;

  return (
    <div className="verdict-tab">
      {/* ── Header bar ── */}
      <div className="vt-header">
        <div className="vt-header-left">
          <span className="vt-title">Precision Verdict</span>
          {data?.cached && <span className="vt-cache-badge">cached</span>}
          {timeSince !== null && (
            <span className="vt-time-badge">
              {timeSince < 60 ? `${timeSince}s ago` : `${Math.floor(timeSince / 60)}m ago`}
            </span>
          )}
        </div>
        <button
          className="vt-refresh-btn"
          onClick={fetchVerdict}
          disabled={loading}
        >
          {loading ? '⏳ Fetching...' : '🔄 Refresh'}
        </button>
      </div>

      {/* ── Error ── */}
      {error && (
        <div className="vt-error">
          <span>⚠️ {error}</span>
          <button onClick={fetchVerdict}>Retry</button>
        </div>
      )}

      {/* ── Loading skeleton ── */}
      {loading && !data && (
        <div className="vt-loading">
          <div className="vt-spinner" />
          <p>Scoring 13 signals...</p>
        </div>
      )}

      {/* ── Main content ── */}
      {data && (
        <div className="vt-content">
          {/* Top row: verdict badge + trade setup */}
          <div className="vt-top-row">
            <VerdictBadge data={data} />
            {data.trade_setup ? (
              <TradeSetupCard setup={data.trade_setup} coin={data.coin} />
            ) : (
              <div className="vt-no-trade">
                <span className="vt-no-trade-icon">
                  {data.verdict === 'HOLD' ? '→' : data.verdict === 'AVOID' ? '⚠️' : '—'}
                </span>
                <span className="vt-no-trade-label">
                  {data.verdict === 'HOLD' ? 'No trade setup — wait for clearer signal'
                   : data.verdict === 'AVOID' ? 'Signal too weak — stay out'
                   : 'No actionable trade'}
                </span>
              </div>
            )}
          </div>

          {/* Signal breakdown */}
          <div className="vt-section">
            <div className="vt-section-title">Signal Breakdown</div>
            <SignalGrid signals={data.signals || []} />
          </div>

          {/* AI reasoning */}
          <AIReasoning data={data} />

          {/* Accuracy */}
          {accuracy && <AccuracyCard stats={accuracy} />}
        </div>
      )}
    </div>
  );
};

export default VerdictTab;
