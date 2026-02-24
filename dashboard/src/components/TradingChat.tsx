import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './TradingChat.css';

const API_BASE = process.env.REACT_APP_API_URL || window.location.origin;

// ─── Types ────────────────────────────────────────────────────────────────────

interface Target {
  label: string;
  price: string;
  action: string;
  probability?: string;
}

interface StopLoss {
  price: string;
  reasoning: string;
}

interface ScenarioProbabilities {
  bullish: number;
  neutral: number;
  bearish: number;
}

interface TimeframeAlignment {
  '1H'?: string;
  '4H'?: string;
  '1D'?: string;
  '1W'?: string;
  alignment?: string;
  confidence_delta?: string;
}

interface WeeklyMomentum {
  structure?: string;
  status?: string;
  red_flags?: string[];
}

interface KeyLevels {
  major_support?: string;
  major_resistance?: string;
  invalidation?: string;
}

interface BitcoinInfluence {
  btc_price?: number;
  environment?: string;
  impact_on_trade?: string;
}

interface ExecutionPlan {
  current_status?: string;
  action?: string;
  reasoning?: string;
}

interface PositionSizing {
  deploy_now_usd?: number;
  add_condition?: string;
}

interface AIResponse {
  status?: 'INPUT_REQUIRED' | 'ANALYSIS_COMPLETE';
  // INPUT_REQUIRED fields
  message?: string;
  required_fields?: string[];
  optional_fields?: string[];
  // Core display fields (normalized by backend)
  reasoning?: string;
  verdict?: 'BUY' | 'SELL' | 'EXIT' | 'HOLD' | 'WAIT' | 'ADD' | null;
  decision?: string;
  stop_loss?: StopLoss;
  targets?: Target[];
  scenario_probabilities?: ScenarioProbabilities;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  confidence_score?: number;
  asset_tier?: string;
  add_capital_condition?: string;
  warning?: string;
  providers_used: string[];
  consensus: boolean;
  // Rich ANALYSIS_COMPLETE fields
  market_bias?: string;
  timeframe_alignment?: TimeframeAlignment;
  weekly_momentum?: WeeklyMomentum;
  volume_analysis?: string;
  key_levels?: KeyLevels;
  bitcoin_influence?: BitcoinInfluence;
  news_sentiment?: string;
  execution_plan?: ExecutionPlan;
  position_sizing?: PositionSizing;
  time_review?: string;
}

interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
  aiData?: AIResponse;
  timestamp: Date;
}

interface TradingChatProps {
  coin: string;
  analysis: Record<string, unknown> | null;
  currentPrice?: number;
}

// ─── Config ───────────────────────────────────────────────────────────────────

const VERDICT_CONFIG: Record<string, { color: string; bg: string; label: string }> = {
  BUY:  { color: '#22c55e', bg: '#22c55e18', label: '▲ BUY' },
  ADD:  { color: '#22c55e', bg: '#22c55e18', label: '▲ ADD' },
  HOLD: { color: '#f59e0b', bg: '#f59e0b18', label: '◆ HOLD' },
  WAIT: { color: '#94a3b8', bg: '#94a3b818', label: '◇ WAIT' },
  SELL: { color: '#ef4444', bg: '#ef444418', label: '▼ SELL' },
  EXIT: { color: '#ef4444', bg: '#ef444418', label: '▼ EXIT' },
};

const CONFIDENCE_CONFIG: Record<string, { color: string; label: string }> = {
  HIGH:   { color: '#22c55e', label: '🔥 High' },
  MEDIUM: { color: '#f59e0b', label: '⚡ Medium' },
  LOW:    { color: '#94a3b8', label: '⚠️ Low' },
};

const HORIZON_OPTIONS = [
  { id: 'scalp',     label: 'Scalp',     sub: '<24h' },
  { id: 'short',     label: 'Short',     sub: '1–7d' },
  { id: 'swing',     label: 'Swing',     sub: '1–4w' },
  { id: 'investing', label: 'Investing', sub: '3m+' },
];

const SUGGESTED_QUESTIONS = [
  'Should I hold or exit my position?',
  "What's a good stop loss level?",
  'Is this a good DCA entry point?',
  'What are the key levels to watch?',
  'Should I take profit now?',
];

// ─── Sub-components ───────────────────────────────────────────────────────────

const ScenarioBar: React.FC<{ probs: ScenarioProbabilities }> = ({ probs }) => (
  <div className="scenario-bars">
    {(['bullish', 'neutral', 'bearish'] as const).map(k => (
      <div key={k} className="scenario-row">
        <span className={`scenario-label ${k}`}>{k.charAt(0).toUpperCase() + k.slice(1)}</span>
        <div className="scenario-track">
          <div className={`scenario-fill ${k}`} style={{ width: `${probs[k]}%` }} />
        </div>
        <span className="scenario-pct">{probs[k]}%</span>
      </div>
    ))}
  </div>
);

const TfSignal: React.FC<{ label: string; value: string }> = ({ label, value }) => {
  const lower = value.toLowerCase();
  const cls = lower.includes('bull') ? 'bull' : lower.includes('bear') ? 'bear' : 'neutral';
  const short = value.length > 20 ? value.split(/[\s—–-]/)[0] : value;
  return (
    <div className="tf-cell">
      <span className="tf-label">{label}</span>
      <span className={`tf-signal ${cls}`}>{short}</span>
    </div>
  );
};

const AIMessageCard: React.FC<{ data: AIResponse }> = ({ data }) => {
  const verdictKey = (data.verdict || data.decision || '').toUpperCase() as string;
  const vc = VERDICT_CONFIG[verdictKey] || null;
  const cc = CONFIDENCE_CONFIG[data.confidence] || CONFIDENCE_CONFIG.LOW;
  const tfa = data.timeframe_alignment;

  return (
    <div className="ai-card">
      {/* Verdict + Confidence */}
      <div className="ai-card-header">
        {vc && (
          <span className="verdict-chip" style={{ color: vc.color, background: vc.bg, borderColor: `${vc.color}40` }}>
            {vc.label}
          </span>
        )}
        <div className="confidence-info">
          <span className="confidence-text" style={{ color: cc.color }}>{cc.label}</span>
          {data.confidence_score !== undefined && (
            <>
              <div className="confidence-bar-wrap">
                <div className="confidence-bar-fill" style={{ width: `${data.confidence_score}%`, background: cc.color }} />
              </div>
              <span className="confidence-score" style={{ color: cc.color }}>{data.confidence_score}%</span>
            </>
          )}
        </div>
      </div>

      {/* Warning */}
      {data.warning && <div className="ai-warning">⚠️ {data.warning}</div>}

      {/* Market bias + BTC environment */}
      {(data.market_bias || data.bitcoin_influence?.environment) && (
        <div className="ai-market-row">
          {data.market_bias && <span className="market-bias-text">{data.market_bias}</span>}
          {data.bitcoin_influence?.environment && (
            <span className="btc-env-tag">₿ {data.bitcoin_influence.environment}</span>
          )}
        </div>
      )}

      {/* Timeframe alignment */}
      {tfa && (
        <div className="tf-alignment">
          {(['1H', '4H', '1D', '1W'] as const).map(tf =>
            tfa[tf] ? <TfSignal key={tf} label={tf} value={tfa[tf]!} /> : null
          )}
          {tfa.alignment && <TfSignal label="Overall" value={tfa.alignment} />}
          {tfa.confidence_delta && (
            <span className="tf-delta">{tfa.confidence_delta}</span>
          )}
        </div>
      )}

      {/* Reasoning */}
      {data.reasoning && <p className="ai-reasoning">{data.reasoning}</p>}

      {/* Stop Loss */}
      {data.stop_loss && (
        <div className="ai-detail-row stop-loss">
          <span className="detail-icon">🛑</span>
          <div>
            <span className="detail-label">Stop Loss</span>
            <span className="detail-value">{data.stop_loss.price}</span>
            {data.stop_loss.reasoning && <span className="detail-sub">{data.stop_loss.reasoning}</span>}
          </div>
        </div>
      )}

      {/* Targets */}
      {data.targets && data.targets.length > 0 && (
        <div className="ai-targets">
          {data.targets.map((t, i) => (
            <div key={i} className="target-row">
              <span className="target-label">{t.label}</span>
              <span className="target-price">{t.price}</span>
              <span className="target-action">{t.action}</span>
              {t.probability && <span className="target-prob">{t.probability}</span>}
            </div>
          ))}
        </div>
      )}

      {/* Scenario probabilities */}
      {data.scenario_probabilities && <ScenarioBar probs={data.scenario_probabilities} />}

      {/* Key levels */}
      {data.key_levels && (data.key_levels.major_support || data.key_levels.major_resistance || data.key_levels.invalidation) && (
        <div className="ai-key-levels">
          {data.key_levels.major_support && (
            <div className="kl-row">
              <span className="kl-label">Support</span>
              <span className="kl-value kl-support">{data.key_levels.major_support}</span>
            </div>
          )}
          {data.key_levels.major_resistance && (
            <div className="kl-row">
              <span className="kl-label">Resistance</span>
              <span className="kl-value kl-resistance">{data.key_levels.major_resistance}</span>
            </div>
          )}
          {data.key_levels.invalidation && (
            <div className="kl-row">
              <span className="kl-label">Invalidation</span>
              <span className="kl-value kl-invalid">{data.key_levels.invalidation}</span>
            </div>
          )}
        </div>
      )}

      {/* Add capital / position sizing */}
      {((data.add_capital_condition && data.add_capital_condition.toLowerCase() !== 'none') ||
        (data.position_sizing?.deploy_now_usd !== undefined && data.position_sizing.deploy_now_usd > 0)) && (
        <div className="ai-detail-row add-condition">
          <span className="detail-icon">💰</span>
          <div>
            {data.position_sizing?.deploy_now_usd !== undefined && data.position_sizing.deploy_now_usd > 0 && (
              <>
                <span className="detail-label">Deploy Now</span>
                <span className="detail-value">${data.position_sizing.deploy_now_usd.toLocaleString()}</span>
              </>
            )}
            {(data.add_capital_condition || data.position_sizing?.add_condition) && (
              <span className="detail-sub">{data.add_capital_condition || data.position_sizing?.add_condition}</span>
            )}
          </div>
        </div>
      )}

      {/* News sentiment */}
      {data.news_sentiment && data.news_sentiment.toLowerCase() !== 'none' && (
        <div className="ai-news-sentiment">
          📰 {data.news_sentiment}
        </div>
      )}

      {/* Time review */}
      {data.time_review && data.time_review.toLowerCase() !== 'none' && (
        <div className="ai-time-review">🔄 {data.time_review}</div>
      )}

      {/* Footer */}
      <div className="ai-card-footer">
        {data.asset_tier && <span className="asset-tier-tag">{data.asset_tier}</span>}
        <div className="providers-row">
          {data.providers_used.map(p => <span key={p} className="provider-tag">{p}</span>)}
          {data.consensus && <span className="consensus-tag">✓ Consensus</span>}
        </div>
      </div>
    </div>
  );
};

const InputRequiredCard: React.FC<{
  data: AIResponse;
  onSelectHorizon: (h: string) => void;
}> = ({ data, onSelectHorizon }) => (
  <div className="ai-card input-required-card">
    <div className="ai-card-header">
      <span className="verdict-chip" style={{ color: '#f59e0b', background: '#f59e0b18', borderColor: '#f59e0b40' }}>
        ⏳ Input Needed
      </span>
    </div>
    <p className="ai-reasoning">{data.message || 'Provide your time horizon to proceed.'}</p>
    {data.required_fields?.some(f => f.toLowerCase().includes('time_horizon')) && (
      <div className="horizon-inline-selector">
        <span className="detail-label" style={{ display: 'block', marginBottom: 8 }}>Select your time horizon:</span>
        <div className="horizon-inline-chips">
          {HORIZON_OPTIONS.map(h => (
            <button key={h.id} className="horizon-chip-inline" onClick={() => onSelectHorizon(h.id)}>
              <span className="hci-label">{h.label}</span>
              <span className="hci-sub">{h.sub}</span>
            </button>
          ))}
        </div>
      </div>
    )}
    {data.optional_fields && data.optional_fields.length > 0 && (
      <p className="ai-optional-hint">
        Optional: {data.optional_fields.join(' · ')}
      </p>
    )}
  </div>
);

// ─── Main Component ───────────────────────────────────────────────────────────

const TradingChat: React.FC<TradingChatProps> = ({ coin, analysis, currentPrice }) => {
  const [messages, setMessages]           = useState<ChatMessage[]>([]);
  const [input, setInput]                 = useState('');
  const [entryPrice, setEntryPrice]       = useState('');
  const [capital, setCapital]             = useState('');
  const [timeHorizon, setTimeHorizon]     = useState('');
  const [loading, setLoading]             = useState(false);
  const [showEntryInput, setShowEntryInput] = useState(false);
  const messagesEndRef  = useRef<HTMLDivElement>(null);
  const lastUserMsgRef  = useRef<string>('');

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, loading]);

  const sendMessage = async (text?: string, horizonOverride?: string) => {
    const messageText = (text || input).trim();
    if (!messageText || loading) return;

    lastUserMsgRef.current = messageText;
    const userMsg: ChatMessage = { role: 'user', content: messageText, timestamp: new Date() };
    setMessages(prev => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const historyToSend = messages.slice(-8).map(m => ({
        role: m.role,
        content: m.aiData ? (m.aiData.reasoning || m.aiData.message || '') : m.content,
      }));

      const enrichedContext = analysis
        ? { ...analysis, price: currentPrice || (analysis as Record<string, unknown>).price }
        : null;

      const effectiveHorizon = horizonOverride ?? timeHorizon;

      const resp = await axios.post<AIResponse>(`${API_BASE}/chat/${coin}`, {
        message: messageText,
        chat_history: historyToSend,
        analysis_context: enrichedContext,
        entry_price: entryPrice ? parseFloat(entryPrice) : null,
        time_horizon: effectiveHorizon || null,
        capital: capital ? parseFloat(capital) : null,
      });

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: resp.data.reasoning || resp.data.message || '',
        aiData: resp.data,
        timestamp: new Date(),
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Failed to get a response. Check that the API is running.',
        timestamp: new Date(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  // Called when user taps a horizon chip inside an INPUT_REQUIRED card
  const handleHorizonSelect = async (h: string) => {
    const lastMsg = lastUserMsgRef.current;
    if (!lastMsg || loading) return;

    setTimeHorizon(h);

    // Remove the INPUT_REQUIRED card and re-query silently
    const filteredMsgs = messages[messages.length - 1]?.aiData?.status === 'INPUT_REQUIRED'
      ? messages.slice(0, -1)
      : messages;
    setMessages(filteredMsgs);
    setLoading(true);

    try {
      const historyToSend = filteredMsgs.slice(-8).map(m => ({
        role: m.role,
        content: m.aiData ? (m.aiData.reasoning || m.aiData.message || '') : m.content,
      }));

      const enrichedContext = analysis
        ? { ...analysis, price: currentPrice || (analysis as Record<string, unknown>).price }
        : null;

      const resp = await axios.post<AIResponse>(`${API_BASE}/chat/${coin}`, {
        message: lastMsg,
        chat_history: historyToSend,
        analysis_context: enrichedContext,
        entry_price: entryPrice ? parseFloat(entryPrice) : null,
        time_horizon: h,
        capital: capital ? parseFloat(capital) : null,
      });

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: resp.data.reasoning || resp.data.message || '',
        aiData: resp.data,
        timestamp: new Date(),
      }]);
    } catch {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: 'Failed to get a response. Check that the API is running.',
        timestamp: new Date(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const coinSymbol = coin.replace('_USDT', '').replace('_', '/');

  return (
    <section className="trading-chat-section">
      {/* Header */}
      <div className="chat-header-bar">
        <div className="chat-title">
          <span className="chat-icon">🤖</span>
          <div>
            <h3>AI Trading Advisor</h3>
            <span className="chat-subtitle">Ask anything about {coinSymbol} — straight verdict, no fluff</span>
          </div>
        </div>
        <button
          className={`entry-toggle-btn ${showEntryInput ? 'active' : ''}`}
          onClick={() => setShowEntryInput(v => !v)}
        >
          {showEntryInput ? 'Hide' : '📍 Entry Price'}
        </button>
      </div>

      {/* Time horizon + Capital row */}
      <div className="horizon-row">
        <span className="horizon-row-label">Horizon:</span>
        {HORIZON_OPTIONS.map(h => (
          <button
            key={h.id}
            className={`horizon-chip ${timeHorizon === h.id ? 'active' : ''}`}
            onClick={() => setTimeHorizon(prev => prev === h.id ? '' : h.id)}
            title={h.sub}
          >
            {h.label}
          </button>
        ))}
        <input
          type="number"
          min="0"
          step="any"
          placeholder="Capital $"
          value={capital}
          onChange={e => setCapital(e.target.value)}
          className="capital-input"
        />
      </div>

      {/* Entry price */}
      {showEntryInput && (
        <div className="entry-price-row">
          <label>Avg buy price for {coinSymbol}:</label>
          <input
            type="number"
            step="any"
            placeholder={`e.g. ${currentPrice ?? '0.00000400'}`}
            value={entryPrice}
            onChange={e => setEntryPrice(e.target.value)}
            className="entry-price-input"
          />
          {entryPrice && currentPrice && !isNaN(parseFloat(entryPrice)) && (
            <span className={`live-pnl ${((currentPrice - parseFloat(entryPrice)) / parseFloat(entryPrice)) * 100 >= 0 ? 'positive' : 'negative'}`}>
              {(((currentPrice - parseFloat(entryPrice)) / parseFloat(entryPrice)) * 100).toFixed(2)}%
            </span>
          )}
        </div>
      )}

      {/* Suggested questions */}
      {messages.length === 0 && (
        <div className="suggested-questions">
          {SUGGESTED_QUESTIONS.map((q, i) => (
            <button key={i} className="suggested-btn" onClick={() => sendMessage(q)}>{q}</button>
          ))}
        </div>
      )}

      {/* Message thread */}
      {messages.length > 0 && (
        <div className="chat-messages">
          {messages.map((msg, i) => (
            <div key={i} className={`chat-bubble-wrapper ${msg.role}`}>
              {msg.role === 'assistant' && msg.aiData?.status === 'INPUT_REQUIRED' ? (
                <InputRequiredCard data={msg.aiData} onSelectHorizon={handleHorizonSelect} />
              ) : msg.role === 'assistant' && msg.aiData ? (
                <AIMessageCard data={msg.aiData} />
              ) : msg.role === 'assistant' ? (
                <div className="chat-bubble assistant"><p>{msg.content}</p></div>
              ) : (
                <div className="chat-bubble user"><p>{msg.content}</p></div>
              )}
            </div>
          ))}

          {loading && (
            <div className="chat-bubble-wrapper assistant">
              <div className="chat-bubble assistant typing">
                <span className="dot" /><span className="dot" /><span className="dot" />
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>
      )}

      {/* Input */}
      <div className="chat-input-row">
        <textarea
          className="chat-input"
          placeholder={`Ask about ${coinSymbol}… e.g. "I'm down 10%, should I hold or exit?"`}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          rows={1}
          disabled={loading}
        />
        <button
          className="chat-send-btn"
          onClick={() => sendMessage()}
          disabled={!input.trim() || loading}
        >
          {loading ? '…' : '↑'}
        </button>
      </div>
    </section>
  );
};

export default TradingChat;
