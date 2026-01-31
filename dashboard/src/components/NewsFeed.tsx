import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import './NewsFeed.css';

interface NewsItem {
  title: string;
  source: string;
  url: string;
  published: string;
  description: string;
  sentiment: 'BULLISH' | 'BEARISH' | 'NEUTRAL';
  sentiment_score: number;
  category: 'CRYPTO' | 'GEOPOLITICAL';
  impact_type?: string;
  has_market_impact: boolean;
}

interface MarketSentiment {
  overall_sentiment: string;
  sentiment_score: number;
  bullish_count: number;
  bearish_count: number;
  neutral_count: number;
  high_impact_count: number;
  total_news: number;
}

interface NewsResponse {
  crypto_news: NewsItem[];
  geopolitical_news: NewsItem[];
  market_sentiment: MarketSentiment;
  timestamp: string;
}

type NewsCategory = 'all' | 'crypto' | 'geopolitical';

const NewsFeed: React.FC = () => {
  const [news, setNews] = useState<NewsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<NewsCategory>('all');
  const [lastUpdated, setLastUpdated] = useState<Date | null>(null);

  const fetchNews = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get<NewsResponse>('http://localhost:8000/news/all');
      setNews(response.data);
      setLastUpdated(new Date());
      setError(null);
    } catch (err) {
      console.error('Error fetching news:', err);
      setError('Failed to load news feed');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchNews();
    // Refresh news every 5 minutes
    const interval = setInterval(fetchNews, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchNews]);

  const getSentimentIcon = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH': return '🟢';
      case 'BEARISH': return '🔴';
      default: return '⚪';
    }
  };

  const getSentimentClass = (sentiment: string) => {
    switch (sentiment) {
      case 'BULLISH': return 'sentiment-bullish';
      case 'BEARISH': return 'sentiment-bearish';
      default: return 'sentiment-neutral';
    }
  };

  const getImpactBadge = (item: NewsItem) => {
    if (item.category === 'GEOPOLITICAL' && item.impact_type) {
      const badges: Record<string, { icon: string; label: string }> = {
        'HIGH_RISK': { icon: '⚠️', label: 'High Risk' },
        'MONETARY': { icon: '💰', label: 'Fed/Rates' },
        'REGULATORY': { icon: '⚖️', label: 'Regulation' },
      };
      return badges[item.impact_type] || null;
    }
    return null;
  };

  const formatTimeAgo = (dateString: string) => {
    if (!dateString) return '';
    try {
      const date = new Date(dateString);
      const now = new Date();
      const diffMs = now.getTime() - date.getTime();
      const diffMins = Math.floor(diffMs / 60000);
      const diffHours = Math.floor(diffMins / 60);
      const diffDays = Math.floor(diffHours / 24);

      if (diffMins < 60) return `${diffMins}m ago`;
      if (diffHours < 24) return `${diffHours}h ago`;
      return `${diffDays}d ago`;
    } catch {
      return '';
    }
  };

  const getFilteredNews = () => {
    if (!news) return [];

    switch (activeCategory) {
      case 'crypto':
        return news.crypto_news;
      case 'geopolitical':
        return news.geopolitical_news;
      default:
        // Interleave crypto and geo news
        const combined: NewsItem[] = [];
        const maxLen = Math.max(news.crypto_news.length, news.geopolitical_news.length);
        for (let i = 0; i < maxLen; i++) {
          if (i < news.crypto_news.length) combined.push(news.crypto_news[i]);
          if (i < news.geopolitical_news.length) combined.push(news.geopolitical_news[i]);
        }
        return combined.slice(0, 20);
    }
  };

  const renderSentimentSummary = () => {
    if (!news?.market_sentiment) return null;
    const { market_sentiment: ms } = news;

    return (
      <div className="sentiment-summary">
        <div className={`overall-sentiment ${getSentimentClass(ms.overall_sentiment)}`}>
          <span className="sentiment-icon">{getSentimentIcon(ms.overall_sentiment)}</span>
          <span className="sentiment-label">Market Mood: {ms.overall_sentiment}</span>
          <span className="sentiment-score">
            ({ms.sentiment_score > 0 ? '+' : ''}{ms.sentiment_score})
          </span>
        </div>
        <div className="sentiment-breakdown">
          <span className="bullish">🟢 {ms.bullish_count} Bullish</span>
          <span className="bearish">🔴 {ms.bearish_count} Bearish</span>
          <span className="neutral">⚪ {ms.neutral_count} Neutral</span>
          {ms.high_impact_count > 0 && (
            <span className="high-impact">⚡ {ms.high_impact_count} High Impact</span>
          )}
        </div>
      </div>
    );
  };

  if (loading && !news) {
    return (
      <div className="news-feed loading">
        <div className="news-header">
          <h2>📰 Market News</h2>
        </div>
        <div className="news-loading-content">
          <div className="news-spinner-ring"></div>
          <p className="news-loading-text">Loading news...</p>
        </div>
      </div>
    );
  }

  if (error && !news) {
    return (
      <div className="news-feed error">
        <div className="news-header">
          <h2>📰 Market News</h2>
        </div>
        <div className="error-message">{error}</div>
        <button onClick={fetchNews} className="retry-btn">Retry</button>
      </div>
    );
  }

  const filteredNews = getFilteredNews();

  return (
    <div className="news-feed">
      <div className="news-header">
        <h2>📰 Market News</h2>
        <div className="news-controls">
          <button
            onClick={fetchNews}
            className="refresh-btn"
            disabled={loading}
          >
            {loading ? '⏳' : '🔄'}
          </button>
          {lastUpdated && (
            <span className="last-updated">
              Updated {formatTimeAgo(lastUpdated.toISOString())}
            </span>
          )}
        </div>
      </div>

      {renderSentimentSummary()}

      <div className="category-tabs">
        <button
          className={`tab ${activeCategory === 'all' ? 'active' : ''}`}
          onClick={() => setActiveCategory('all')}
        >
          All News
        </button>
        <button
          className={`tab ${activeCategory === 'crypto' ? 'active' : ''}`}
          onClick={() => setActiveCategory('crypto')}
        >
          🪙 Crypto
        </button>
        <button
          className={`tab ${activeCategory === 'geopolitical' ? 'active' : ''}`}
          onClick={() => setActiveCategory('geopolitical')}
        >
          🌍 Geopolitical
        </button>
      </div>

      <div className="news-list">
        {filteredNews.length === 0 ? (
          <div className="no-news">No news available</div>
        ) : (
          filteredNews.map((item, index) => (
            <a
              key={`${item.source}-${index}`}
              href={item.url}
              target="_blank"
              rel="noopener noreferrer"
              className={`news-item ${item.has_market_impact ? 'high-impact' : ''}`}
            >
              <div className="news-item-header">
                <span className={`sentiment-badge ${getSentimentClass(item.sentiment)}`}>
                  {getSentimentIcon(item.sentiment)}
                </span>
                <span className="news-source">{item.source}</span>
                <span className="news-category">
                  {item.category === 'CRYPTO' ? '🪙' : '🌍'}
                </span>
                {getImpactBadge(item) && (
                  <span className="impact-badge">
                    {getImpactBadge(item)?.icon} {getImpactBadge(item)?.label}
                  </span>
                )}
                <span className="news-time">{formatTimeAgo(item.published)}</span>
              </div>
              <h3 className="news-title">{item.title}</h3>
              {item.description && (
                <p className="news-description">{item.description}</p>
              )}
            </a>
          ))
        )}
      </div>
    </div>
  );
};

export default NewsFeed;
