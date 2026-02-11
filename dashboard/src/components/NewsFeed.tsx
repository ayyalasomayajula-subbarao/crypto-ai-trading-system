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

type NewsCategory = 'all' | 'crypto' | 'geopolitical' | 'bullish' | 'bearish' | 'high-impact';

const CATEGORIES: { key: NewsCategory; label: string }[] = [
  { key: 'all', label: 'All Posts' },
  { key: 'crypto', label: 'Crypto' },
  { key: 'geopolitical', label: 'Geopolitical' },
  { key: 'bullish', label: 'Bullish' },
  { key: 'bearish', label: 'Bearish' },
  { key: 'high-impact', label: 'High Impact' },
];

const SOURCE_ICONS: Record<string, string> = {
  'CoinTelegraph': 'CT',
  'CoinDesk': 'CD',
  'Decrypt': 'DC',
  'Bitcoin Magazine': 'BM',
  'BBC Business': 'BBC',
  'NY Times Business': 'NYT',
  'Reuters': 'R',
  'CNBC': 'CNBC',
};

const getCardGradient = (item: NewsItem): string => {
  if (item.sentiment === 'BULLISH') {
    return 'linear-gradient(135deg, #0a3d2e 0%, #00d39540 60%, #0a3d2e 100%)';
  }
  if (item.sentiment === 'BEARISH') {
    return 'linear-gradient(135deg, #3d0a0a 0%, #ff6b6b40 60%, #3d0a0a 100%)';
  }
  return 'linear-gradient(135deg, #1a1a3e 0%, #667eea30 60%, #1a1a3e 100%)';
};

const formatDate = (dateString: string): string => {
  if (!dateString) return '';
  try {
    const date = new Date(dateString);
    return date.toLocaleDateString('en-US', {
      month: 'long',
      day: '2-digit',
      year: 'numeric',
    }).toUpperCase();
  } catch {
    return '';
  }
};

const NewsFeed: React.FC = () => {
  const [news, setNews] = useState<NewsResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeCategory, setActiveCategory] = useState<NewsCategory>('all');

  const fetchNews = useCallback(async () => {
    try {
      setLoading(true);
      const response = await axios.get<NewsResponse>('http://localhost:8000/news/all');
      setNews(response.data);
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
    const interval = setInterval(fetchNews, 5 * 60 * 1000);
    return () => clearInterval(interval);
  }, [fetchNews]);

  const getAllNews = useCallback((): NewsItem[] => {
    if (!news) return [];
    const combined: NewsItem[] = [];
    const maxLen = Math.max(news.crypto_news.length, news.geopolitical_news.length);
    for (let i = 0; i < maxLen; i++) {
      if (i < news.crypto_news.length) combined.push(news.crypto_news[i]);
      if (i < news.geopolitical_news.length) combined.push(news.geopolitical_news[i]);
    }
    return combined;
  }, [news]);

  const getFilteredNews = useCallback((): NewsItem[] => {
    const all = getAllNews();
    switch (activeCategory) {
      case 'crypto':
        return all.filter(n => n.category === 'CRYPTO');
      case 'geopolitical':
        return all.filter(n => n.category === 'GEOPOLITICAL');
      case 'bullish':
        return all.filter(n => n.sentiment === 'BULLISH');
      case 'bearish':
        return all.filter(n => n.sentiment === 'BEARISH');
      case 'high-impact':
        return all.filter(n => n.has_market_impact);
      default:
        return all;
    }
  }, [activeCategory, getAllNews]);

  const getBadges = (item: NewsItem): { label: string; className: string }[] => {
    const badges: { label: string; className: string }[] = [];

    if (item.category === 'CRYPTO') {
      badges.push({ label: 'Crypto', className: 'badge-crypto' });
    } else {
      badges.push({ label: 'Geopolitical', className: 'badge-geo' });
    }

    if (item.sentiment === 'BULLISH') {
      badges.push({ label: 'Bullish', className: 'badge-bullish' });
    } else if (item.sentiment === 'BEARISH') {
      badges.push({ label: 'Bearish', className: 'badge-bearish' });
    }

    if (item.has_market_impact && item.impact_type) {
      const impactLabels: Record<string, string> = {
        'HIGH_RISK': 'High Risk',
        'MONETARY': 'Fed/Rates',
        'REGULATORY': 'Regulation',
      };
      if (impactLabels[item.impact_type]) {
        badges.push({ label: impactLabels[item.impact_type], className: 'badge-impact' });
      }
    }

    return badges;
  };

  const filteredNews = getFilteredNews();
  const totalCount = getAllNews().length;

  if (loading && !news) {
    return (
      <div className="news-section">
        <div className="news-section-header">
          <h2>Market News</h2>
        </div>
        <div className="news-loading">
          <div className="news-spinner-ring"></div>
          <p>Loading news...</p>
        </div>
      </div>
    );
  }

  if (error && !news) {
    return (
      <div className="news-section">
        <div className="news-section-header">
          <h2>Market News</h2>
        </div>
        <div className="news-error">
          <p>{error}</p>
          <button onClick={fetchNews} className="news-retry-btn">Retry</button>
        </div>
      </div>
    );
  }

  return (
    <div className="news-section">
      <div className="news-section-header">
        <div className="news-tabs">
          {CATEGORIES.map(cat => (
            <button
              key={cat.key}
              className={`news-tab ${activeCategory === cat.key ? 'active' : ''}`}
              onClick={() => setActiveCategory(cat.key)}
            >
              {cat.label}
            </button>
          ))}
        </div>
        <span className="news-count">{filteredNews.length} of {totalCount} posts</span>
      </div>

      <div className="news-cards-grid">
        {filteredNews.map((item, index) => (
          <a
            key={`${item.source}-${index}`}
            href={item.url}
            target="_blank"
            rel="noopener noreferrer"
            className="news-card"
          >
            <div className="card-visual" style={{ background: getCardGradient(item) }}>
              <span className="card-source-icon">
                {SOURCE_ICONS[item.source] || item.source.substring(0, 2).toUpperCase()}
              </span>
              <div className="card-badges">
                {getBadges(item).map((badge, bi) => (
                  <span key={bi} className={`card-badge ${badge.className}`}>{badge.label}</span>
                ))}
              </div>
            </div>
            <div className="card-body">
              <h3 className="card-title">{item.title}</h3>
              {item.description && (
                <p className="card-description">{item.description}</p>
              )}
              <div className="card-footer">
                <span className="card-source">{item.source}</span>
                <span className="card-date">{formatDate(item.published)}</span>
              </div>
            </div>
          </a>
        ))}
      </div>

      {filteredNews.length === 0 && (
        <div className="news-empty">No news found for this category</div>
      )}
    </div>
  );
};

export default NewsFeed;
