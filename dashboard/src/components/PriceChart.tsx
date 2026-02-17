import React, { useEffect, useRef, useState, useCallback } from 'react';
import { createChart, ColorType, CrosshairMode, CandlestickSeries, AreaSeries } from 'lightweight-charts';
import type { IChartApi, ISeriesApi, CandlestickData, AreaData, Time } from 'lightweight-charts';
import axios from 'axios';
import './PriceChart.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:8000';

interface PriceChartProps {
  coin: string;
  coinColor: string;
}

interface KlineData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

const TIMEFRAMES = [
  { label: '1H', interval: '1m', limit: 60 },
  { label: '6H', interval: '5m', limit: 72 },
  { label: '1D', interval: '15m', limit: 96 },
  { label: '3D', interval: '1h', limit: 72 },
  { label: '7D', interval: '1h', limit: 168 },
  { label: '1M', interval: '4h', limit: 180 },
  { label: '3M', interval: '1d', limit: 90 },
  { label: '1Y', interval: '1d', limit: 365 },
  { label: 'ALL', interval: '1w', limit: 500 },
];

const PriceChart: React.FC<PriceChartProps> = ({ coin, coinColor }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | ISeriesApi<'Area'> | null>(null);
  const [activeTimeframe, setActiveTimeframe] = useState(4); // Default 7D
  const [chartType, setChartType] = useState<'area' | 'candlestick'>('area');
  const [loading, setLoading] = useState(false);
  const [priceInfo, setPriceInfo] = useState<{ price: number; change: number; changePct: number } | null>(null);

  const formatPrice = (p: number): string => {
    if (p < 0.0001) return p.toFixed(8);
    if (p < 0.01) return p.toFixed(6);
    if (p < 1) return p.toFixed(4);
    if (p < 1000) return p.toFixed(2);
    return p.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  };

  const fetchData = useCallback(async (tf: typeof TIMEFRAMES[number]) => {
    setLoading(true);
    try {
      const response = await axios.get(`${API_BASE}/klines/${coin}`, {
        params: { interval: tf.interval, limit: tf.limit }
      });
      const data: KlineData[] = response.data.data || [];
      if (data.length > 0) {
        const first = data[0];
        const last = data[data.length - 1];
        const change = last.close - first.open;
        const changePct = (change / first.open) * 100;
        setPriceInfo({ price: last.close, change, changePct });
      }
      return data;
    } catch (error) {
      console.error('Error fetching klines:', error);
      return [];
    } finally {
      setLoading(false);
    }
  }, [coin]);

  // Create and update chart
  useEffect(() => {
    if (!chartContainerRef.current) return;

    // Remove existing chart
    if (chartRef.current) {
      chartRef.current.remove();
      chartRef.current = null;
      seriesRef.current = null;
    }

    const container = chartContainerRef.current;

    const chart = createChart(container, {
      layout: {
        background: { type: ColorType.Solid, color: 'transparent' },
        textColor: '#888',
        fontSize: 11,
      },
      grid: {
        vertLines: { color: '#1a1a2e' },
        horzLines: { color: '#1a1a2e' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
        vertLine: { color: '#444', width: 1, style: 3, labelBackgroundColor: '#1a1a2e' },
        horzLine: { color: '#444', width: 1, style: 3, labelBackgroundColor: '#1a1a2e' },
      },
      rightPriceScale: {
        borderColor: '#222',
        scaleMargins: { top: 0.1, bottom: 0.1 },
      },
      timeScale: {
        borderColor: '#222',
        timeVisible: true,
        secondsVisible: false,
      },
      width: container.clientWidth,
      height: 400,
    });

    chartRef.current = chart;

    // Add series based on chart type
    const tf = TIMEFRAMES[activeTimeframe];
    fetchData(tf).then((data) => {
      if (!chartRef.current || data.length === 0) return;

      if (chartType === 'candlestick') {
        const series = chart.addSeries(CandlestickSeries, {
          upColor: '#00e676',
          downColor: '#ff5252',
          borderUpColor: '#00e676',
          borderDownColor: '#ff5252',
          wickUpColor: '#00e676',
          wickDownColor: '#ff5252',
        });
        const candleData: CandlestickData<Time>[] = data.map(d => ({
          time: d.time as Time,
          open: d.open,
          high: d.high,
          low: d.low,
          close: d.close,
        }));
        series.setData(candleData);
        seriesRef.current = series;
      } else {
        const series = chart.addSeries(AreaSeries, {
          lineColor: coinColor,
          topColor: `${coinColor}40`,
          bottomColor: `${coinColor}05`,
          lineWidth: 2,
          crosshairMarkerVisible: true,
          crosshairMarkerRadius: 4,
        });
        const areaData: AreaData<Time>[] = data.map(d => ({
          time: d.time as Time,
          value: d.close,
        }));
        series.setData(areaData);
        seriesRef.current = series;
      }

      chart.timeScale().fitContent();
    });

    // Responsive resize
    const resizeObserver = new ResizeObserver(entries => {
      for (const entry of entries) {
        const { width } = entry.contentRect;
        if (chartRef.current) {
          chartRef.current.applyOptions({ width });
        }
      }
    });
    resizeObserver.observe(container);

    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        chartRef.current.remove();
        chartRef.current = null;
        seriesRef.current = null;
      }
    };
  }, [coin, activeTimeframe, chartType, coinColor, fetchData]);

  const handleTimeframeChange = (index: number) => {
    setActiveTimeframe(index);
  };

  return (
    <section className="price-chart-section">
      <div className="chart-header">
        <div className="chart-title-row">
          <h2>Price {coin.replace('_', '/')}</h2>
          {priceInfo && (
            <div className="chart-price-info">
              <span className="chart-current-price">${formatPrice(priceInfo.price)}</span>
              <span className={`chart-change ${priceInfo.changePct >= 0 ? 'positive' : 'negative'}`}>
                {priceInfo.changePct >= 0 ? '+' : ''}{priceInfo.changePct.toFixed(2)}%
              </span>
            </div>
          )}
        </div>

        <div className="chart-controls">
          <div className="timeframe-buttons">
            {TIMEFRAMES.map((tf, i) => (
              <button
                key={tf.label}
                className={`tf-btn ${activeTimeframe === i ? 'active' : ''}`}
                style={activeTimeframe === i ? { backgroundColor: coinColor, color: '#000' } : {}}
                onClick={() => handleTimeframeChange(i)}
              >
                {tf.label}
              </button>
            ))}
          </div>

          <div className="chart-type-toggle">
            <button
              className={`type-btn ${chartType === 'area' ? 'active' : ''}`}
              onClick={() => setChartType('area')}
              title="Area chart"
            >
              Area
            </button>
            <button
              className={`type-btn ${chartType === 'candlestick' ? 'active' : ''}`}
              onClick={() => setChartType('candlestick')}
              title="Candlestick chart"
            >
              Candles
            </button>
          </div>
        </div>
      </div>

      <div className="chart-container" ref={chartContainerRef}>
        {loading && (
          <div className="chart-loading">
            <div className="chart-spinner"></div>
          </div>
        )}
      </div>
    </section>
  );
};

export default PriceChart;
