"""
SCENARIO ENGINE - Separate Gate Layer

This engine:
- Runs FIRST, before any model logic
- Has OVERRIDE authority (can block all trades)
- Is completely independent of ML models
- Detects environmental facts, not predictions

Scenarios are EXTERNAL STATES:
- Market crash
- News shock  
- Weekend liquidity
- Overtrading behavior
- FOMO detection

This separation ensures:
- Clean debugging
- Safe overrides
- Independent evolution
- Clear explainability
"""

from datetime import datetime
from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel


# ============================================================
# SCENARIO TYPES & MODELS
# ============================================================

class ScenarioDecision(str, Enum):
    """What the scenario layer decides"""
    ALLOW = "ALLOW"           # Proceed to model
    BLOCK = "BLOCK"           # Stop here, no trading
    MODIFY = "MODIFY"         # Proceed but with adjustments


class ScenarioSeverity(str, Enum):
    """Severity levels"""
    CRITICAL = "CRITICAL"     # Blocks everything
    HIGH = "HIGH"             # Blocks most trades
    MEDIUM = "MEDIUM"         # Modifies parameters
    LOW = "LOW"               # Warning only


class Scenario(BaseModel):
    """A detected scenario"""
    type: str
    severity: ScenarioSeverity
    icon: str
    title: str
    message: str
    effect: str
    details: Optional[str] = None
    
    class Config:
        use_enum_values = True


class ScenarioResult(BaseModel):
    """Output from Scenario Engine"""
    decision: ScenarioDecision
    can_trade: bool
    active_scenarios: List[Scenario]
    
    # Modifications (only if decision is MODIFY)
    position_size_multiplier: float = 1.0
    threshold_boost: float = 0.0
    
    # For blocked trades
    block_reason: Optional[str] = None
    resume_check: Optional[str] = None  # When to check again
    
    class Config:
        use_enum_values = True


# ============================================================
# SCENARIO ENGINE
# ============================================================

class ScenarioEngine:
    """
    Independent gate layer that runs BEFORE any model logic.
    
    Key principles:
    1. Scenarios are facts, not predictions
    2. Has absolute override authority
    3. Completely decoupled from ML models
    4. Fast and deterministic
    """
    
    def __init__(self):
        # Manual flags for news events (could be API-driven later)
        self.active_news_flags: List[str] = []
        
        # Configuration
        self.config = {
            'btc_crash_1h_threshold': -5.0,      # BTC drop > 5% in 1h
            'btc_crash_24h_threshold': -10.0,    # BTC drop > 10% in 24h
            'btc_stress_1h_threshold': -3.0,
            'btc_stress_24h_threshold': -6.0,
            'extended_move_thresholds': {
                'BTC_USDT': 8,
                'ETH_USDT': 12,
                'SOL_USDT': 18,
                'PEPE_USDT': 25
            },
            'overextended_ema_distance': 10,     # % from EMA20
            'max_consecutive_losses': 3,
            'default_max_daily_trades': 5
        }
    
    # ============================================================
    # MAIN ENTRY POINT
    # ============================================================
    
    def evaluate(
        self,
        btc_data: Dict[str, Any],
        coin: str,
        coin_data: Any,  # DataFrame or None
        user_context: Dict[str, Any]
    ) -> ScenarioResult:
        """
        Main entry point. Evaluates all scenarios.
        
        Returns ScenarioResult with:
        - decision: ALLOW / BLOCK / MODIFY
        - active_scenarios: List of detected scenarios
        - modifications: If MODIFY, what parameters to adjust
        """
        
        scenarios: List[Scenario] = []
        
        # 1. Market-level scenarios (highest priority)
        crash_scenario = self._check_market_crash(btc_data)
        if crash_scenario:
            scenarios.append(crash_scenario)
        
        # 2. Time-based scenarios
        time_scenario = self._check_time_based()
        if time_scenario:
            scenarios.append(time_scenario)
        
        # 3. News scenarios (manual flags)
        news_scenario = self._check_news_flags()
        if news_scenario:
            scenarios.append(news_scenario)
        
        # 4. Asset-specific scenarios
        extended_scenario = self._check_extended_move(coin, coin_data)
        if extended_scenario:
            scenarios.append(extended_scenario)
        
        # 5. User behavior scenarios
        behavior_scenarios = self._check_user_behavior(user_context)
        scenarios.extend(behavior_scenarios)
        
        # ============================================
        # DETERMINE FINAL DECISION
        # ============================================
        return self._compute_decision(scenarios)
    
    # ============================================================
    # SCENARIO DETECTORS
    # ============================================================
    
    def _check_market_crash(self, btc_data: Dict[str, Any]) -> Optional[Scenario]:
        """Detect market crash/panic - HIGHEST PRIORITY"""
        
        if not btc_data:
            return None
        
        change_1h = btc_data.get('change_1h', 0)
        change_24h = btc_data.get('change_24h', 0)
        
        # CRITICAL: Severe crash
        if change_1h <= self.config['btc_crash_1h_threshold'] or \
           change_24h <= self.config['btc_crash_24h_threshold']:
            return Scenario(
                type='MARKET_CRASH',
                severity=ScenarioSeverity.CRITICAL,
                icon='🚨',
                title='Market Crash Detected',
                message='Market panic detected – capital preservation mode',
                effect='BLOCK_ALL',
                details=f'BTC: {change_1h:+.1f}% (1h), {change_24h:+.1f}% (24h)'
            )
        
        # HIGH: Market stress
        if change_1h <= self.config['btc_stress_1h_threshold'] or \
           change_24h <= self.config['btc_stress_24h_threshold']:
            return Scenario(
                type='MARKET_STRESS',
                severity=ScenarioSeverity.HIGH,
                icon='⚠️',
                title='Market Under Stress',
                message='Elevated volatility – reduce exposure',
                effect='REDUCE_SIZE_50',
                details=f'BTC: {change_1h:+.1f}% (1h), {change_24h:+.1f}% (24h)'
            )
        
        return None
    
    def _check_time_based(self) -> Optional[Scenario]:
        """Detect weekend/low liquidity periods"""
        
        now = datetime.now()
        
        # Weekend
        if now.weekday() >= 5:
            return Scenario(
                type='WEEKEND',
                severity=ScenarioSeverity.LOW,
                icon='📅',
                title='Weekend Trading',
                message='Lower liquidity, wider spreads expected',
                effect='REDUCE_SIZE_25',
                details=f'{now.strftime("%A")} - reduced market activity'
            )
        
        # Off-hours (simplified - could be more sophisticated)
        if 0 <= now.hour < 6:
            return Scenario(
                type='LOW_LIQUIDITY_HOURS',
                severity=ScenarioSeverity.LOW,
                icon='🌙',
                title='Off-Peak Hours',
                message='Low liquidity trading hours',
                effect='WARNING',
                details='Consider waiting for major market open'
            )
        
        return None
    
    def _check_news_flags(self) -> Optional[Scenario]:
        """Check manually set news flags"""
        
        if not self.active_news_flags:
            return None
        
        # Return highest priority news flag
        for flag in self.active_news_flags:
            if flag in ['FED_DECISION', 'CPI_RELEASE', 'FOMC']:
                return Scenario(
                    type='MACRO_NEWS',
                    severity=ScenarioSeverity.HIGH,
                    icon='📰',
                    title='Major News Event',
                    message=f'{flag} - expect high volatility',
                    effect='BLOCK_UNTIL_CLEAR',
                    details='Wait 1-2 hours after release'
                )
            
            if flag in ['EXCHANGE_HACK', 'MAJOR_EXPLOIT']:
                return Scenario(
                    type='CRYPTO_NEWS',
                    severity=ScenarioSeverity.CRITICAL,
                    icon='🔴',
                    title='Crypto Emergency',
                    message=f'{flag} - trading suspended',
                    effect='BLOCK_ALL',
                    details='Wait for clarity before trading'
                )
        
        return None
    
    def _check_extended_move(self, coin: str, coin_data: Any) -> Optional[Scenario]:
        """Detect if asset has already moved significantly"""
        
        if coin_data is None or len(coin_data) < 48:
            return None
        
        try:
            close = coin_data['close'].values
            
            # 24h move
            move_24h = (close[-1] - close[-24]) / close[-24] * 100 if len(close) >= 24 else 0
            
            threshold = self.config['extended_move_thresholds'].get(coin, 15)
            
            if abs(move_24h) > threshold:
                direction = "up" if move_24h > 0 else "down"
                return Scenario(
                    type='EXTENDED_MOVE',
                    severity=ScenarioSeverity.MEDIUM,
                    icon='📊',
                    title='Extended Move Detected',
                    message=f'Asset already moved {abs(move_24h):.1f}% {direction}',
                    effect='RAISE_THRESHOLD',
                    details='Late entry risk – consider waiting for pullback'
                )
            
            # Check EMA distance
            if len(close) >= 20:
                import pandas as pd
                ema20 = pd.Series(close).ewm(span=20).mean().iloc[-1]
                distance = (close[-1] - ema20) / ema20 * 100
                
                if abs(distance) > self.config['overextended_ema_distance']:
                    direction = "above" if distance > 0 else "below"
                    return Scenario(
                        type='OVEREXTENDED',
                        severity=ScenarioSeverity.LOW,
                        icon='📈' if distance > 0 else '📉',
                        title='Price Overextended',
                        message=f'Price {abs(distance):.1f}% {direction} EMA20',
                        effect='WARNING',
                        details='Mean reversion likely'
                    )
        except Exception:
            pass
        
        return None
    
    def _check_user_behavior(self, user_context: Dict[str, Any]) -> List[Scenario]:
        """Detect problematic user behavior patterns"""
        
        scenarios = []
        
        # FOMO Detection
        reason = user_context.get('reason_for_trade')
        if reason == 'FOMO':
            scenarios.append(Scenario(
                type='FOMO_DETECTED',
                severity=ScenarioSeverity.HIGH,
                icon='😰',
                title='FOMO Detected',
                message='Emotional trading leads to losses',
                effect='RAISE_THRESHOLD',
                details='Take a breath. Good trades come from strategy.'
            ))
        
        if reason == 'TIP':
            scenarios.append(Scenario(
                type='TIP_BASED',
                severity=ScenarioSeverity.MEDIUM,
                icon='💬',
                title='Tip-Based Trade',
                message='Verify with your own analysis',
                effect='WARNING',
                details='Tips are often wrong. DYOR.'
            ))
        
        # Overtrading
        trades_today = user_context.get('trades_today', 0)
        max_trades = user_context.get('max_daily_trades', self.config['default_max_daily_trades'])
        
        if trades_today >= max_trades:
            scenarios.append(Scenario(
                type='OVERTRADING',
                severity=ScenarioSeverity.CRITICAL,
                icon='🛑',
                title='Trade Limit Reached',
                message=f'Daily limit reached ({trades_today}/{max_trades})',
                effect='BLOCK_ALL',
                details='Overtrading destroys profits. Rest and reset.'
            ))
        elif trades_today >= max_trades - 1:
            scenarios.append(Scenario(
                type='NEAR_LIMIT',
                severity=ScenarioSeverity.MEDIUM,
                icon='⚠️',
                title='Approaching Trade Limit',
                message=f'{trades_today}/{max_trades} trades today',
                effect='WARNING',
                details='Make this trade count.'
            ))
        
        # Losing streak
        recent_losses = user_context.get('recent_losses', 0)
        
        if recent_losses >= self.config['max_consecutive_losses']:
            scenarios.append(Scenario(
                type='LOSING_STREAK',
                severity=ScenarioSeverity.CRITICAL,
                icon='🛑',
                title='Losing Streak',
                message=f'{recent_losses} consecutive losses – take a break',
                effect='BLOCK_ALL',
                details='Step away. Review. Come back fresh.'
            ))
        elif recent_losses == 2:
            scenarios.append(Scenario(
                type='CONSECUTIVE_LOSSES',
                severity=ScenarioSeverity.MEDIUM,
                icon='⚠️',
                title='Back-to-Back Losses',
                message='2 consecutive losses – trade with caution',
                effect='REDUCE_SIZE_50',
                details='Don\'t chase losses.'
            ))
        
        return scenarios
    
    # ============================================================
    # DECISION COMPUTATION
    # ============================================================
    
    def _compute_decision(self, scenarios: List[Scenario]) -> ScenarioResult:
        """Compute final decision from all scenarios"""
        
        if not scenarios:
            return ScenarioResult(
                decision=ScenarioDecision.ALLOW,
                can_trade=True,
                active_scenarios=[]
            )
        
        # Check for blocking scenarios
        blocking_scenarios = [s for s in scenarios if s.severity == ScenarioSeverity.CRITICAL]
        
        if blocking_scenarios:
            return ScenarioResult(
                decision=ScenarioDecision.BLOCK,
                can_trade=False,
                active_scenarios=scenarios,
                block_reason=blocking_scenarios[0].message,
                resume_check='1H'
            )
        
        # Check for high-severity scenarios
        high_scenarios = [s for s in scenarios if s.severity == ScenarioSeverity.HIGH]
        
        if high_scenarios:
            # High severity = significant modifications
            return ScenarioResult(
                decision=ScenarioDecision.MODIFY,
                can_trade=True,
                active_scenarios=scenarios,
                position_size_multiplier=0.5,
                threshold_boost=0.10
            )
        
        # Calculate modifications from remaining scenarios
        position_mult = 1.0
        threshold_boost = 0.0
        
        for scenario in scenarios:
            effect = scenario.effect
            
            if effect == 'REDUCE_SIZE_50':
                position_mult *= 0.5
            elif effect == 'REDUCE_SIZE_25':
                position_mult *= 0.75
            elif effect == 'RAISE_THRESHOLD':
                threshold_boost += 0.05
        
        # If any modifications, return MODIFY
        if position_mult < 1.0 or threshold_boost > 0:
            return ScenarioResult(
                decision=ScenarioDecision.MODIFY,
                can_trade=True,
                active_scenarios=scenarios,
                position_size_multiplier=position_mult,
                threshold_boost=threshold_boost
            )
        
        # Otherwise just warnings
        return ScenarioResult(
            decision=ScenarioDecision.ALLOW,
            can_trade=True,
            active_scenarios=scenarios
        )
    
    # ============================================================
    # NEWS FLAG MANAGEMENT
    # ============================================================
    
    def set_news_flag(self, flag: str):
        """Set a news flag (manual or API-driven)"""
        if flag not in self.active_news_flags:
            self.active_news_flags.append(flag)
    
    def clear_news_flag(self, flag: str):
        """Clear a news flag"""
        if flag in self.active_news_flags:
            self.active_news_flags.remove(flag)
    
    def clear_all_news_flags(self):
        """Clear all news flags"""
        self.active_news_flags = []
    
    def get_active_news_flags(self) -> List[str]:
        """Get current news flags"""
        return self.active_news_flags.copy()