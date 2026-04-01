"""
Recolector de datos de mercado financiero.

Usa solo stdlib (urllib) para obtener datos de mercado.
Fuentes: Yahoo Finance API (gratuita, sin auth), Alpha Vantage (con key opcional).

Variables que mide:
- Precio actual, OHLC, volumen
- Variacion % diaria, semanal, mensual
- RSI, MACD simplificados (calculados internamente)
- Sentimiento implicito en volumen anormal
"""

import json
import time
import math
import logging
import urllib.request
import urllib.parse
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque

logger = logging.getLogger("autoia.prediction.market")


@dataclass
class PriceBar:
    """OHLCV bar."""
    timestamp:  float
    open:       float
    high:       float
    low:        float
    close:      float
    volume:     float

    @property
    def body_pct(self) -> float:
        """Tamaño del cuerpo del candlestick como % del rango."""
        rng = self.high - self.low
        if rng == 0:
            return 0.0
        return abs(self.close - self.open) / rng

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def change_pct(self) -> float:
        if self.open == 0:
            return 0.0
        return (self.close - self.open) / self.open * 100


@dataclass
class MarketSnapshot:
    """Estado actual de un instrumento financiero."""
    symbol:       str
    name:         str
    price:        float
    change_pct:   float     # % cambio desde apertura
    volume:       float
    avg_volume:   float
    timestamp:    float = field(default_factory=time.time)
    week_change:  float = 0.0
    month_change: float = 0.0
    rsi:          float = 50.0   # 0-100
    macd_signal:  str = "neutral"  # "bullish" | "bearish" | "neutral"
    behavioral_signal: str = ""   # señal ABA

    @property
    def volume_anomaly(self) -> float:
        """Anomalia de volumen: >1.5 = volumen inusualmente alto."""
        if self.avg_volume == 0:
            return 1.0
        return self.volume / self.avg_volume

    @property
    def is_overbought(self) -> bool:
        return self.rsi > 70

    @property
    def is_oversold(self) -> bool:
        return self.rsi < 30


class MarketCollector:
    """
    Recolecta datos de mercado financiero.

    Sin API key: Yahoo Finance (datos con 15min de delay).
    Los datos se usan para alimentar el motor ABA.
    """

    YF_BASE = "https://query1.finance.yahoo.com/v8/finance/chart/"

    def __init__(self, symbols: List[str] = None):
        self.symbols   = symbols or []
        self.cache:    Dict[str, MarketSnapshot] = {}
        self.history:  Dict[str, deque] = {}  # symbol -> deque of PriceBar
        self._cache_ttl = 60  # segundos
        self._last_fetch: Dict[str, float] = {}

    def add_symbol(self, symbol: str):
        if symbol not in self.symbols:
            self.symbols.append(symbol)
            self.history[symbol] = deque(maxlen=200)

    def fetch(self, symbol: str, force: bool = False) -> Optional[MarketSnapshot]:
        """
        Obtiene datos actuales del simbolo.
        Usa cache si los datos tienen menos de 60s.
        """
        now = time.time()
        last = self._last_fetch.get(symbol, 0)
        if not force and (now - last) < self._cache_ttl and symbol in self.cache:
            return self.cache[symbol]

        try:
            url = f"{self.YF_BASE}{urllib.parse.quote(symbol)}?interval=1d&range=1mo"
            req = urllib.request.Request(url, headers={
                "User-Agent": "Mozilla/5.0",
                "Accept": "application/json",
            })
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            result = data.get("chart", {}).get("result", [])
            if not result:
                return None

            r        = result[0]
            meta     = r.get("meta", {})
            quotes   = r.get("indicators", {}).get("quote", [{}])[0]
            timestamps = r.get("timestamp", [])

            closes  = [c for c in (quotes.get("close") or []) if c is not None]
            volumes = [v for v in (quotes.get("volume") or []) if v is not None]
            opens   = [o for o in (quotes.get("open") or []) if o is not None]
            highs   = [h for h in (quotes.get("high") or []) if h is not None]
            lows    = [l for l in (quotes.get("low") or []) if l is not None]

            if not closes:
                return None

            current_price = meta.get("regularMarketPrice") or closes[-1]
            prev_close    = meta.get("chartPreviousClose") or (closes[-2] if len(closes) > 1 else closes[-1])
            change_pct    = ((current_price - prev_close) / prev_close * 100
                             if prev_close != 0 else 0.0)

            avg_vol = sum(volumes[-20:]) / len(volumes[-20:]) if volumes else 0

            # RSI simplificado (14 periodos)
            rsi = self._calculate_rsi(closes)

            # Cambios semana/mes
            week_change = ((closes[-1] - closes[-5]) / closes[-5] * 100
                           if len(closes) >= 5 else 0.0)
            month_change = ((closes[-1] - closes[0]) / closes[0] * 100
                            if len(closes) >= 20 else 0.0)

            # MACD simplificado
            macd_signal = self._macd_signal(closes)

            # Añadir al historial
            if symbol not in self.history:
                self.history[symbol] = deque(maxlen=200)
            if timestamps and len(closes) == len(timestamps):
                for i, (ts, c, v) in enumerate(zip(timestamps, closes, volumes)):
                    if i < len(opens) and i < len(highs) and i < len(lows):
                        self.history[symbol].append(PriceBar(
                            timestamp=ts, open=opens[i], high=highs[i],
                            low=lows[i], close=c, volume=v
                        ))

            snap = MarketSnapshot(
                symbol=symbol,
                name=meta.get("shortName", symbol),
                price=round(current_price, 4),
                change_pct=round(change_pct, 3),
                volume=volumes[-1] if volumes else 0,
                avg_volume=avg_vol,
                week_change=round(week_change, 3),
                month_change=round(month_change, 3),
                rsi=round(rsi, 1),
                macd_signal=macd_signal,
            )
            # Señal conductual ABA
            snap.behavioral_signal = self._aba_signal(snap)

            self.cache[symbol] = snap
            self._last_fetch[symbol] = now
            logger.info(f"Market {symbol}: ${snap.price} {snap.change_pct:+.2f}%"
                        f" RSI={snap.rsi} [{snap.behavioral_signal}]")
            return snap

        except Exception as e:
            logger.warning(f"No se pudo obtener {symbol}: {e}")
            return self.cache.get(symbol)

    def fetch_all(self) -> Dict[str, MarketSnapshot]:
        """Obtiene datos de todos los simbolos configurados."""
        results = {}
        for sym in self.symbols:
            snap = self.fetch(sym)
            if snap:
                results[sym] = snap
        return results

    def _calculate_rsi(self, closes: List[float], period: int = 14) -> float:
        """RSI simplificado."""
        if len(closes) < period + 1:
            return 50.0
        deltas = [closes[i] - closes[i-1] for i in range(1, len(closes))]
        recent = deltas[-period:]
        gains = [d for d in recent if d > 0]
        losses = [-d for d in recent if d < 0]
        avg_gain = sum(gains) / period if gains else 0.001
        avg_loss = sum(losses) / period if losses else 0.001
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))

    def _macd_signal(self, closes: List[float]) -> str:
        """MACD simplificado (EMA12 vs EMA26)."""
        if len(closes) < 26:
            return "neutral"

        def ema(data, period):
            k = 2 / (period + 1)
            ema_val = data[0]
            for d in data[1:]:
                ema_val = d * k + ema_val * (1 - k)
            return ema_val

        ema12 = ema(closes[-26:], 12)
        ema26 = ema(closes[-26:], 26)
        macd  = ema12 - ema26

        if macd > 0 and closes[-1] > closes[-2]:
            return "bullish"
        elif macd < 0 and closes[-1] < closes[-2]:
            return "bearish"
        return "neutral"

    def _aba_signal(self, snap: MarketSnapshot) -> str:
        """
        Señal conductual ABA basada en el estado tecnico.
        Traduce indicadores tecnicos a lenguaje de conducta de masas.
        """
        signals = []
        if snap.volume_anomaly > 2.0:
            signals.append("VOLUMEN_ANOMALO")   # Activacion masiva
        if snap.is_overbought:
            signals.append("SACIACION")          # AO: el reforzador pierde valor
        elif snap.is_oversold:
            signals.append("PRIVACION")          # EO: el reforzador aumenta valor
        if snap.change_pct > 3:
            signals.append("REFUERZO_POSITIVO")  # Conducta reforzada hoy
        elif snap.change_pct < -3:
            signals.append("CASTIGO")            # Conducta castigada hoy
        if snap.macd_signal == "bullish":
            signals.append("MOMENTUM_ALCISTA")
        elif snap.macd_signal == "bearish":
            signals.append("MOMENTUM_BAJISTA")
        return " | ".join(signals) if signals else "NEUTRO"

    def get_correlation_matrix(self) -> Dict[str, Dict[str, float]]:
        """
        Calcula correlacion de precios entre todos los simbolos.
        Util para detectar conducta grupal de la masa inversora.
        """
        matrix = {}
        symbols_with_data = [s for s in self.symbols
                             if s in self.history and len(self.history[s]) > 10]
        for s1 in symbols_with_data:
            matrix[s1] = {}
            c1 = [b.close for b in self.history[s1]]
            for s2 in symbols_with_data:
                if s1 == s2:
                    matrix[s1][s2] = 1.0
                    continue
                c2 = [b.close for b in self.history[s2]]
                n = min(len(c1), len(c2))
                if n < 5:
                    matrix[s1][s2] = 0.0
                    continue
                c1n, c2n = c1[-n:], c2[-n:]
                mean1, mean2 = sum(c1n)/n, sum(c2n)/n
                num = sum((c1n[i]-mean1)*(c2n[i]-mean2) for i in range(n))
                d1  = math.sqrt(sum((c-mean1)**2 for c in c1n))
                d2  = math.sqrt(sum((c-mean2)**2 for c in c2n))
                matrix[s1][s2] = round(num / (d1*d2 + 1e-9), 3)
        return matrix
