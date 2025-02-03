import os
import sys
import time
import math
import json
import asyncio
import logging
import datetime
import threading
import argparse
from functools import wraps, partial
from typing import Any, Dict, Tuple, List, Optional, Callable, Union

import concurrent.futures
import multiprocessing as mp

import numpy as np
import pandas as pd
import yfinance as yf
from numpy.linalg import svd

# Third-party libraries
try:
    from numba import njit, prange
except ImportError:
    sys.exit("Please install numba (pip install numba)")

try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, lax, random
except ImportError:
    sys.exit("Please install JAX (pip install jax jaxlib)")

try:
    from transformers import pipeline
except ImportError:
    sys.exit("Please install transformers (pip install transformers)")

try:
    from bayes_opt import BayesianOptimization
except ImportError:
    sys.exit("Please install bayesian-optimization (pip install bayesian-optimization)")

try:
    import dash
    from dash import dcc, html, Output, Input
    import dash_bootstrap_components as dbc
except ImportError:
    sys.exit("Please install dash and dash-bootstrap-components (pip install dash dash-bootstrap-components)")

try:
    import metalcompute as mc
except ImportError:
    sys.exit("Please install metalcompute (pip install metalcompute)")

# -----------------------------
# Global Configuration
# -----------------------------
CACHE_FILE: str = 'data_cache.json'
LOG_FILE: str = "model_log.txt"
RUN_DAY: int = 0  # Monday (0)
RUN_HOUR: int = 16

# -----------------------------
# Logger Setup
# -----------------------------
logger: logging.Logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)
file_handler = logging.FileHandler(LOG_FILE, mode='w')
file_handler.setFormatter(formatter)
logger.addHandler(stream_handler)
logger.addHandler(file_handler)

# -----------------------------
# Custom Exception Classes
# -----------------------------
class NetworkError(Exception):
    """Exception raised for network-related errors."""
    pass

class DataFetchError(Exception):
    """Exception raised when data fetching fails."""
    pass

class GPUInitializationError(Exception):
    """Exception raised when GPU accelerator initialization fails."""
    pass

class ModelTrainingError(Exception):
    """Exception raised during model training errors."""
    pass

# -----------------------------
# Decorators
# -----------------------------
def retry(exceptions: Tuple[Exception, ...], tries: int = 3, delay: float = 1, backoff: int = 2) -> Callable:
    """
    Retry decorator for synchronous functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return func(*args, **kwargs)
                except exceptions as e:
                    logger.error(f"{func.__name__} failed with {e}, retrying in {mdelay} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def async_retry(exceptions: Tuple[Exception, ...], tries: int = 3, delay: float = 1, backoff: int = 2) -> Callable:
    """
    Retry decorator for asynchronous functions.
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            mtries, mdelay = tries, delay
            while mtries > 1:
                try:
                    return await func(*args, **kwargs)
                except exceptions as e:
                    logger.error(f"{func.__name__} (async) failed with {e}, retrying in {mdelay} seconds...")
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def profile(func: Callable) -> Callable:
    """
    Profiling decorator that logs the execution time.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        end = time.perf_counter()
        logger.info(f"{func.__name__} took {end - start:.4f} seconds")
        return result
    return wrapper

# -----------------------------
# Utility Functions
# -----------------------------
def atomic_save(cache: Dict[str, Any], filename: str) -> None:
    """Atomically save the cache dictionary as JSON."""
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, 'w') as f:
            json.dump(cache, f)
        os.replace(temp_filename, filename)
    except Exception as e:
        logger.error(f"Error saving cache: {e}")

# -----------------------------
# Optimized CPU Kernel (Numba)
# -----------------------------
@njit(parallel=True, fastmath=True)
def fast_cpu_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the squared distance between two arrays using Numba for acceleration."""
    n = x.shape[0]
    total = 0.0
    for i in prange(n):
        diff = x[i] - y[i]
        total += diff * diff
    return total

# -----------------------------
# GPU Accelerator using Metal (metalcompute)
# -----------------------------
class GPUAccelerator:
    """
    GPU accelerator using Apple's Metal API via the metalcompute package.
    Provides element-wise vector multiplication.
    """
    def __init__(self) -> None:
        try:
            self.device = mc.Device()
            logger.info(f"Metal GPU Accelerator initialized with device: {self.device.name}")
        except Exception as e:
            logger.error(f"Failed to initialize Metal GPU Accelerator: {e}")
            raise GPUInitializationError(e)
        
        self.kernel_code: str = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void vec_mult(const device float* a [[ buffer(0) ]],
                             const device float* b [[ buffer(1) ]],
                             device float* result [[ buffer(2) ]],
                             uint id [[ thread_position_in_grid ]]) {
            result[id] = a[id] * b[id];
        }
        """
        try:
            self.compiled_kernel = self.device.kernel(self.kernel_code).function("vec_mult")
        except Exception as e:
            logger.error(f"Kernel compilation failed: {e}")
            raise GPUInitializationError(e)
    
    def vector_multiply(self, a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
        """Multiply two vectors using the GPU."""
        a = np.ascontiguousarray(a_np.astype(np.float32))
        b = np.ascontiguousarray(b_np.astype(np.float32))
        count = a.shape[0]
        from array import array
        A_py = array('f', a.tolist())
        B_py = array('f', b.tolist())
        bytes_count = count * 4
        result_buf = self.device.buffer(bytes_count)
        try:
            self.compiled_kernel(count, A_py, B_py, result_buf)
        except Exception as e:
            logger.error(f"Kernel execution failed: {e}")
            raise GPUInitializationError(e)
        result_view = memoryview(result_buf).cast('f')
        result_np = np.array(result_view)
        return result_np

# -----------------------------
# FinBERT Sentiment Analysis
# -----------------------------
class SentimentTransformer:
    """
    FinBERT-based sentiment transformer for analyzing financial text sentiment.
    """
    def __init__(self) -> None:
        try:
            self.pipeline = pipeline("sentiment-analysis",
                                     model="yiyanghkust/finbert-pretrain",
                                     tokenizer="yiyanghkust/finbert-pretrain")
            logger.info("Loaded FinBERT-based sentiment transformer.")
        except Exception as e:
            logger.error(f"Failed to load FinBERT transformer: {e}")
            raise

    def analyze(self, text: str) -> float:
        """Analyze the sentiment of the text and return a score."""
        result = self.pipeline(text[:512])
        if result and result[0]['label'].upper() == "POSITIVE":
            return result[0]['score']
        elif result and result[0]['label'].upper() == "NEGATIVE":
            return -result[0]['score']
        return 0.0

    @staticmethod
    def analyze_static(text: str) -> float:
        return SentimentTransformer().analyze(text)

# -----------------------------
# Data Fetching & Caching
# -----------------------------
class DataFetcher:
    """
    Fetches stock data and news sentiment while caching results.
    """
    def __init__(self, cache_file: str = CACHE_FILE) -> None:
        self.cache_file: str = cache_file
        self.cache: Dict[str, Any] = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
                self.cache = {}
        self._sentiment_transformer: Optional[SentimentTransformer] = None

    def save_cache(self) -> None:
        atomic_save(self.cache, self.cache_file)

    @retry((DataFetchError, Exception), tries=3, delay=2, backoff=2)
    async def async_fetch_stock_data(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        """Fetch historical stock data asynchronously, using cache if available."""
        if ticker in self.cache:
            logger.info(f"Loading cached data for ticker: {ticker}")
            df = pd.read_json(self.cache[ticker], convert_dates=True)
        else:
            logger.info(f"Fetching data from yfinance for ticker: {ticker}")
            try:
                df = await asyncio.to_thread(lambda: yf.Ticker(ticker).history(period=period))
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = ['_'.join(col).strip() for col in df.columns.values]
                self.cache[ticker] = df.to_json(date_format='iso')
                self.save_cache()
            except Exception as e:
                logger.error(f"Error fetching data for {ticker}: {e}")
                raise DataFetchError(e)
        return df

    @async_retry((NetworkError, Exception), tries=3, delay=2, backoff=2)
    async def fetch_news_sentiment(self, ticker: str) -> float:
        """Fetch news articles for a ticker and calculate an aggregated sentiment score."""
        import aiohttp
        from bs4 import BeautifulSoup
        urls: List[str] = [
            f"https://www.businesswire.com/portal/site/home/?search={ticker}",
            f"https://www.reuters.com/search/news?blob={ticker}",
            f"https://www.bloomberg.com/search?query={ticker}"
        ]
        sentiment_scores: List[float] = []
        if self._sentiment_transformer is None:
            self._sentiment_transformer = SentimentTransformer()
        async with aiohttp.ClientSession() as session:
            async def fetch(url: str) -> None:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            text = await response.text()
                            soup = BeautifulSoup(text, 'html.parser')
                            texts = " ".join(p.get_text() for p in soup.find_all('p'))
                            score = self._sentiment_transformer.analyze(texts)
                            sentiment_scores.append(score)
                except Exception as ex:
                    logger.error(f"Error fetching news from {url}: {ex}")
            await asyncio.gather(*(fetch(url) for url in urls))
        return np.mean(sentiment_scores) if sentiment_scores else 0.0

# -----------------------------
# Option Greeks and Pricing
# -----------------------------
def calculate_option_greeks(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> Tuple[float, float, float, float]:
    """
    Calculate option Greeks: Delta, Gamma, Theta, and Vega.
    """
    d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T) + 1e-12)
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    n = lambda x: math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    if option_type.lower() == "call":
        delta = N(d1)
        theta = (-S * n(d1) * sigma / (2 * math.sqrt(T)) 
                 - r * K * math.exp(-r * T) * N(d2))
    else:
        delta = N(d1) - 1
        theta = (-S * n(d1) * sigma / (2 * math.sqrt(T)) 
                 + r * K * math.exp(-r * T) * N(-d2))
    gamma = n(d1) / (S * sigma * math.sqrt(T) + 1e-12)
    vega = S * math.sqrt(T) * n(d1)
    return delta, gamma, theta, vega

def get_benchmark_returns(benchmark_ticker: str = "^GSPC", start: Optional[str] = None, end: Optional[str] = None) -> np.ndarray:
    """
    Fetch benchmark returns (e.g., S&P 500) over a specified date range.
    """
    try:
        benchmark = yf.Ticker(benchmark_ticker)
        df_bench = benchmark.history(start=start, end=end).dropna()
        returns = np.log(df_bench["Close"] / df_bench["Close"].shift(1)).dropna().values
        return returns
    except Exception as e:
        logger.error(f"Error fetching benchmark returns: {e}")
        return np.array([])

def risk_adjusted_metrics(returns: np.ndarray, benchmark: Optional[np.ndarray] = None, rf: float = 0.01) -> Dict[str, float]:
    """
    Compute various risk-adjusted metrics including Sharpe, Sortino, Treynor, Calmar ratios and Beta.
    """
    std = np.std(returns)
    sharpe = np.mean(returns - rf) / (std + 1e-6) if std > 0 else 0.0
    downside = np.std([r for r in returns if r < rf])
    sortino = np.mean(returns - rf) / (downside + 1e-6) if downside > 0 else 0.0
    if benchmark is not None and benchmark.size > 0:
        cov = np.cov(returns, benchmark)[0, 1]
        var_bench = np.var(benchmark)
        beta = cov / (var_bench + 1e-6)
    else:
        beta = 1.0
    treynor = np.mean(returns - rf) / (beta + 1e-6)
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    calmar = np.mean(returns - rf) / (max_drawdown + 1e-6)
    return {"Sharpe": sharpe, "Sortino": sortino, "Treynor": treynor, "Calmar": calmar, "Beta": beta}

class OptionPricing:
    """
    Option pricing models including Black–Scholes, Binomial, Monte Carlo, and Jump Diffusion.
    """
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        d1 = (math.log(S / K) + (r + sigma**2 / 2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        if option_type.lower() == 'call':
            return S * N(d1) - K * math.exp(-r * T) * N(d2)
        return K * math.exp(-r * T) * N(-d2)

    @staticmethod
    def binomial(S: float, K: float, T: float, r: float, sigma: float, N: int = 100, option_type: str = 'call') -> float:
        dt = T / N
        u = math.exp(sigma * math.sqrt(dt))
        d = 1 / u
        p = (math.exp(r * dt) - d) / (u - d)
        asset_prices = np.array([S * (u**j) * (d**(N - j)) for j in range(N + 1)])
        option_values = np.maximum(asset_prices - K, 0) if option_type.lower() == 'call' else np.maximum(K - asset_prices, 0)
        for i in range(N, 0, -1):
            option_values = math.exp(-r * dt) * (p * option_values[1:i + 1] + (1 - p) * option_values[0:i])
        return option_values[0]

    @staticmethod
    def monte_carlo(S: float, K: float, T: float, r: float, sigma: float, simulations: int = 10000, option_type: str = 'call') -> float:
        dt = T
        rand = np.random.standard_normal(simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rand)
        payoffs = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)
        return math.exp(-r * T) * np.mean(payoffs)

    @staticmethod
    def jump_diffusion(S: float, K: float, T: float, r: float, sigma: float, lam: float = 0.1,
                       muJ: float = 0, sigmaJ: float = 0.1, option_type: str = 'call') -> float:
        price = 0.0
        for k in range(50):
            poisson_prob = math.exp(-lam * T) * (lam * T) ** k / math.factorial(k)
            sigma_k = math.sqrt(sigma**2 + k * (sigmaJ**2) / T)
            price += poisson_prob * OptionPricing.black_scholes(S, K, T, r, sigma_k, option_type)
        return price

# -----------------------------
# Technical Indicators
# -----------------------------
class TechnicalIndicators:
    """Collection of technical indicator functions."""
    @staticmethod
    def WMA(series: pd.Series, period: int) -> pd.Series:
        weights = np.arange(1, period + 1)
        return series.rolling(period).apply(lambda prices: np.dot(prices, weights) / weights.sum(), raw=True)

    @staticmethod
    def DEMA(series: pd.Series, period: int) -> pd.Series:
        ema = series.ewm(span=period, adjust=False).mean()
        return 2 * ema - ema.ewm(span=period, adjust=False).mean()

    @staticmethod
    def TEMA(series: pd.Series, period: int) -> pd.Series:
        ema = series.ewm(span=period, adjust=False).mean()
        ema_ema = ema.ewm(span=period, adjust=False).mean()
        ema_ema_ema = ema_ema.ewm(span=period, adjust=False).mean()
        return 3 * (ema - ema_ema) + ema_ema_ema

    @staticmethod
    def SSMA(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).mean().ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def VWMA(series: pd.Series, volume: pd.Series, period: int) -> pd.Series:
        return (series * volume).rolling(period).sum() / volume.rolling(period).sum()

    @staticmethod
    def HMA(series: pd.Series, period: int) -> pd.Series:
        half = int(period / 2)
        sqrt = int(np.sqrt(period))
        wma_half = TechnicalIndicators.WMA(series, half)
        wma_full = TechnicalIndicators.WMA(series, period)
        diff = 2 * wma_half - wma_full
        return TechnicalIndicators.WMA(diff, sqrt)

    @staticmethod
    def KAMA(series: pd.Series, period: int, fast: int = 2, slow: int = 30) -> pd.Series:
        change = abs(series.diff(period))
        volatility = series.diff().abs().rolling(period).sum()
        er = change / volatility.replace(0, np.nan)
        sc = ((er * (fast - slow)) + slow) ** 2
        kama = [series.iloc[0]]
        for i in range(1, len(series)):
            kama.append(kama[-1] + sc.iloc[i] * (series.iloc[i] - kama[-1]))
        return pd.Series(kama, index=series.index)

    @staticmethod
    def ALMA(series: pd.Series, window: int, offset: float = 0.85, sigma: float = 6) -> pd.Series:
        m = offset * (window - 1)
        s = window / sigma
        weights = np.array([np.exp(-((i - m) ** 2) / (2 * s * s)) for i in range(window)])
        weights /= weights.sum()
        return series.rolling(window).apply(lambda x: np.dot(x, weights), raw=True)

    @staticmethod
    def GMA(series: pd.Series, period: int) -> pd.Series:
        return series.rolling(period).apply(lambda x: np.exp(np.mean(np.log(x[x > 0]))), raw=True)

    @staticmethod
    def MACD(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        ema_fast = series.ewm(span=fast, adjust=False).mean()
        ema_slow = series.ewm(span=slow, adjust=False).mean()
        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        return macd_line, signal_line, macd_line - signal_line

    @staticmethod
    def VWAP(df: pd.DataFrame) -> pd.Series:
        pv = df['Close'] * df['Volume']
        return pv.cumsum() / df['Volume'].cumsum()

    @staticmethod
    def stochastic_oscillator(df: pd.DataFrame, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        low_min = df['Low'].rolling(k_period).min()
        high_max = df['High'].rolling(k_period).max()
        k = 100 * (df['Close'] - low_min) / (high_max - low_min)
        d = k.rolling(d_period).mean()
        return k, d

    @staticmethod
    def ADX(df: pd.DataFrame, period: int = 14) -> pd.Series:
        df = df.copy()
        df['TR'] = np.maximum.reduce([df['High'] - df['Low'],
                                      abs(df['High'] - df['Close'].shift()),
                                      abs(df['Low'] - df['Close'].shift())])
        df['+DM'] = np.where((df['High'] - df['High'].shift()) > (df['Low'].shift() - df['Low']),
                             np.maximum(df['High'] - df['High'].shift(), 0), 0)
        df['-DM'] = np.where((df['Low'].shift() - df['Low']) > (df['High'] - df['High'].shift()),
                             np.maximum(df['Low'].shift() - df['Low'], 0), 0)
        tr14 = df['TR'].rolling(period).sum()
        plus_dm14 = df['+DM'].rolling(period).sum()
        minus_dm14 = df['-DM'].rolling(period).sum()
        plus_di = 100 * (plus_dm14 / tr14)
        minus_di = 100 * (minus_dm14 / tr14)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-12)
        return dx.rolling(period).mean()

    @staticmethod
    def CCI(df: pd.DataFrame, period: int = 20) -> pd.Series:
        tp = (df['High'] + df['Low'] + df['Close']) / 3
        ma = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.mean(np.abs(x - np.mean(x))), raw=True)
        return (tp - ma) / (0.015 * md)

    @staticmethod
    def ROC(series: pd.Series, period: int = 12) -> pd.Series:
        return series.diff(period) / series.shift(period) * 100

    @staticmethod
    def Momentum(series: pd.Series, period: int = 10) -> pd.Series:
        return series.diff(period)

    @staticmethod
    def williams_r(df: pd.DataFrame, period: int = 14) -> pd.Series:
        high_max = df['High'].rolling(period).max()
        low_min = df['Low'].rolling(period).min()
        return -100 * (high_max - df['Close']) / (high_max - low_min)

    @staticmethod
    def chaikin_oscillator(df: pd.DataFrame, short_period: int = 3, long_period: int = 10) -> pd.Series:
        ad = ((2 * df['Close'] - df['Low'] - df['High']) / (df['High'] - df['Low'])).fillna(0) * df['Volume']
        mfm = ((df['Close'] - df['Low']) - (df['High'] - df['Close'])) / (df['High'] - df['Low']).replace(0, np.nan)
        adl = (mfm * df['Volume']).cumsum()
        return adl.rolling(short_period).mean() - adl.rolling(long_period).mean()

    @staticmethod
    def OBV(df: pd.DataFrame) -> pd.Series:
        direction = np.where(df['Close'].diff() >= 0, 1, -1)
        return (direction * df['Volume']).cumsum()

# -----------------------------
# Hidden Markov Model (HMM)
# -----------------------------
class HiddenMarkovModel:
    """
    A Hidden Markov Model for identifying market regimes.
    """
    def __init__(self, n_states: int = 2, seed: Optional[int] = None) -> None:
        self.n_states: int = n_states
        if seed is not None:
            np.random.seed(seed)
        self.trans_mat: np.ndarray = np.full((n_states, n_states), 1.0 / n_states)
        self.means: np.ndarray = np.random.randn(n_states)
        self.vars: np.ndarray = np.ones(n_states)
        self.pi: np.ndarray = np.full(n_states, 1.0 / n_states)

    def _emission_probs(self, observations: np.ndarray) -> np.ndarray:
        T: int = observations.shape[0]
        obs = observations.reshape(-1, 1)
        coef = 1.0 / np.sqrt(2 * math.pi * self.vars)
        exponents = -((obs - self.means) ** 2) / (2 * self.vars)
        return coef * np.exp(exponents)

    @profile
    def _fit_single(self, observations: np.ndarray, n_iter: int = 10) -> float:
        T: int = observations.shape[0]
        log_likelihood: float = -np.inf
        for iteration in range(n_iter):
            E = self._emission_probs(observations)
            alpha = np.zeros((T, self.n_states))
            scale = np.zeros(T)
            alpha[0] = self.pi * E[0]
            scale[0] = alpha[0].sum()
            alpha[0] /= (scale[0] + 1e-12)
            for t in range(1, T):
                alpha[t] = E[t] * np.dot(alpha[t - 1], self.trans_mat)
                scale[t] = alpha[t].sum()
                alpha[t] /= (scale[t] + 1e-12)
            beta = np.zeros((T, self.n_states))
            beta[T - 1] = np.ones(self.n_states) / (scale[T - 1] + 1e-12)
            for t in range(T - 2, -1, -1):
                beta[t] = np.dot(self.trans_mat, (E[t + 1] * beta[t + 1]))
                beta[t] /= (scale[t] + 1e-12)
            gamma = alpha * beta
            gamma /= (gamma.sum(axis=1, keepdims=True) + 1e-12)
            num = alpha[:-1, :, None] * self.trans_mat[None, :, :] * E[1:, None, :] * beta[1:, None, :]
            denom = num.sum(axis=(1, 2), keepdims=True) + 1e-12
            xi = num / denom
            self.trans_mat = xi.sum(axis=0) / (xi.sum(axis=(0, 2), keepdims=True) + 1e-12)
            self.trans_mat = self.trans_mat.squeeze()
            gamma_sum = gamma.sum(axis=0)
            self.means = (gamma * observations.reshape(-1, 1)).sum(axis=0) / (gamma_sum + 1e-12)
            self.vars = (gamma * (observations.reshape(-1, 1) - self.means) ** 2).sum(axis=0) / (gamma_sum + 1e-12)
            self.pi = gamma[0]
            log_likelihood = np.sum(np.log(scale + 1e-12))
            logger.debug(f"HMM Iteration {iteration+1}/{n_iter}, Log Likelihood: {log_likelihood:.4f}")
        return log_likelihood

    def fit(self, observations: np.ndarray, n_iter: int = 10, parallel: bool = False, n_init: int = 1) -> float:
        """
        Fit the HMM to observations. Use parallel fits if specified.
        """
        if parallel and n_init > 1:
            best_model = fit_parallel_hmm(observations, self.n_states, n_iter, n_init)
            self.trans_mat = best_model.trans_mat
            self.means = best_model.means
            self.vars = best_model.vars
            self.pi = best_model.pi
            log_likelihood = np.sum(np.log(self._emission_probs(observations).sum(axis=1) + 1e-12))
            logger.info(f"Selected best HMM from parallel fits with log likelihood: {log_likelihood:.4f}")
            return log_likelihood
        return self._fit_single(observations, n_iter)

    def predict(self, observations: np.ndarray) -> np.ndarray:
        """
        Predict the hidden state sequence for the given observations.
        """
        T: int = observations.shape[0]
        N: int = self.n_states
        E = self._emission_probs(observations)
        delta = np.zeros((T, N))
        psi = np.zeros((T, N), dtype=int)
        delta[0] = np.log(self.pi + 1e-12) + np.log(E[0] + 1e-12)
        for t in range(1, T):
            for j in range(N):
                temp = delta[t - 1] + np.log(self.trans_mat[:, j] + 1e-12)
                psi[t, j] = np.argmax(temp)
                delta[t, j] = np.max(temp) + np.log(E[t, j] + 1e-12)
        states = np.zeros(T, dtype=int)
        states[T - 1] = np.argmax(delta[T - 1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

def _fit_instance_hmm(observations: np.ndarray, n_states: int, n_iter: int, seed: int) -> dict:
    model = HiddenMarkovModel(n_states=n_states, seed=seed)
    ll = model._fit_single(observations, n_iter)
    return {
        'log_likelihood': ll,
        'trans_mat': model.trans_mat,
        'means': model.means,
        'vars': model.vars,
        'pi': model.pi
    }

def fit_parallel_hmm(observations: np.ndarray, n_states: int, n_iter: int = 10, n_init: int = 4) -> HiddenMarkovModel:
    best_result = None
    best_ll = -np.inf
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(_fit_instance_hmm, observations, n_states, n_iter, seed)
                   for seed in range(n_init)]
        for future in concurrent.futures.as_completed(futures):
            res = future.result()
            if res['log_likelihood'] > best_ll:
                best_ll = res['log_likelihood']
                best_result = res
    best_model = HiddenMarkovModel(n_states=n_states)
    best_model.trans_mat = best_result['trans_mat']
    best_model.means = best_result['means']
    best_model.vars = best_result['vars']
    best_model.pi = best_result['pi']
    logger.info(f"Best log likelihood from parallel HMM fits: {best_ll:.4f}")
    return best_model

# -----------------------------
# Feature Extraction and Reduction
# -----------------------------
def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extract key features and technical indicators from the input DataFrame."""
    features = pd.DataFrame(index=df.index)
    features["Close"] = df["Close"]
    features["WMA"] = TechnicalIndicators.WMA(df["Close"], 20)
    features["DEMA"] = TechnicalIndicators.DEMA(df["Close"], 20)
    features["TEMA"] = TechnicalIndicators.TEMA(df["Close"], 20)
    features["SSMA"] = TechnicalIndicators.SSMA(df["Close"], 20)
    features["VWMA"] = TechnicalIndicators.VWMA(df["Close"], df["Volume"], 20)
    features["HMA"] = TechnicalIndicators.HMA(df["Close"], 20)
    features["KAMA"] = TechnicalIndicators.KAMA(df["Close"], 20)
    features["ALMA"] = TechnicalIndicators.ALMA(df["Close"], 20)
    features["GMA"] = TechnicalIndicators.GMA(df["Close"], 20)
    macd_line, _, _ = TechnicalIndicators.MACD(df["Close"])
    features["MACD"] = macd_line
    features["VWAP"] = TechnicalIndicators.VWAP(df)
    k, _ = TechnicalIndicators.stochastic_oscillator(df)
    features["Stochastic_K"] = k
    features["ADX"] = TechnicalIndicators.ADX(df)
    features["CCI"] = TechnicalIndicators.CCI(df)
    features["ROC"] = TechnicalIndicators.ROC(df["Close"])
    features["Momentum"] = TechnicalIndicators.Momentum(df["Close"])
    features["WilliamsR"] = TechnicalIndicators.williams_r(df)
    features["Chaikin"] = TechnicalIndicators.chaikin_oscillator(df)
    features["OBV"] = TechnicalIndicators.OBV(df)
    return features.fillna(method="bfill").fillna(method="ffill")

def augment_features(features: pd.DataFrame) -> pd.DataFrame:
    """Augment features by adding slight Gaussian noise."""
    return features + np.random.normal(0, 0.001, size=features.shape)

def pca_reduce(features: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Reduce feature dimensions using SVD-based PCA."""
    X = features.values
    U, s, _ = svd(X - np.mean(X, axis=0), full_matrices=False)
    X_reduced = U[:, :n_components] * s[:n_components]
    return pd.DataFrame(X_reduced, index=features.index)

# -----------------------------
# Neural Network with JAX
# -----------------------------
class NeuralNetworkJAX:
    """
    A neural network implementation using JAX with support for dropout and various activations.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 hidden_layers: Optional[List[int]] = None,
                 activation: str = "sin",
                 learning_rate: float = 1e-3,
                 dropout_rate: float = 0.0,
                 regularization: float = 0.0,
                 version: str = "v1.0") -> None:
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.hidden_layers: List[int] = hidden_layers
        self.activation_name: str = activation
        self.learning_rate: float = learning_rate
        self.dropout_rate: float = dropout_rate
        self.regularization: float = regularization
        self.version: str = version
        self.loss_history: List[float] = []
        self.activation_fn: Callable[[jnp.ndarray], jnp.ndarray] = {
            "sin": jnp.sin,
            "relu": jax.nn.relu,
            "tanh": jnp.tanh,
            "sigmoid": jax.nn.sigmoid,
            "leaky_relu": lambda x: jax.nn.leaky_relu(x, negative_slope=0.01),
            "elu": jax.nn.elu,
            "gelu": jax.nn.gelu
        }.get(activation, jnp.sin)
        layers = [input_dim] + hidden_layers + [output_dim]
        self.params: Dict[str, jnp.ndarray] = {
            f"W{i+1}": jnp.array(np.random.randn(layers[i], layers[i+1]) * 0.1)
            for i in range(len(layers) - 1)
        }
        self.params.update({
            f"b{i+1}": jnp.array(np.zeros(layers[i+1]))
            for i in range(len(layers) - 1)
        })

    def forward(self, X: jnp.ndarray, params: Optional[Dict[str, jnp.ndarray]] = None,
                dropout_key: Optional[jnp.ndarray] = None, is_training: bool = True) -> jnp.ndarray:
        """
        Forward pass through the network.
        """
        if params is None:
            params = self.params
        a = X
        num_layers = len(self.hidden_layers) + 1
        key = dropout_key
        for i in range(1, num_layers + 1):
            z = jnp.dot(a, params[f"W{i}"]) + params[f"b{i}"]
            if i < num_layers:
                a = self.activation_fn(z)
                if is_training and self.dropout_rate > 0.0:
                    key = key or random.PRNGKey(int(time.time()))
                    key, subkey = random.split(key)
                    mask = random.bernoulli(subkey, p=1.0 - self.dropout_rate, shape=a.shape)
                    a = a * mask / (1.0 - self.dropout_rate)
            else:
                a = z
        return a

    def loss(self, params: Dict[str, jnp.ndarray], X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
        """
        Mean squared error loss with L2 regularization.
        """
        pred = self.forward(X, params, is_training=True)
        mse = jnp.mean((pred - y) ** 2)
        l2_reg = sum(jnp.sum(jnp.square(p)) for p in params.values())
        return mse + self.regularization * l2_reg

    @profile
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, early_stopping: int = 10) -> List[float]:
        """Train the network and plot the loss history."""
        X_jax = jnp.array(X)
        y_jax = jnp.array(y)
        loss_grad = grad(self.loss)
        best_loss = float("inf")
        early_stop_counter = 0
        params = self.params

        @jit
        def update_params(params, X_jax, y_jax, key):
            l = self.loss(params, X_jax, y_jax)
            grads = loss_grad(params, X_jax, y_jax)
            new_params = {k: params[k] - self.learning_rate * grads[k] for k in params}
            return new_params, l

        key = random.PRNGKey(int(time.time()))
        for epoch in range(epochs):
            key, subkey = random.split(key)
            params, l = update_params(params, X_jax, y_jax, subkey)
            self.loss_history.append(float(l))
            if l < best_loss:
                best_loss = l
                early_stop_counter = 0
            else:
                early_stop_counter += 1
                if early_stop_counter >= early_stopping:
                    logger.info(f"[JAX] Early stopping triggered at epoch {epoch+1}.")
                    break
        self.params = params
        import matplotlib.pyplot as plt
        plt.figure(figsize=(6, 4))
        plt.plot(self.loss_history, label="JAX Training Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Neural Network (JAX) Training Loss")
        plt.legend()
        plt.savefig("loss_curve_jax.png")
        plt.close()
        return self.loss_history

    def predict(self, X: np.ndarray) -> jnp.ndarray:
        """Predict using the trained network."""
        X_jax = jnp.array(X)
        return self.forward(X_jax, is_training=False)

# -----------------------------
# Heston Simulation using JAX
# -----------------------------
@partial(jit, static_argnames=['N', 'M'])
def heston_model_sim_jax(S0: float, v0: float, rho: float, kappa: float, theta: float,
                         sigma: float, r: float, T: float, N: int, M: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Simulate asset prices and variances using the Heston model.
    """
    dt = T / N
    mu = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, rho], [rho, 1.0]])
    S = jnp.full((N+1, M), S0)
    v = jnp.full((N+1, M), v0)
    key = random.PRNGKey(0)
    Z = random.multivariate_normal(key, mean=mu, cov=cov, shape=(N, M))
    
    def body(i, carry):
        S, v = carry
        S_prev = S[i-1]
        v_prev = v[i-1]
        dS = (r - 0.5 * v_prev) * dt + jnp.sqrt(v_prev * dt) * Z[i-1, :, 0]
        new_S = S_prev * jnp.exp(dS)
        dv = kappa * (theta - v_prev) * dt + sigma * jnp.sqrt(v_prev * dt) * Z[i-1, :, 1]
        new_v = jnp.maximum(v_prev + dv, 0.0)
        S = S.at[i].set(new_S)
        v = v.at[i].set(new_v)
        return (S, v)

    S, v = lax.fori_loop(1, N+1, body, (S, v))
    return S, v

# -----------------------------
# Dashboard for Real-Time Monitoring
# -----------------------------
def start_dashboard(metrics_data: Dict[str, Any]) -> None:
    """Launch a Dash dashboard to monitor live performance metrics."""
    app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        dbc.Row([dbc.Col(html.H2("Live Performance Dashboard"), width=12)]),
        dbc.Row([
            dbc.Col(dbc.Card([
                dbc.CardHeader("Average Black–Scholes Price"),
                dbc.CardBody(html.H4(id="avg-bs", children=str(metrics_data.get("avg_bs", "N/A"))))
            ]), width=6),
            dbc.Col(dbc.Card([
                dbc.CardHeader("Average Sentiment"),
                dbc.CardBody(html.H4(id="avg-sent", children=str(metrics_data.get("avg_sent", "N/A"))))
            ]), width=6)
        ]),
        dcc.Interval(id="interval", interval=10*1000, n_intervals=0)
    ], fluid=True)

    @app.callback(
        [Output("avg-bs", "children"), Output("avg-sent", "children")],
        [Input("interval", "n_intervals")]
    )
    def update_metrics(n):
        return (str(metrics_data.get("avg_bs", "N/A")), str(metrics_data.get("avg_sent", "N/A")))
    
    threading.Thread(target=app.run_server, kwargs={'port':8050}, daemon=True).start()
    logger.info("Dashboard started at http://127.0.0.1:8050")

# -----------------------------
# Optimization Functions
# -----------------------------
def advanced_bayesian_optimization_lr(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 25) -> Tuple[float, Any]:
    """
    Use Bayesian Optimization to select the best learning rate.
    """
    def objective_lr(lr: float) -> float:
        nn = NeuralNetworkJAX(input_dim=X_train.shape[1], output_dim=1, learning_rate=lr)
        loss_history = nn.train(X_train, y_train, epochs=10, early_stopping=3)
        return -loss_history[-1]
    optimizer = BayesianOptimization(
        f=objective_lr,
        pbounds={'lr': (1e-5, 1e-1)},
        random_state=42,
        verbose=0
    )
    optimizer.maximize(init_points=5, n_iter=n_iter)
    best_lr = optimizer.max['params']['lr']
    logger.info(f"Advanced Bayesian Optimization selected learning_rate = {best_lr}")
    return best_lr, optimizer

def walk_forward_optimization(features: pd.DataFrame, targets: pd.Series, window: int = 100) -> List[Tuple[Any, float]]:
    """
    Walk-forward optimization: train the model on rolling windows and generate predictions.
    """
    predictions: List[Tuple[Any, float]] = []
    for start in range(0, len(features) - window, window):
        X_train = features.iloc[start:start + window].values
        y_train = targets.iloc[start:start + window].values.reshape(-1, 1)
        best_lr, _ = advanced_bayesian_optimization_lr(X_train, y_train, n_iter=5)
        nn_jax = NeuralNetworkJAX(input_dim=X_train.shape[1], output_dim=1, learning_rate=best_lr)
        nn_jax.train(X_train, y_train, epochs=50, early_stopping=5)
        X_pred = features.iloc[start + window:start + window + 1].values
        pred = nn_jax.predict(X_pred)
        predictions.append((features.index[start + window], float(pred[0])))
    return predictions

# -----------------------------
# Ticker Processing Functions
# -----------------------------
def process_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    """
    Process a single ticker: fetch data, extract features, run predictions, price options, and compute risk metrics.
    """
    try:
        data_fetcher = DataFetcher()
        df = asyncio.run(data_fetcher.async_fetch_stock_data(ticker))
        target = df["Close"].shift(-4) / df["Close"] - 1
        target = target.fillna(0)
        features = extract_features(df)
        augmented = augment_features(features)
        reduced = pca_reduce(augmented, n_components=10)
        predictions = walk_forward_optimization(reduced, target)
        latest_close = df["Close"].iloc[-1]
        strike = latest_close
        r = 0.01
        T = 4 / 252
        sigma = np.std(np.log(df["Close"] / df["Close"].shift(1)).dropna()) * math.sqrt(252)
        bs_price = OptionPricing.black_scholes(latest_close, strike, T, r, sigma, option_type="call")
        binom_price = OptionPricing.binomial(latest_close, strike, T, r, sigma, N=100, option_type="call")
        mc_price = OptionPricing.monte_carlo(latest_close, strike, T, r, sigma, simulations=10000, option_type="call")
        delta, gamma, theta, vega = calculate_option_greeks(latest_close, strike, T, r, sigma)
        sentiment = asyncio.run(data_fetcher.fetch_news_sentiment(ticker))
        returns = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
        benchmark_returns = get_benchmark_returns(start=str(df.index[0].date()), end=str(df.index[-1].date()))
        risk_metrics = {"RiskMetrics": risk_adjusted_metrics(returns, benchmark=benchmark_returns)}
        hmm = HiddenMarkovModel(n_states=2)
        hmm.fit(returns, n_iter=10, parallel=True, n_init=4)
        regimes = hmm.predict(returns)
        regime = regimes[-1]
        result = {
            "Ticker": ticker,
            "LatestClose": latest_close,
            "Strike": strike,
            "BS_Price": bs_price,
            "Binom_Price": binom_price,
            "MC_Price": mc_price,
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Sentiment": sentiment,
            "CurrentRegime": regime,
            "RiskMetrics": risk_metrics,
            "Predictions": predictions
        }
        return result
    except Exception as e:
        logger.error(f"Error processing ticker {ticker}: {e}")
        return None

def parallel_ticker_processing(tickers: List[str]) -> List[Dict[str, Any]]:
    """Process multiple tickers in parallel."""
    with mp.Pool(processes=min(len(tickers), mp.cpu_count())) as pool:
        results = pool.map(process_ticker, tickers)
    return [res for res in results if res is not None]

def gpu_vector_test() -> Optional[np.ndarray]:
    """Test the GPU vector multiplication routine."""
    try:
        gpu_accel = GPUAccelerator()
        a = np.random.rand(1000000)
        b = np.random.rand(1000000)
        result = gpu_accel.vector_multiply(a, b)
        expected = a * b
        if not np.allclose(result, expected, atol=1e-5):
            raise ValueError("GPU vector multiplication does not match expected result.")
        logger.info("GPU vector multiply test completed successfully.")
        return result
    except Exception as e:
        logger.error(f"GPU test failed: {e}")
        return None

def build_prediction_table(results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
    """
    Build a table of predictions for tickers with significant predicted call option increases.
    """
    rows: List[Dict[str, Any]] = []
    for res in results:
        if res.get("Predictions"):
            pred_date, pred_change = res["Predictions"][-1]
            if pred_change >= 10.0:
                rows.append({
                    "Date": pred_date.strftime("%Y-%m-%d") if isinstance(pred_date, datetime.datetime) else str(pred_date),
                    "Ticker": res["Ticker"],
                    "Strike": res["Strike"],
                    "Predicted Increase (%)": pred_change * 100,
                    "BS Price": res["BS_Price"],
                    "Sentiment": res["Sentiment"]
                })
    return pd.DataFrame(rows) if rows else None

def plot_performance_metrics(results: List[Dict[str, Any]]) -> None:
    """
    Plot scatter metrics of news sentiment vs. Black–Scholes price.
    """
    tickers = [res["Ticker"] for res in results if res is not None]
    sentiments = [res["Sentiment"] for res in results if res is not None]
    bs_prices = [res["BS_Price"] for res in results if res is not None]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(sentiments, bs_prices, c="blue")
    plt.xlabel("News Sentiment")
    plt.ylabel("Black–Scholes Price")
    plt.title("Performance Metrics per Ticker")
    plt.grid(True)
    plt.savefig("performance_metrics.png")
    plt.show()

# -----------------------------
# Main Execution
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit")
    args = parser.parse_args()
    if args.test:
        run_tests()
        return
    now = datetime.datetime.now()
    if now.weekday() != RUN_DAY or now.hour < RUN_HOUR:
        logger.error("This script is designed to run at Monday close (after 16:00). Exiting.")
        sys.exit(1)
    try:
        tickers_input = input("Enter tickers (comma separated): ")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        if not tickers:
            logger.error("No tickers provided. Exiting.")
            sys.exit(1)
    except Exception as e:
        logger.error(f"Input error: {e}")
        sys.exit(1)
    try:
        gpu_vector_test()
    except Exception as e:
        logger.error(f"GPU acceleration test failed: {e}")
    results = parallel_ticker_processing(tickers)
    if not results:
        logger.info("No predictions generated due to data or market conditions.")
        return
    prediction_table = build_prediction_table(results)
    if prediction_table is not None and not prediction_table.empty:
        print("\nPrediction Table:")
        print(prediction_table.to_string(index=False))
    else:
        print("No call option price increase predictions meeting the threshold were found.")
    plot_performance_metrics(results)
    avg_bs = np.mean([res["BS_Price"] for res in results if "BS_Price" in res])
    avg_sent = np.mean([res["Sentiment"] for res in results if "Sentiment" in res])
    metrics_data = {"avg_bs": avg_bs, "avg_sent": avg_sent}
    start_dashboard(metrics_data)
    logger.info("Processing complete. Dashboard is live; press Ctrl+C to exit.")
    try:
        while True:
            time.sleep(10)
    except KeyboardInterrupt:
        sys.exit(0)

# -----------------------------
# Unit and Integration Tests
# -----------------------------
def run_tests() -> None:
    import matplotlib.pyplot as plt
    logger.info("Running unit tests...")
    # Numba CPU test
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    cpu_result = fast_cpu_distance(a, b)
    assert abs(cpu_result) < 1e-6, "CPU kernel (Numba) test failed."
    # GPU test
    gpu = GPUAccelerator()
    res_gpu = gpu.vector_multiply(a, b)
    expected = a * b
    assert np.allclose(res_gpu, expected, atol=1e-5), "GPU routine test failed."
    # JAX neural network test
    X_test = np.random.rand(5, 10)
    y_test = np.random.rand(5, 1)
    nn_jax = NeuralNetworkJAX(input_dim=10, output_dim=1, dropout_rate=0.2, activation="relu")
    l_val = nn_jax.loss(nn_jax.params, jnp.array(X_test), jnp.array(y_test))
    g = grad(nn_jax.loss)(nn_jax.params, jnp.array(X_test), jnp.array(y_test))
    for key in nn_jax.params:
        assert key in g, f"Missing gradient for {key}"
    # Sentiment transformer test
    st = SentimentTransformer()
    score = st.analyze("This is a great day for trading!")
    assert isinstance(score, float), "Sentiment transformer test failed."
    # Feature extraction test
    df_sample = pd.DataFrame({
        "Close": np.linspace(100, 110, 30),
        "Volume": np.random.randint(1000, 5000, 30),
        "High": np.linspace(101, 111, 30),
        "Low": np.linspace(99, 109, 30)
    }, index=pd.date_range("2020-01-01", periods=30))
    feats = extract_features(df_sample)
    assert not feats.empty, "Feature extraction test failed."
    # Retry decorator test
    call_counter = {"count": 0}
    @retry(Exception, tries=3, delay=0.1, backoff=1)
    def test_retry_success():
        call_counter["count"] += 1
        if call_counter["count"] < 3:
            raise ValueError("Failing")
        return "succeeded"
    assert test_retry_success() == "succeeded", "Retry decorator test failed."
    # Profiling decorator test
    @profile
    def dummy_sleep():
        time.sleep(0.2)
        return "done"
    result = dummy_sleep()
    assert result == "done", "Profiling decorator test failed."
    # Integration test for ticker processing
    sample_ticker = "AAPL"
    result = process_ticker(sample_ticker)
    assert result is not None, "Integration test for process_ticker failed."
    logger.info("All unit and integration tests passed.")

if __name__ == "__main__":
    main()
