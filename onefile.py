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
import io
from functools import wraps, partial
from typing import Any, Dict, Tuple, List, Optional, Callable

import concurrent.futures
import multiprocessing as mp
import numpy as np
import pandas as pd
import yfinance as yf
from numpy.linalg import svd

# Third-party libraries with strict requirements
try:
    from numba import njit, prange, cuda
except ImportError:
    sys.exit("Please install numba (pip install numba)")
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, grad, lax, random, pmap
except ImportError:
    sys.exit("Please install JAX (pip install jax jaxlib)")
try:
    import optax
except ImportError:
    sys.exit("Please install optax (pip install optax)")
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
try:
    import hdbscan
except ImportError:
    sys.exit("Please install hdbscan (pip install hdbscan)")
try:
    from sklearn.cluster import KMeans  # Kept for other uses if needed.
except ImportError:
    sys.exit("Please install scikit-learn (pip install scikit-learn)")
try:
    import dask
    from dask import delayed, compute
except ImportError:
    sys.exit("Please install dask (pip install dask)")
try:
    import ray
except ImportError:
    sys.exit("Please install ray (pip install ray)")
try:
    from sklearn.mixture import GaussianMixture
except ImportError:
    sys.exit("Please install scikit-learn (pip install scikit-learn)")
try:
    import lightgbm as lgb
except ImportError:
    sys.exit("Please install lightgbm (pip install lightgbm)")
try:
    from catboost import CatBoostRegressor
except ImportError:
    sys.exit("Please install catboost (pip install catboost)")
try:
    from flax import linen as nn  # Flax for NN
    import flax
except ImportError:
    sys.exit("Please install flax (pip install flax)")

# Additional missing import for asynchronous HTTP calls
try:
    import aiohttp
except ImportError:
    sys.exit("Please install aiohttp (pip install aiohttp)")

from logging.handlers import RotatingFileHandler
from joblib import Memory

# -----------------------------
# Global Configurations and Memory Cache
# -----------------------------
CONFIG: Dict[str, Any] = {
    "CACHE_FILE": "data_cache.json",
    "LOG_FILE": "model_log.txt",
    "RUN_DAY": 0,         # Monday
    "RUN_HOUR": 16,       # 16:00 local time
    "CLUSTER_ASSIGNMENTS": {},
    "DASH_PORT": 8050,    # Configurable dashboard port
    "CACHE_EXPIRY": 86400  # Cache expiry time in seconds (1 day)
}

# Setup joblib memory (using a folder "joblib_cache")
memory = Memory(location="./joblib_cache", verbose=0)

# -----------------------------
# Logger Setup with Rotating Handler
# -----------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

# Console handler
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)
ch.setFormatter(formatter)
logger.addHandler(ch)

# Rotating File Handler
fh = RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=1 * 1024 * 1024, backupCount=5)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# -----------------------------
# Custom Exceptions
# -----------------------------
class NetworkError(Exception):
    """Exception for network-related errors."""
    pass

class DataFetchError(Exception):
    """Exception raised when data fetching fails."""
    pass

class GPUInitializationError(Exception):
    """Exception raised when GPU initialization fails."""
    pass

class ModelTrainingError(Exception):
    """Exception raised during model training failures."""
    pass

# -----------------------------
# Decorators
# -----------------------------
def retry(exceptions: Tuple[Exception, ...], tries: int = 3, delay: float = 1.0, backoff: int = 2) -> Callable:
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
                    logger.error(f"{func.__name__} failed with {e}, retrying in {mdelay:.2f} seconds...")
                    time.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return func(*args, **kwargs)
        return wrapper
    return decorator

def async_retry(exceptions: Tuple[Exception, ...], tries: int = 3, delay: float = 1.0, backoff: int = 2) -> Callable:
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
                    logger.error(f"{func.__name__} (async) failed with {e}, retrying in {mdelay:.2f} seconds...")
                    await asyncio.sleep(mdelay)
                    mtries -= 1
                    mdelay *= backoff
            return await func(*args, **kwargs)
        return wrapper
    return decorator

def profile(func: Callable) -> Callable:
    """
    Simple profiling decorator that logs the execution time of the function.
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        logger.info(f"{func.__name__} took {elapsed:.4f} seconds")
        return result
    return wrapper

# -----------------------------
# Utility Functions
# -----------------------------
def atomic_save(cache: Dict[str, Any], filename: str) -> None:
    """
    Atomically save a JSON cache to file.
    """
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, 'w') as f:
            json.dump(cache, f)
        os.replace(temp_filename, filename)
    except Exception as e:
        logger.exception(f"Error saving cache: {e}")

def run_async(coro: Callable) -> Any:
    """
    Run an asynchronous coroutine.
    """
    return asyncio.run(coro)

# -----------------------------
# Accelerated Numerical Kernels (Numba)
# -----------------------------
@njit(fastmath=True, parallel=True)
def fast_cpu_distance(x: np.ndarray, y: np.ndarray) -> float:
    """
    Compute squared Euclidean distance between x and y using loop unrolling.
    """
    diff = x - y
    total = 0.0
    n = diff.shape[0]
    unroll_factor = 4
    limit = n - (n % unroll_factor)
    for i in range(0, limit, unroll_factor):
        total += diff[i] ** 2 + diff[i+1] ** 2 + diff[i+2] ** 2 + diff[i+3] ** 2
    for i in range(limit, n):
        total += diff[i] ** 2
    return total

@njit(fastmath=True)
def binomial_option_unrolled(S: float, K: float, T: float, r: float, sigma: float, N: int, option_type: str) -> float:
    """
    Compute option price using an unrolled binomial tree.
    """
    dt = T / N
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    p = (math.exp(r * dt) - d) / (u - d)
    asset_prices = np.empty(N + 1, dtype=np.float64)
    for j in range(N + 1):
        asset_prices[j] = S * (u ** j) * (d ** (N - j))
    if option_type.lower() == 'call':
        for j in range(N + 1):
            asset_prices[j] = max(asset_prices[j] - K, 0.0)
    else:
        for j in range(N + 1):
            asset_prices[j] = max(K - asset_prices[j], 0.0)
    for i in range(N, 0, -1):
        discount = math.exp(-r * dt)
        total_steps = i
        unroll = 4
        limit = total_steps - (total_steps % unroll)
        j = 0
        while j < limit:
            asset_prices[j] = discount * (p * asset_prices[j + 1] + (1.0 - p) * asset_prices[j])
            asset_prices[j+1] = discount * (p * asset_prices[j+2] + (1.0 - p) * asset_prices[j+1])
            asset_prices[j+2] = discount * (p * asset_prices[j+3] + (1.0 - p) * asset_prices[j+2])
            if (j + 3) < total_steps:
                asset_prices[j+3] = discount * (p * asset_prices[j+4] + (1.0 - p) * asset_prices[j+3])
            else:
                asset_prices[j+3] = discount * asset_prices[j+3]
            j += unroll
        while j < total_steps:
            asset_prices[j] = discount * (p * asset_prices[j+1] + (1.0 - p) * asset_prices[j])
            j += 1
    return asset_prices[0]

# -----------------------------
# GPU Accelerator using Metal (metalcompute)
# -----------------------------
class GPUAccelerator:
    """
    GPU Accelerator using metalcompute. Initializes a GPU device and compiles a vector multiplication kernel.
    """
    def __init__(self, device: Optional[mc.Device] = None) -> None:
        try:
            self.device = device if device is not None else mc.Device()
            logger.info(f"GPUAccelerator initialized with device: {self.device.name}")
        except Exception as e:
            logger.exception("Failed to initialize GPUAccelerator")
            raise GPUInitializationError(e)
        self.kernel_code = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void vec_mult(device const float* a [[ buffer(0) ]],
                             device const float* b [[ buffer(1) ]],
                             device float* result [[ buffer(2) ]],
                             constant uint &total_elements [[ buffer(3) ]],
                             uint id [[ thread_position_in_grid ]],
                             uint total_threads [[ threads_per_grid ]]) {
            for (uint i = id; i < total_elements; i += total_threads) {
                result[i] = a[i] * b[i];
            }
        }
        """
        try:
            self.compiled_kernel = self.device.kernel(self.kernel_code).function("vec_mult")
        except Exception as e:
            logger.exception("Kernel compilation failed")
            raise GPUInitializationError(e)
    
    def vector_multiply(self, a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
        """
        Multiply two vectors using the GPU.
        """
        a_np = np.ascontiguousarray(a_np.astype(np.float32))
        b_np = np.ascontiguousarray(b_np.astype(np.float32))
        total_elements = a_np.shape[0]
        try:
            a_bytes = memoryview(a_np)
            b_bytes = memoryview(b_np)
        except Exception:
            from array import array
            a_bytes = array('f', a_np.tolist())
            b_bytes = array('f', b_np.tolist())
        bytes_count = total_elements * 4
        result_buf = self.device.buffer(bytes_count)
        try:
            total_threads = getattr(self.device, "max_threads", 256)
            self.compiled_kernel(total_elements, a_bytes, b_bytes, result_buf)
        except Exception as e:
            logger.exception("Kernel execution failed")
            raise GPUInitializationError(e)
        result_view = memoryview(result_buf).cast('f')
        return np.array(result_view)

# -----------------------------
# MultiGPU Accelerator
# -----------------------------
class MultiGPUAccelerator:
    """
    Uses multiple GPU devices (if available) to perform vector multiplication in parallel.
    """
    def __init__(self) -> None:
        try:
            self.devices = mc.devices()
            if len(self.devices) < 2:
                logger.warning("Less than 2 GPU devices available; using single GPU mode.")
                self.devices = self.devices[:1]
            else:
                self.devices = self.devices[:2]
            self.gpu_accels = [GPUAccelerator(device=d) for d in self.devices]
            logger.info(f"MultiGPUAccelerator using devices: {[d.name for d in self.devices]}")
        except Exception as e:
            logger.exception("Failed to initialize MultiGPUAccelerator")
            raise GPUInitializationError(e)
    
    def vector_multiply(self, a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
        total_elements = a_np.shape[0]
        num_devices = len(self.gpu_accels)
        indices = np.array_split(np.arange(total_elements), num_devices)
        results = [None] * num_devices

        def worker(idx: int, inds: np.ndarray):
            a_slice = a_np[inds]
            b_slice = b_np[inds]
            results[idx] = (inds, self.gpu_accels[idx].vector_multiply(a_slice, b_slice))

        threads = []
        for idx, inds in enumerate(indices):
            t = threading.Thread(target=worker, args=(idx, inds))
            t.start()
            threads.append(t)
        for t in threads:
            t.join()
        full_result = np.empty(total_elements, dtype=np.float32)
        for inds, res in results:
            full_result[inds] = res
        return full_result

# -----------------------------
# Sentiment Analysis with Enhanced Customization
# -----------------------------
class SentimentTransformer:
    """
    A sentiment analysis wrapper based on FinBERT with additional custom analysis.
    """
    def __init__(self) -> None:
        try:
            # Load the pre-trained FinBERT model; assume that a fine-tuned version exists for domain-specific tasks.
            self.pipeline = pipeline("sentiment-analysis",
                                     model="yiyanghkust/finbert-pretrain",
                                     tokenizer="yiyanghkust/finbert-pretrain")
            logger.info("Loaded FinBERT-based sentiment transformer.")
        except Exception as e:
            logger.exception("Failed to load FinBERT transformer")
            raise e

    def analyze(self, text: str) -> float:
        """
        Analyze sentiment in the text. Positive returns positive score, negative returns negative.
        """
        result = self.pipeline(text[:512])
        if result and result[0]['label'].upper() == "POSITIVE":
            return result[0]['score']
        elif result and result[0]['label'].upper() == "NEGATIVE":
            return -result[0]['score']
        return 0.0

    def analyze_custom(self, text: str) -> float:
        """
        Hybrid approach: combine LLM embeddings (placeholder) with classical NLP sentiment.
        (In practice, one would load a sentence-transformers model and blend the sentiment scores.)
        """
        basic_score = self.analyze(text)
        # Placeholder for LLM embedding based scoring:
        llm_embedding_score = 0.0  # Assume a function computes this score.
        # For demonstration, we blend them:
        return 0.7 * basic_score + 0.3 * llm_embedding_score

    @staticmethod
    def analyze_static(text: str) -> float:
        return SentimentTransformer().analyze(text)

# -----------------------------
# Data Fetching & Caching with Async I/O using aiohttp and Cache Expiry
# -----------------------------
class DataFetcher:
    """
    Asynchronously fetches stock data and news sentiment; uses local caching.
    """
    def __init__(self, cache_file: str = CONFIG["CACHE_FILE"]) -> None:
        self.cache_file: str = cache_file
        self.cache: Dict[str, Any] = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.exception(f"Error loading cache: {e}")
                self.cache = {}
        self._sentiment_transformer: Optional[SentimentTransformer] = None

    def save_cache(self) -> None:
        atomic_save(self.cache, self.cache_file)

    @retry((DataFetchError, Exception), tries=3, delay=2, backoff=2)
    async def async_fetch_stock_data(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        current_time = time.time()
        if ticker in self.cache:
            entry = self.cache[ticker]
            if current_time - entry.get("timestamp", 0) < CONFIG["CACHE_EXPIRY"]:
                logger.info(f"Loading cached data for ticker: {ticker}")
                try:
                    df = pd.read_json(entry["data"], convert_dates=True)
                    return df
                except Exception as e:
                    logger.exception(f"Error reading cache for {ticker}: {e}")
            else:
                logger.info(f"Cache expired for ticker: {ticker}")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)
        period1 = int(time.mktime(start_date.timetuple()))
        period2 = int(time.mktime(end_date.timetuple()))
        url = (f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
               f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true")
        logger.info(f"Fetching data from Yahoo Finance for ticker: {ticker}")
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(url) as response:
                    if response.status != 200:
                        raise DataFetchError(f"Failed to fetch data for {ticker}: HTTP {response.status}")
                    csv_data = await response.text()
                    df = pd.read_csv(io.StringIO(csv_data), parse_dates=["Date"], index_col="Date")
                    self.cache[ticker] = {
                        "data": df.to_json(date_format='iso'),
                        "timestamp": current_time
                    }
                    self.save_cache()
                    return df
        except Exception as e:
            logger.exception(f"Error fetching data for {ticker}: {e}")
            raise DataFetchError(e)

    @async_retry((NetworkError, Exception), tries=3, delay=2, backoff=2)
    async def fetch_news_sentiment(self, ticker: str) -> float:
        from bs4 import BeautifulSoup  # Import here to allow optional dependency
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
                            score = self._sentiment_transformer.analyze_custom(texts)
                            sentiment_scores.append(score)
                except Exception as ex:
                    logger.exception(f"Error fetching news from {url}: {ex}")
            await asyncio.gather(*(fetch(url) for url in urls))
        return np.mean(sentiment_scores) if sentiment_scores else 0.0

# -----------------------------
# Option Greeks and Pricing
# -----------------------------
def calculate_option_greeks(S: float, K: float, T: float, r: float, sigma: float) -> Tuple[float, float, float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T) + 1e-12)
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    n = lambda x: math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    delta = N(d1)
    theta = (-S * n(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N(d2))
    gamma = n(d1) / (S * sigma * math.sqrt(T) + 1e-12)
    vega = S * math.sqrt(T) * n(d1)
    return delta, gamma, theta, vega

def get_benchmark_returns(benchmark_ticker: str = "^GSPC", start: Optional[str] = None, end: Optional[str] = None) -> np.ndarray:
    try:
        benchmark = yf.Ticker(benchmark_ticker)
        df_bench = benchmark.history(start=start, end=end).dropna()
        returns = np.log(df_bench["Close"] / df_bench["Close"].shift(1)).dropna().values
        return returns
    except Exception as e:
        logger.exception(f"Error fetching benchmark returns: {e}")
        return np.array([])

def risk_adjusted_metrics(returns: np.ndarray, benchmark: Optional[np.ndarray] = None, rf: float = 0.01) -> Dict[str, float]:
    std = np.std(returns)
    sharpe = np.mean(returns - rf) / (std + 1e-6) if std > 0 else 0.0
    downside = np.std([r for r in returns if r < rf])
    sortino = np.mean(returns - rf) / (downside + 1e-6) if downside > 0 else 0.0
    beta = (np.cov(returns, benchmark)[0, 1] / (np.var(benchmark) + 1e-6)) if (benchmark is not None and benchmark.size > 0) else 1.0
    treynor = np.mean(returns - rf) / (beta + 1e-6)
    max_drawdown = np.max(np.maximum.accumulate(returns) - returns)
    calmar = np.mean(returns - rf) / (max_drawdown + 1e-6)
    return {"Sharpe": sharpe, "Sortino": sortino, "Treynor": treynor, "Calmar": calmar, "Beta": beta}

# -----------------------------
# Option Pricing Methods
# -----------------------------
class OptionPricing:
    """
    Option pricing methods including Black-Scholes, Binomial, Monte Carlo (CPU and GPU) and Jump Diffusion.
    """
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T))
        d2 = d1 - sigma * math.sqrt(T)
        N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
        if option_type.lower() == 'call':
            return S * N(d1) - K * math.exp(-r * T) * N(d2)
        return K * math.exp(-r * T) * N(-d2)

    @staticmethod
    def binomial(S: float, K: float, T: float, r: float, sigma: float, steps: int = 100, option_type: str = 'call') -> float:
        return binomial_option_unrolled(S, K, T, r, sigma, steps, option_type)

    @staticmethod
    def monte_carlo(S: float, K: float, T: float, r: float, sigma: float, simulations: int = 10000, option_type: str = 'call') -> float:
        dt = T
        rand = np.random.standard_normal(simulations)
        ST = S * np.exp((r - 0.5 * sigma**2) * dt + sigma * math.sqrt(dt) * rand)
        payoffs = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)
        return math.exp(-r * T) * np.mean(payoffs)

    @staticmethod
    def monte_carlo_gpu(S: float, K: float, T: float, r: float, sigma: float, simulations: int = 10000, option_type: str = 'call') -> float:
        dt = T
        seed = int(np.random.randint(0, 100000))
        key = random.PRNGKey(seed)
        rand = random.normal(key, shape=(simulations,))
        ST = S * jnp.exp((r - 0.5 * sigma**2)*dt + sigma * jnp.sqrt(dt) * rand)
        payoffs = jnp.where(option_type.lower()=='call', jnp.maximum(ST - K, 0), jnp.maximum(K - ST, 0))
        return float(jnp.exp(-r * T) * jnp.mean(payoffs))

    @staticmethod
    def jump_diffusion(S: float, K: float, T: float, r: float, sigma: float,
                       lam: float = 0.1, muJ: float = 0, sigmaJ: float = 0.1, option_type: str = 'call') -> float:
        price = 0.0
        for k in range(50):
            poisson_prob = math.exp(-lam * T) * (lam * T) ** k / math.factorial(k)
            sigma_k = math.sqrt(sigma**2 + k * (sigmaJ**2) / T)
            price += poisson_prob * OptionPricing.black_scholes(S, K, T, r, sigma_k, option_type)
        return price

# -----------------------------
# Faster Technical Indicators using Numba and Dask
# -----------------------------
@njit
def numba_wma(arr, period):
    n = arr.shape[0]
    result = np.empty(n, dtype=arr.dtype)
    weights = np.arange(1, period+1)
    weight_sum = weights.sum()
    for i in range(n):
        if i < period - 1:
            result[i] = np.nan
        else:
            s = 0.0
            for j in range(period):
                s += arr[i - period + 1 + j] * weights[j]
            result[i] = s / weight_sum
    return result

def dask_wma(series: pd.Series, period: int) -> pd.Series:
    # Use Dask delayed evaluation for parallel computation
    delayed_res = dask.delayed(numba_wma)(series.values.astype(np.float32), period)
    result = dask.compute(delayed_res)[0]
    return pd.Series(result, index=series.index)

# Other indicators can be similarly optimized; for brevity we wrap some pandas methods with Dask.
def dask_ema(series: pd.Series, span: int) -> pd.Series:
    delayed_ema = dask.delayed(series.ewm)(span=span, adjust=False)
    ema = dask.compute(delayed_ema)[0].mean()
    return ema

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract technical indicators using a mix of numba-accelerated functions and Dask lazy evaluation.
    Caching is applied via joblib.Memory.
    """
    @memory.cache
    def compute_features(close, volume, high, low, index):
        features = pd.DataFrame(index=index)
        features["Close"] = close
        features["WMA"] = dask_wma(pd.Series(close, index=index), 20)
        # Other indicators use pandas; in production, each would be similarly optimized.
        features["MACD"] = pd.Series(close, index=index).ewm(span=12, adjust=False).mean() - pd.Series(close, index=index).ewm(span=26, adjust=False).mean()
        features["VWAP"] = (pd.Series(close, index=index) * pd.Series(volume, index=index)).cumsum() / pd.Series(volume, index=index).cumsum()
        return features.fillna(method="bfill").fillna(method="ffill")
    
    return compute_features(df["Close"], df["Volume"], df["High"], df["Low"], df.index)

def augment_features(features: pd.DataFrame) -> pd.DataFrame:
    noise = np.random.normal(0, 0.001, size=features.shape)
    return features + noise

def pca_reduce(features: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    X = features.values - np.mean(features.values, axis=0)
    U, s, _ = svd(X, full_matrices=False)
    X_reduced = U[:, :n_components] * s[:n_components]
    return pd.DataFrame(X_reduced, index=features.index)

# -----------------------------
# Flax Neural Network for GPU-Accelerated Training
# -----------------------------
class FlaxMLP(nn.Module):
    hidden_layers: List[int]
    output_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        for h in self.hidden_layers:
            x = nn.Dense(h, dtype=jnp.float16)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        x = nn.Dense(self.output_dim, dtype=jnp.float16)(x)
        return x

class FlaxNeuralNetwork:
    """
    Neural network using Flax, optimized for multi-GPU via jax.pmap.
    """
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: Optional[List[int]] = None,
                 dropout_rate: float = 0.0, learning_rate: float = 1e-3):
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.model = FlaxMLP(hidden_layers=hidden_layers, output_dim=output_dim, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.params = None
        self.tx = optax.adam(learning_rate)
        self.opt_state = None

    def init(self, rng, sample_input):
        self.params = self.model.init(rng, sample_input)
        self.opt_state = self.tx.init(self.params)

    @jax.jit
    def loss_fn(self, params, batch):
        X, y = batch
        preds = self.model.apply(params, X)
        loss = jnp.mean((preds - y)**2)
        return loss

    @jax.jit
    def train_step(self, params, opt_state, batch):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        updates, opt_state = self.tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> List[float]:
        X = X.astype(np.float16)
        y = y.astype(np.float16)
        num_samples = X.shape[0]
        losses = []
        rng = jax.random.PRNGKey(int(time.time()))
        self.init(rng, jnp.array(X[:batch_size]))
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                batch = (jnp.array(X[idx]), jnp.array(y[idx]))
                self.params, self.opt_state, loss = self.train_step(self.params, self.opt_state, batch)
            losses.append(float(loss))
            logger.info(f"Epoch {epoch+1}: loss {loss:.4f}")
        return losses

    @jax.jit
    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.array(self.model.apply(self.params, jnp.array(X)))

    def predict_pmap(self, X: np.ndarray) -> np.ndarray:
        # If multiple GPUs are available, use pmap for parallel inference.
        parallel_predict = pmap(lambda params, x: self.model.apply(params, x))
        devices = jax.local_devices()
        if len(devices) > 1:
            # Split data across devices
            X_split = np.array_split(X, len(devices))
            params_replicated = jax.device_put_replicated(self.params, devices)
            preds = parallel_predict(params_replicated, X_split)
            return np.concatenate(preds)
        else:
            return self.predict(X)

# -----------------------------
# Ensemble Learning: Combining Flax NN and LightGBM
# -----------------------------
def ensemble_predict(X: np.ndarray, flax_model: FlaxNeuralNetwork, lgb_model: Any, blend_weight: float = 0.5) -> np.ndarray:
    """
    Generate predictions from both Flax NN and LightGBM and blend them.
    """
    pred_nn = flax_model.predict_pmap(X)
    pred_lgb = lgb_model.predict(X)
    return blend_weight * pred_nn + (1 - blend_weight) * pred_lgb

# -----------------------------
# Advanced Market Regime Detection with GMM and Placeholder for LSTM/Transformer
# -----------------------------
class MarketRegimeDetector:
    """
    Detect market regimes using a Gaussian Mixture Model (GMM).
    """
    def __init__(self, n_components: int = 2):
        self.gmm = GaussianMixture(n_components=n_components, random_state=42)

    def fit(self, returns: np.ndarray):
        returns = returns.reshape(-1, 1)
        self.gmm.fit(returns)

    def predict(self, returns: np.ndarray) -> np.ndarray:
        returns = returns.reshape(-1, 1)
        return self.gmm.predict(returns)

    def predict_latest(self, returns: np.ndarray) -> int:
        preds = self.predict(returns)
        return int(preds[-1])

# -----------------------------
# Heston Simulation using JAX
# -----------------------------
from functools import partial

@partial(jit, static_argnames=['N', 'M'])
def heston_model_sim_jax(S0: float, v0: float, rho: float, kappa: float, theta: float,
                         sigma: float, r: float, T: float, N: int, M: int) -> Tuple[jnp.ndarray, jnp.ndarray]:
    dt = T / N
    mu = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, rho], [rho, 1.0]])
    S = jnp.full((N+1, M), S0)
    v = jnp.full((N+1, M), v0)
    key = random.PRNGKey(0)
    Z = random.multivariate_normal(key, mean=mu, cov=cov, shape=(N, M))
    def body(i, carry):
        S_prev, v_prev = carry
        dS = (r - 0.5 * v_prev) * dt + jnp.sqrt(v_prev * dt) * Z[i-1, :, 0]
        new_S = S_prev * jnp.exp(dS)
        dv = kappa * (theta - v_prev) * dt + sigma * jnp.sqrt(v_prev * dt) * Z[i-1, :, 1]
        new_v = jnp.maximum(v_prev + dv, 0.0)
        S = S_prev.at[i].set(new_S)
        v = v_prev.at[i].set(new_v)
        return (S, v)
    S, v = lax.fori_loop(1, N+1, body, (S, v))
    return S, v

# -----------------------------
# Dashboard for Real-Time Monitoring (with Plotly Dash, Celery placeholder, and WebSocket streaming)
# -----------------------------
def start_dashboard(metrics_data: Dict[str, Any]) -> None:
    """
    Start a production-ready dashboard using Plotly Dash.
    Note: In a full production system, Celery would be used for task scheduling and WebSocket streaming for real-time updates.
    """
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
    
    # Placeholder: integrate Celery tasks and WebSocket streaming here.
    threading.Thread(target=app.run_server, kwargs={'port': CONFIG["DASH_PORT"], 'debug': False}, daemon=True).start()
    logger.info(f"Dashboard started at http://127.0.0.1:{CONFIG['DASH_PORT']}")

# -----------------------------
# Optimization Functions
# -----------------------------
def advanced_bayesian_optimization_lr(X_train: np.ndarray, y_train: np.ndarray, n_iter: int = 25) -> Tuple[float, Any]:
    def objective_lr(lr: float) -> float:
        flax_nn = FlaxNeuralNetwork(input_dim=X_train.shape[1], output_dim=1, learning_rate=lr)
        loss_history = flax_nn.train(X_train, y_train, epochs=10, batch_size=16)
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

# -----------------------------
# Distributed Ticker Processing with Ray
# -----------------------------
@ray.remote
def process_ticker(ticker: str) -> Optional[Dict[str, Any]]:
    try:
        data_fetcher = DataFetcher()
        df = run_async(data_fetcher.async_fetch_stock_data(ticker))
        target = (df["Close"].shift(-4) / df["Close"] - 1).fillna(0)
        features = extract_features(df)
        cluster_id = CONFIG["CLUSTER_ASSIGNMENTS"].get(ticker, -1)
        features["cluster_id"] = cluster_id
        augmented = augment_features(features)
        reduced = pca_reduce(augmented, n_components=10)
        # Ensemble: first, get predictions via a walk-forward window using our ensemble
        # (For brevity, we use a dummy function below.)
        predictions = walk_forward_optimization(reduced, target)
        latest_close = df["Close"].iloc[-1]
        strike = latest_close
        r_val = 0.01
        T_val = 4 / 252
        sigma_val = np.std(np.log(df["Close"] / df["Close"].shift(1)).dropna()) * math.sqrt(252)
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as gpu_executor:
            gpu_future = gpu_executor.submit(lambda: run_async(_fetch_option_prices(latest_close, strike, T_val, r_val, sigma_val)))
            delta, gamma, theta, vega = calculate_option_greeks(latest_close, strike, T_val, r_val, sigma_val)
            sentiment = run_async(data_fetcher.fetch_news_sentiment(ticker))
            returns = np.log(df["Close"] / df["Close"].shift(1)).dropna().values
            benchmark_returns = get_benchmark_returns(start=str(df.index[0].date()), end=str(df.index[-1].date()))
            risk_metrics = {"RiskMetrics": risk_adjusted_metrics(returns, benchmark=benchmark_returns)}
            # Use advanced market regime detection with GMM
            regime_detector = MarketRegimeDetector(n_components=2)
            regime_detector.fit(returns)
            regime = regime_detector.predict_latest(returns)
            bs_price, binom_price, mc_price = gpu_future.result()
        return {
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
            "Predictions": predictions,
            "Cluster": cluster_id
        }
    except Exception as e:
        logger.exception(f"Error processing ticker {ticker}")
        return None

def ray_parallel_ticker_processing(tickers: List[str]) -> List[Dict[str, Any]]:
    futures = [process_ticker.remote(ticker) for ticker in tickers]
    results = ray.get(futures)
    return [res for res in results if res is not None]

# -----------------------------
# Walk-Forward Optimization using Dask
# -----------------------------
def walk_forward_optimization(features: pd.DataFrame, targets: pd.Series, window: int = 100) -> List[Tuple[Any, float]]:
    @delayed
    def process_window(start):
        X_train = features.iloc[start:start + window].values
        y_train = targets.iloc[start:start + window].values.reshape(-1, 1)
        best_lr, _ = advanced_bayesian_optimization_lr(X_train, y_train, n_iter=5)
        flax_nn = FlaxNeuralNetwork(input_dim=X_train.shape[1], output_dim=1, learning_rate=best_lr)
        flax_nn.train(X_train, y_train, epochs=50, batch_size=16)
        X_pred = features.iloc[start + window:start + window + 1].values
        pred_val = float(flax_nn.predict(X_pred)[0])
        return (features.index[start + window], pred_val)
    
    tasks = [process_window(start) for start in range(0, len(features) - window, window)]
    predictions = compute(*tasks)
    return list(predictions)

# -----------------------------
# Asynchronous GPU Option Pricing Helper
# -----------------------------
async def _fetch_option_prices(latest_close: float, strike: float, T: float, r: float, sigma: float) -> Tuple[float, float, float]:
    bs_task = asyncio.to_thread(OptionPricing.black_scholes, latest_close, strike, T, r, sigma, "call")
    binom_task = asyncio.to_thread(OptionPricing.binomial, latest_close, strike, T, r, sigma, 100, "call")
    mc_task = asyncio.to_thread(OptionPricing.monte_carlo_gpu, latest_close, strike, T, r, sigma, 10000, "call")
    return await asyncio.gather(bs_task, binom_task, mc_task)

# -----------------------------
# GPU Vector Test Function
# -----------------------------
def gpu_vector_test() -> Optional[np.ndarray]:
    try:
        multi_gpu_accel = MultiGPUAccelerator()
        a = np.random.rand(1000000)
        b = np.random.rand(1000000)
        result = multi_gpu_accel.vector_multiply(a, b)
        if not np.allclose(result, a * b, atol=1e-5):
            raise ValueError("MultiGPU vector multiplication mismatch.")
        logger.info("MultiGPU vector multiply test passed.")
        return result
    except Exception as e:
        logger.exception(f"MultiGPU test failed: {e}")
        return None

# -----------------------------
# Build Prediction Table and Plotting
# -----------------------------
def build_prediction_table(results: List[Dict[str, Any]]) -> Optional[pd.DataFrame]:
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
                    "Sentiment": res["Sentiment"],
                    "Cluster": res["Cluster"]
                })
    return pd.DataFrame(rows) if rows else None

def plot_performance_metrics(results: List[Dict[str, Any]]) -> None:
    tickers = [res["Ticker"] for res in results if res]
    sentiments = [res["Sentiment"] for res in results if res]
    bs_prices = [res["BS_Price"] for res in results if res]
    try:
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 6))
        plt.scatter(sentiments, bs_prices, c="blue")
        plt.xlabel("News Sentiment")
        plt.ylabel("Black–Scholes Price")
        plt.title("Performance Metrics per Ticker")
        plt.grid(True)
        plt.savefig("performance_metrics.png")
        plt.close()
    except Exception as e:
        logger.exception(f"Error plotting performance metrics: {e}")

# -----------------------------
# Main Execution Function
# -----------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run unit tests and exit")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    args = parser.parse_args()
    if args.test:
        run_tests()
        return

    now = datetime.datetime.now()
    if now.weekday() != CONFIG["RUN_DAY"] or now.hour < CONFIG["RUN_HOUR"]:
        logger.error("This script is designed to run on Monday after 16:00. Exiting.")
        sys.exit(1)

    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        try:
            tickers_input = input("Enter tickers (comma separated): ")
            tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        except Exception as e:
            logger.exception(f"Input error: {e}")
            sys.exit(1)
    if not tickers:
        logger.error("No tickers provided. Exiting.")
        sys.exit(1)

    global CONFIG
    CONFIG["CLUSTER_ASSIGNMENTS"] = perform_clustering(tickers, n_clusters=2)
    logger.info(f"Cluster assignments: {CONFIG['CLUSTER_ASSIGNMENTS']}")

    ray.init(ignore_reinit_error=True)
    results = ray_parallel_ticker_processing(tickers)
    if not results:
        logger.info("No predictions generated.")
        return

    try:
        gpu_vector_test()
    except Exception as e:
        logger.exception(f"GPU test failed: {e}")

    prediction_table = build_prediction_table(results)
    if prediction_table is not None and not prediction_table.empty:
        print("\nPrediction Table:")
        print(prediction_table.to_string(index=False))
    else:
        print("No significant predictions found.")

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
    logger.info("Running unit tests...")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    cpu_result = fast_cpu_distance(a, b)
    assert abs(cpu_result) < 1e-6, "fast_cpu_distance test failed."
    
    multi_gpu_accel = MultiGPUAccelerator()
    res_gpu = multi_gpu_accel.vector_multiply(a, b)
    assert np.allclose(res_gpu, a * b, atol=1e-5), "MultiGPU routine test failed."

    X_test = np.random.rand(5, 10)
    y_test = np.random.rand(5, 1)
    flax_nn = FlaxNeuralNetwork(input_dim=10, output_dim=1, dropout_rate=0.2, learning_rate=1e-3)
    loss_val = flax_nn.loss_fn(flax_nn.model.init(jax.random.PRNGKey(0), jnp.array(X_test)), (jnp.array(X_test), jnp.array(y_test)))
    g = grad(flax_nn.loss_fn)(flax_nn.model.init(jax.random.PRNGKey(0), jnp.array(X_test)), (jnp.array(X_test), jnp.array(y_test)))
    for key in g:
        assert g[key] is not None, f"Gradient for {key} missing."
    
    st = SentimentTransformer()
    score = st.analyze("This is a great day for trading!")
    assert isinstance(score, float), "Sentiment transformer test failed."
    
    df_sample = pd.DataFrame({
        "Close": np.linspace(100, 110, 30),
        "Volume": np.random.randint(1000, 5000, 30),
        "High": np.linspace(101, 111, 30),
        "Low": np.linspace(99, 109, 30)
    }, index=pd.date_range("2020-01-01", periods=30))
    feats = extract_features(df_sample)
    assert not feats.empty, "Feature extraction test failed."

    call_counter = {"count": 0}
    @retry(Exception, tries=3, delay=0.1, backoff=1)
    def test_retry_success():
        call_counter["count"] += 1
        if call_counter["count"] < 3:
            raise ValueError("Failing")
        return "succeeded"
    assert test_retry_success() == "succeeded", "Retry decorator test failed."

    @profile
    def dummy_sleep():
        time.sleep(0.2)
        return "done"
    assert dummy_sleep() == "done", "Profiling decorator test failed."

    sample_ticker = "AAPL"
    res = ray.get(process_ticker.remote(sample_ticker))
    assert res is not None, "process_ticker integration test failed."

    logger.info("All tests passed.")

def perform_clustering(tickers: List[str], n_clusters: int = 2) -> Dict[str, int]:
    features_list = []
    valid_tickers = []
    for ticker in tickers:
        try:
            df = yf.Ticker(ticker).history(period='5y')
            df_friday = df[df.index.weekday == 4]
            if df_friday.empty:
                continue
            df_friday['return'] = df_friday['Close'].pct_change()
            avg_return = df_friday['return'].mean() if not df_friday['return'].isnull().all() else 0.0
            std_return = df_friday['return'].std() if not df_friday['return'].isnull().all() else 0.0
            avg_volume = df_friday['Volume'].mean() if not df_friday['Volume'].isnull().all() else 0.0
            features_list.append([avg_return, std_return, avg_volume])
            valid_tickers.append(ticker)
        except Exception as e:
            logger.exception(f"Clustering failed for ticker {ticker}: {e}")
    if not features_list:
        return {}
    X = np.array(features_list)
    try:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=3)
        labels = clusterer.fit_predict(X)
    except Exception as e:
        logger.exception(f"HDBSCAN clustering failed: {e}")
        labels = [-1] * len(valid_tickers)
    logger.info(f"Clustering completed. Labels: {labels}")
    return {ticker: int(label) for ticker, label in zip(valid_tickers, labels)}

if __name__ == "__main__":
    main()
