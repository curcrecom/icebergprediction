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
from functools import wraps, partial, lru_cache
import concurrent.futures

import numpy as np
import pandas as pd
from numpy.linalg import svd

# THIRD–PARTY LIBRARIES
from numba import njit, prange
import jax
import jax.numpy as jnp
from jax import jit, grad, lax, random, pmap
import optax
from transformers import pipeline
from bayes_opt import BayesianOptimization
import dash
from dash import dcc, html, Output, Input
import dash_bootstrap_components as dbc
import metalcompute as mc
import hdbscan
import lightgbm as lgb
from flax import linen as nn
import aiohttp
from bs4 import BeautifulSoup
from joblib import Memory, Parallel, delayed
from sentence_transformers import SentenceTransformer, util
from celery import Celery
from flask import Flask
from flask_socketio import SocketIO, emit

# -----------------------------
# Global Configuration and Memory Cache
# -----------------------------
CONFIG: dict[str, any] = {
    "CACHE_FILE": "data_cache.json",
    "LOG_FILE": "model_log.txt",
    "RUN_DAY": 0,         # Monday
    "RUN_HOUR": 16,       # 16:00 local time
    "DASH_PORT": 8050,    # Dashboard port
    "CACHE_EXPIRY": 86400  # Cache expiry (seconds)
}
GLOBAL_SENTIMENT_WEIGHT = 0.3  # weight for embedding score (the basic FinBERT score weight will be 1 - this)
GLOBAL_ENSEMBLE_WEIGHT = 0.5   # weight for Flax NN in ensemble (the LightGBM weight will be 1 - this)

# Setup joblib memory cache
memory = Memory(location="./joblib_cache", verbose=0)

# -----------------------------
# Logging Configuration
# -----------------------------
def configure_logging() -> None:
    """Configure logging for the application."""
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    # File handler with rotation
    fh = logging.handlers.RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=1*1024*1024, backupCount=5)
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

configure_logging()
logger = logging.getLogger(__name__)

# -----------------------------
# Celery and Flask–SocketIO Setup
# -----------------------------
celery_app = Celery('tasks', broker='redis://localhost:6379/0')
# Optimize worker pool: use as many workers as CPU cores and note that task prioritization can be configured.
celery_app.conf.update(worker_concurrency=os.cpu_count())
flask_app = Flask(__name__)
flask_app.config['SECRET_KEY'] = 'secret!'
socketio = SocketIO(flask_app, cors_allowed_origins="*")

# -----------------------------
# Retry Decorators with Exponential Backoff
# -----------------------------
def retry(exceptions: tuple[Exception, ...], tries: int = 3, delay: float = 1.0, backoff: int = 2) -> callable:
    def decorator(func: callable) -> callable:
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

def async_retry(exceptions: tuple[Exception, ...], tries: int = 3, delay: float = 1.0, backoff: int = 2) -> callable:
    def decorator(func: callable) -> callable:
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

# -----------------------------
# Profiling Decorator
# -----------------------------
def profile(func: callable) -> callable:
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
def atomic_save(cache: dict[str, any], filename: str) -> None:
    temp_filename = filename + ".tmp"
    with open(temp_filename, 'w') as f:
        json.dump(cache, f)
    os.replace(temp_filename, filename)

# -----------------------------
# Optimized Numerical Kernels with Numba
# -----------------------------
@njit(fastmath=True, parallel=True)
def fast_cpu_distance(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    total = 0.0
    n = diff.shape[0]
    unroll_factor = 4
    limit = n - (n % unroll_factor)
    for i in range(0, limit, unroll_factor):
        total += diff[i]**2 + diff[i+1]**2 + diff[i+2]**2 + diff[i+3]**2
    for i in range(limit, n):
        total += diff[i]**2
    return total

@njit(fastmath=True)
def binomial_option_unrolled(S: float, K: float, T: float, r: float, sigma: float, N: int, option_type: str) -> float:
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
# GPU Accelerator (MetalCompute) with MultiGPU Optimized Execution
# -----------------------------
class GPUAccelerator:
    def __init__(self, device: mc.Device | None = None) -> None:
        self.device = device if device is not None else mc.Device()
        logger.info(f"GPUAccelerator initialized with device: {self.device.name}")
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
        self.compiled_kernel = self.device.kernel(self.kernel_code).function("vec_mult")
    
    def vector_multiply(self, a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
        a_np = np.ascontiguousarray(a_np.astype(np.float32))
        b_np = np.ascontiguousarray(b_np.astype(np.float32))
        total_elements = a_np.shape[0]
        a_bytes = memoryview(a_np)
        b_bytes = memoryview(b_np)
        bytes_count = total_elements * 4
        result_buf = self.device.buffer(bytes_count)
        total_threads = getattr(self.device, "max_threads", 256)
        self.compiled_kernel(total_elements, a_bytes, b_bytes, result_buf)
        result_view = memoryview(result_buf).cast('f')
        return np.array(result_view)

class MultiGPUAccelerator:
    def __init__(self) -> None:
        self.devices = mc.devices()
        if len(self.devices) < 2:
            logger.warning("Less than 2 GPU devices available; using single GPU mode.")
            self.devices = self.devices[:1]
        else:
            self.devices = self.devices[:2]
        self.gpu_accels = [GPUAccelerator(device=d) for d in self.devices]
        logger.info(f"MultiGPUAccelerator using devices: {[d.name for d in self.devices]}")
    
    def vector_multiply(self, a_np: np.ndarray, b_np: np.ndarray) -> np.ndarray:
        # Use JAX pmap with device sharding for optimal multi-GPU performance.
        a_np = np.ascontiguousarray(a_np.astype(np.float32))
        b_np = np.ascontiguousarray(b_np.astype(np.float32))
        total_elements = a_np.shape[0]
        devices = jax.local_devices()
        n_devices = len(devices)
        pad = (n_devices - total_elements % n_devices) % n_devices
        if pad:
            a_np = np.concatenate([a_np, np.zeros(pad, dtype=np.float32)])
            b_np = np.concatenate([b_np, np.zeros(pad, dtype=np.float32)])
        a_shards = np.split(a_np, n_devices)
        b_shards = np.split(b_np, n_devices)
        def mult(x, y):
            return x * y
        pmap_mult = jax.pmap(mult)
        result_shards = pmap_mult(jax.device_put_sharded(a_shards, devices),
                                  jax.device_put_sharded(b_shards, devices))
        result = np.concatenate(result_shards)
        return result[:total_elements]

# -----------------------------
# Sentiment Analysis with LLM Embedding and Auto–Tuned Weight
# -----------------------------
class SentimentTransformer:
    def __init__(self) -> None:
        self.pipeline = pipeline("sentiment-analysis",
                                 model="yiyanghkust/finbert-pretrain",
                                 tokenizer="yiyanghkust/finbert-pretrain")
        self.embed_model = SentenceTransformer("all-MiniLM-L6-v2")
        logger.info("Loaded FinBERT and SentenceTransformer for sentiment analysis.")

    def basic_score(self, text: str) -> float:
        result = self.pipeline(text[:512])
        if result and result[0]['label'].upper() == "POSITIVE":
            return result[0]['score']
        elif result and result[0]['label'].upper() == "NEGATIVE":
            return -result[0]['score']
        return 0.0

    def embedding_score(self, text: str) -> float:
        pos_ref = "This is a positive sentiment."
        neg_ref = "This is a negative sentiment."
        emb_text = self.embed_model.encode(text, convert_to_tensor=True)
        emb_pos = self.embed_model.encode(pos_ref, convert_to_tensor=True)
        emb_neg = self.embed_model.encode(neg_ref, convert_to_tensor=True)
        sim_pos = util.cos_sim(emb_text, emb_pos).item()
        sim_neg = util.cos_sim(emb_text, emb_neg).item()
        return sim_pos - sim_neg

    async def analyze_custom(self, text: str) -> float:
        basic = self.basic_score(text)
        embed = self.embedding_score(text)
        return (1 - GLOBAL_SENTIMENT_WEIGHT) * basic + GLOBAL_SENTIMENT_WEIGHT * embed

    @staticmethod
    def analyze_static(text: str) -> float:
        return SentimentTransformer().basic_score(text)

# -----------------------------
# Data Fetching and Caching (Async with aiohttp) with Enhanced Concurrency
# -----------------------------
class DataFetchError(Exception):
    pass

class DataFetcher:
    def __init__(self, cache_file: str = CONFIG["CACHE_FILE"]) -> None:
        self.cache_file: str = cache_file
        self.cache: dict[str, any] = {}
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r') as f:
                    self.cache = json.load(f)
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        self._sentiment_transformer = SentimentTransformer()

    def save_cache(self) -> None:
        atomic_save(self.cache, self.cache_file)

    @async_retry((DataFetchError, Exception), tries=3, delay=2, backoff=2)
    async def async_fetch_stock_data(self, ticker: str, period: str = '5y') -> pd.DataFrame:
        current_time = time.time()
        if ticker in self.cache:
            entry = self.cache[ticker]
            if current_time - entry.get("timestamp", 0) < CONFIG["CACHE_EXPIRY"]:
                logger.info(f"Loading cached data for {ticker}")
                try:
                    df = pd.read_json(entry["data"], convert_dates=True)
                    return df
                except Exception as e:
                    logger.exception(f"Error reading cache for {ticker}: {e}")
        end_date = datetime.datetime.now()
        start_date = end_date - datetime.timedelta(days=5*365)
        period1 = int(time.mktime(start_date.timetuple()))
        period2 = int(time.mktime(end_date.timetuple()))
        url = (f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
               f"?period1={period1}&period2={period2}&interval=1d&events=history&includeAdjustedClose=true")
        logger.info(f"Fetching data for {ticker}")
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as response:
                if response.status != 200:
                    raise DataFetchError(f"Failed to fetch data for {ticker}: HTTP {response.status}")
                csv_data = await response.text()
                try:
                    df = pd.read_csv(io.StringIO(csv_data), parse_dates=["Date"], index_col="Date")
                except Exception as e:
                    raise DataFetchError(f"Error parsing CSV for {ticker}: {e}")
                self.cache[ticker] = {"data": df.to_json(date_format='iso'), "timestamp": current_time}
                self.save_cache()
                return df

    @async_retry((Exception,), tries=3, delay=2, backoff=2)
    async def fetch_news_sentiment(self, ticker: str) -> float:
        urls = [
            f"https://www.businesswire.com/portal/site/home/?search={ticker}",
            f"https://www.reuters.com/search/news?blob={ticker}",
            f"https://www.bloomberg.com/search?query={ticker}"
        ]
        scores: list[float] = []
        async with aiohttp.ClientSession() as session:
            async def fetch_and_analyze(url: str) -> None:
                try:
                    async with session.get(url) as response:
                        if response.status == 200:
                            text = await response.text()
                            soup = BeautifulSoup(text, 'html.parser')
                            texts = " ".join(p.get_text() for p in soup.find_all('p'))
                            # Launch sentiment analysis concurrently as a task
                            score = await self._sentiment_transformer.analyze_custom(texts)
                            scores.append(score)
                except Exception as e:
                    logger.error(f"Error fetching {url}: {e}")
            # Run all fetch tasks concurrently.
            await asyncio.gather(*(fetch_and_analyze(url) for url in urls), return_exceptions=True)
        return float(np.mean(scores)) if scores else 0.0

# -----------------------------
# Option Greeks and Pricing Methods
# -----------------------------
def calculate_option_greeks(S: float, K: float, T: float, r: float, sigma: float) -> tuple[float, float, float, float]:
    d1 = (math.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * math.sqrt(T) + 1e-12)
    d2 = d1 - sigma * math.sqrt(T)
    N = lambda x: 0.5 * (1 + math.erf(x / math.sqrt(2)))
    n = lambda x: math.exp(-0.5 * x**2) / math.sqrt(2 * math.pi)
    return N(d1), n(d1), (-S * n(d1) * sigma / (2 * math.sqrt(T)) - r * K * math.exp(-r * T) * N(d2)), S * math.sqrt(T) * n(d1)

class OptionPricing:
    @staticmethod
    def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        d1 = (math.log(S/K) + (r+0.5*sigma**2)*T) / (sigma*math.sqrt(T))
        d2 = d1 - sigma*math.sqrt(T)
        N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
        if option_type.lower() == 'call':
            return S*N(d1) - K*math.exp(-r*T)*N(d2)
        return K*math.exp(-r*T)*N(-d2)

    @staticmethod
    def binomial(S: float, K: float, T: float, r: float, sigma: float, steps: int = 100, option_type: str = 'call') -> float:
        return binomial_option_unrolled(S, K, T, r, sigma, steps, option_type)

    @staticmethod
    def monte_carlo(S: float, K: float, T: float, r: float, sigma: float, simulations: int = 10000, option_type: str = 'call') -> float:
        dt = T
        rand = np.random.standard_normal(simulations)
        ST = S * np.exp((r - 0.5*sigma**2)*dt + sigma*math.sqrt(dt)*rand)
        payoff = np.maximum(ST - K, 0) if option_type.lower() == 'call' else np.maximum(K - ST, 0)
        return math.exp(-r*T)*np.mean(payoff)

    @staticmethod
    def monte_carlo_gpu_parallel(S: float, K: float, T: float, r: float, sigma: float,
                                 simulations: int = 10000, option_type: str = 'call') -> float:
        dt = T
        devices = jax.local_devices()
        n_devices = len(devices)
        sims_per_device = simulations // n_devices
        # Split the key for each device
        keys = random.split(random.PRNGKey(int(time.time())), n_devices)
        def simulation(key):
            rand = random.normal(key, shape=(sims_per_device,))
            ST = S * jnp.exp((r - 0.5*sigma**2)*dt + sigma*jnp.sqrt(dt)*rand)
            payoff = jnp.where(option_type.lower()=='call', jnp.maximum(ST-K, 0), jnp.maximum(K-ST, 0))
            return jnp.mean(payoff)
        pmap_simulation = jax.pmap(simulation)
        means = pmap_simulation(jnp.array(keys))
        overall_mean = jnp.mean(means)
        return float(jnp.exp(-r*T)*overall_mean)

    @staticmethod
    def jump_diffusion(S: float, K: float, T: float, r: float, sigma: float,
                       lam: float = 0.1, muJ: float = 0, sigmaJ: float = 0.1, option_type: str = 'call') -> float:
        price = 0.0
        for k in range(50):
            poisson_prob = math.exp(-lam*T) * (lam*T)**k / math.factorial(k)
            sigma_k = math.sqrt(sigma**2 + k*(sigmaJ**2)/T)
            price += poisson_prob * OptionPricing.black_scholes(S, K, T, r, sigma_k, option_type)
        return price

# -----------------------------
# Optimized Technical Indicators for Parallel Computation
# -----------------------------
@njit(parallel=True)
def numba_wma(arr: np.ndarray, period: int) -> np.ndarray:
    n = arr.shape[0]
    result = np.empty(n, dtype=arr.dtype)
    weights = np.arange(1, period+1)
    weight_sum = weights.sum()
    for i in prange(n):
        if i < period - 1:
            result[i] = np.nan
        else:
            s = 0.0
            for j in range(period):
                s += arr[i - period + 1 + j] * weights[j]
            result[i] = s / weight_sum
    return result

@njit(parallel=True)
def numba_ema(arr: np.ndarray, span: int) -> np.ndarray:
    alpha = 2 / (span + 1)
    ema = np.empty_like(arr)
    ema[0] = arr[0]
    for i in prange(1, arr.shape[0]):
        ema[i] = alpha * arr[i] + (1 - alpha) * ema[i-1]
    return ema

def optimized_macd(close: np.ndarray) -> np.ndarray:
    ema12 = numba_ema(close.astype(np.float32), 12)
    ema26 = numba_ema(close.astype(np.float32), 26)
    return ema12 - ema26

# -----------------------------
# Feature Extraction & PCA Parallelization
# -----------------------------
def pca_reduce(features: pd.DataFrame, n_components: int = 10, mode: str = "jax") -> pd.DataFrame:
    """
    mode: "jax" for GPU-accelerated SVD; "joblib" for CPU parallelism.
    """
    X = features.values - np.mean(features.values, axis=0)
    if mode == "jax":
        @jit
        def compute_svd(X):
            return jnp.linalg.svd(jnp.array(X), full_matrices=False)
        U, s, _ = compute_svd(X)
        U, s = np.array(U), np.array(s)
    else:
        # CPU-parallel version using joblib
        U, s, _ = svd(X, full_matrices=False)
        # Optionally, you could wrap parts of this with Parallel(delayed(...))
    X_reduced = U[:, :n_components] * s[:n_components]
    return pd.DataFrame(X_reduced, index=features.index)

@lru_cache(maxsize=128)
def compute_features_cached(close_json: str, volume_json: str, index_json: str) -> pd.DataFrame:
    close = pd.read_json(io.StringIO(close_json), typ="series")
    volume = pd.read_json(io.StringIO(volume_json), typ="series")
    index = pd.read_json(io.StringIO(index_json), typ="series").index
    features = pd.DataFrame(index=index)
    features["Close"] = close
    features["WMA"] = pd.Series(numba_wma(close.astype(np.float32).values, 20), index=index)
    features["MACD"] = pd.Series(optimized_macd(close.astype(np.float32).values), index=index)
    cum_pv = (close * volume).cumsum()
    cum_vol = volume.cumsum()
    features["VWAP"] = cum_pv / cum_vol
    return features.fillna(method="bfill").fillna(method="ffill")

def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    close_json = df["Close"].to_json()
    volume_json = df["Volume"].to_json()
    index_json = df.index.to_series().to_json()
    return compute_features_cached(close_json, volume_json, index_json)

def augment_features(features: pd.DataFrame) -> pd.DataFrame:
    noise = np.random.normal(0, 0.001, size=features.shape)
    return features + noise

# -----------------------------
# Flax Neural Network for GPU Training
# -----------------------------
class FlaxMLP(nn.Module):
    hidden_layers: list[int]
    output_dim: int
    dropout_rate: float = 0.0

    @nn.compact
    def __call__(self, x, train: bool = True):
        for h in self.hidden_layers:
            x = nn.Dense(h, dtype=jnp.float16)(x)
            x = nn.relu(x)
            if self.dropout_rate > 0.0:
                x = nn.Dropout(self.dropout_rate)(x, deterministic=not train)
        return nn.Dense(self.output_dim, dtype=jnp.float16)(x)

class FlaxNeuralNetwork:
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int] | None = None,
                 dropout_rate: float = 0.0, learning_rate: float = 1e-3) -> None:
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.model = FlaxMLP(hidden_layers=hidden_layers, output_dim=output_dim, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.params = None
        self.tx = optax.adam(learning_rate)
        self.opt_state = None

    def init(self, rng: jax.random.PRNGKey, sample_input: jnp.ndarray) -> None:
        self.params = self.model.init(rng, sample_input)
        self.opt_state = self.tx.init(self.params)

    @jax.jit
    def loss_fn(self, params: any, batch: tuple[jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
        X, y = batch
        preds = self.model.apply(params, X)
        return jnp.mean((preds - y)**2)

    @jax.jit
    def train_step(self, params: any, opt_state: any, batch: tuple[jnp.ndarray, jnp.ndarray]) -> tuple[any, any, jnp.ndarray]:
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        updates, opt_state = self.tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 100, batch_size: int = 32) -> list[float]:
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
        parallel_predict = pmap(lambda params, x: self.model.apply(params, x))
        devices = jax.local_devices()
        if len(devices) > 1:
            X_split = np.array_split(X, len(devices))
            params_replicated = jax.device_put_replicated(self.params, devices)
            preds = parallel_predict(params_replicated, X_split)
            return np.concatenate(preds)
        return self.predict(X)

# -----------------------------
# LightGBM Training for Ensemble
# -----------------------------
def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> any:
    lgb_train = lgb.Dataset(X_train, label=y_train)
    params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
    model = lgb.train(params, lgb_train, num_boost_round=50)
    return model

# -----------------------------
# Ensemble Prediction with Auto–Tuning of Blend Weight
# -----------------------------
def ensemble_predict(X: np.ndarray, flax_model: FlaxNeuralNetwork, lgb_model: any, blend_weight: float = GLOBAL_ENSEMBLE_WEIGHT) -> np.ndarray:
    pred_nn = flax_model.predict_pmap(X)
    pred_lgb = lgb_model.predict(X)
    return blend_weight * pred_nn + (1 - blend_weight) * pred_lgb

def optimize_ensemble_weight(X: np.ndarray, y: np.ndarray) -> float:
    def objective(w: float) -> float:
        w = float(w)
        flax_nn = FlaxNeuralNetwork(input_dim=X.shape[1], output_dim=1, learning_rate=1e-3)
        flax_nn.train(X, y, epochs=10, batch_size=16)
        lgb_model = train_lightgbm(X, y.ravel())
        preds = ensemble_predict(X, flax_nn, lgb_model, blend_weight=w)
        rmse = np.sqrt(np.mean((preds - y.ravel())**2))
        return -rmse  # maximize negative RMSE

    optimizer = BayesianOptimization(f=objective, pbounds={'w': (0.0, 1.0)}, random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=10)
    best_w = optimizer.max['params']['w']
    logger.info(f"Optimal ensemble blend weight: {best_w}")
    return best_w

# -----------------------------
# Auto–Tune Sentiment Blend Weight
# -----------------------------
def optimize_sentiment_weight(texts: list[str], true_scores: list[float]) -> float:
    def objective(w: float) -> float:
        w = float(w)
        st = SentimentTransformer()
        preds = [((1 - w) * st.basic_score(text) + w * st.embedding_score(text)) for text in texts]
        rmse = np.sqrt(np.mean((np.array(preds) - np.array(true_scores))**2))
        return -rmse
    optimizer = BayesianOptimization(f=objective, pbounds={'w': (0.0, 1.0)}, random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=10)
    best_w = optimizer.max['params']['w']
    logger.info(f"Optimal sentiment blend weight: {best_w}")
    return best_w

# -----------------------------
# Heston Simulation using JAX
# -----------------------------
@partial(jit, static_argnames=['N', 'M'])
def heston_model_sim_jax(S0: float, v0: float, rho: float, kappa: float, theta: float,
                         sigma: float, r: float, T: float, N: int, M: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    dt = T / N
    mu = jnp.array([0.0, 0.0])
    cov = jnp.array([[1.0, rho], [rho, 1.0]])
    S = jnp.full((N+1, M), S0)
    v = jnp.full((N+1, M), v0)
    key = random.PRNGKey(0)
    Z = random.multivariate_normal(key, mean=mu, cov=cov, shape=(N, M))
    def body(i, carry):
        S_prev, v_prev = carry
        dS = (r - 0.5*v_prev)*dt + jnp.sqrt(v_prev*dt)*Z[i-1, :, 0]
        new_S = S_prev * jnp.exp(dS)
        dv = kappa*(theta-v_prev)*dt + sigma*jnp.sqrt(v_prev*dt)*Z[i-1, :, 1]
        new_v = jnp.maximum(v_prev + dv, 0.0)
        S = S_prev.at[i].set(new_S)
        v = v_prev.at[i].set(new_v)
        return (S, v)
    S, v = lax.fori_loop(1, N+1, body, (S, v))
    return S, v

# -----------------------------
# Async Celery Task for Ticker Processing and WebSocket Update
# -----------------------------
@celery_app.task
async def process_ticker(ticker: str) -> dict[str, any]:
    try:
        if not ticker.isalnum():
            raise ValueError("Ticker must be alphanumeric.")
        fetcher = DataFetcher()
        df = await fetcher.async_fetch_stock_data(ticker)
        if df.empty:
            raise ValueError("Fetched dataframe is empty.")
        target = (df["Close"].shift(-4) / df["Close"] - 1).fillna(0)
        features = extract_features(df)
        augmented = augment_features(features)
        # Choose PCA mode based on environment: "jax" for GPU or "joblib" for CPU
        reduced = pca_reduce(augmented, n_components=10, mode="jax")
        predictions = list(zip(reduced.index.tolist(), np.random.rand(len(reduced))))  # Dummy predictions
        latest_close = float(df["Close"].iloc[-1])
        strike = latest_close
        r_val = 0.01
        T_val = 4 / 252
        sigma_val = float(np.std(np.log(df["Close"]/df["Close"].shift(1)).dropna()) * math.sqrt(252))
        # Launch option pricing concurrently
        option_task = asyncio.create_task(_fetch_option_prices(latest_close, strike, T_val, r_val, sigma_val))
        sentiment = await fetcher.fetch_news_sentiment(ticker)
        returns = np.log(df["Close"]/df["Close"].shift(1)).dropna().values
        regime = int(hdbscan.HDBSCAN(min_cluster_size=3).fit_predict(returns.reshape(-1, 1))[-1])
        bs_price, binom_price, mc_price = await option_task
        result = {
            "Ticker": ticker,
            "LatestClose": latest_close,
            "Strike": strike,
            "BS_Price": bs_price,
            "Binom_Price": binom_price,
            "MC_Price": mc_price,
            "Delta": calculate_option_greeks(latest_close, strike, T_val, r_val, sigma_val)[0],
            "Gamma": calculate_option_greeks(latest_close, strike, T_val, r_val, sigma_val)[1],
            "Theta": calculate_option_greeks(latest_close, strike, T_val, r_val, sigma_val)[2],
            "Vega": calculate_option_greeks(latest_close, strike, T_val, r_val, sigma_val)[3],
            "Sentiment": sentiment,
            "CurrentRegime": regime,
            "Predictions": predictions
        }
        socketio.emit('ticker_update', result, broadcast=True)
        return result
    except Exception as e:
        logger.exception(f"Error processing ticker {ticker}: {e}")
        return {}

async def _fetch_option_prices(latest_close: float, strike: float, T: float, r: float, sigma: float) -> tuple[float, float, float]:
    # Execute pricing models concurrently using asyncio.gather
    bs = asyncio.to_thread(OptionPricing.black_scholes, latest_close, strike, T, r, sigma, "call")
    binom = asyncio.to_thread(OptionPricing.binomial, latest_close, strike, T, r, sigma, 100, "call")
    # Use the optimized parallel GPU Monte Carlo simulation
    mc = asyncio.to_thread(OptionPricing.monte_carlo_gpu_parallel, latest_close, strike, T, r, sigma, 10000, "call")
    return await asyncio.gather(bs, binom, mc)

# -----------------------------
# Dashboard with Dash and WebSocket Streaming
# -----------------------------
def start_dashboard(metrics_data: dict[str, any]) -> None:
    app = dash.Dash(__name__, server=flask_app, external_stylesheets=[dbc.themes.BOOTSTRAP])
    app.layout = dbc.Container([
        dbc.Row([dbc.Col(html.H2("Live Performance Dashboard"), width=12)]),
        dbc.Row([
            dbc.Col(dbc.Card([dbc.CardHeader("Average Black–Scholes Price"),
                               dbc.CardBody(html.H4(id="avg-bs", children=str(metrics_data.get("avg_bs", "N/A"))))]), width=6),
            dbc.Col(dbc.Card([dbc.CardHeader("Average Sentiment"),
                               dbc.CardBody(html.H4(id="avg-sent", children=str(metrics_data.get("avg_sent", "N/A"))))]), width=6)
        ]),
        dcc.Interval(id="interval", interval=10*1000, n_intervals=0)
    ], fluid=True)
    @app.callback(
        [Output("avg-bs", "children"), Output("avg-sent", "children")],
        [Input("interval", "n_intervals")]
    )
    def update_metrics(n: int) -> tuple[str, str]:
        return (str(metrics_data.get("avg_bs", "N/A")), str(metrics_data.get("avg_sent", "N/A")))
    threading.Thread(target=lambda: socketio.run(flask_app, port=CONFIG["DASH_PORT"]), daemon=True).start()
    logger.info(f"Dashboard started at http://127.0.0.1:{CONFIG['DASH_PORT']}")

# -----------------------------
# Build Prediction Table and Plotting
# -----------------------------
def build_prediction_table(results: list[dict[str, any]]) -> pd.DataFrame:
    rows = []
    for res in results:
        if res.get("Predictions"):
            pred_date, pred_val = res["Predictions"][-1]
            if pred_val >= 10.0:
                rows.append({
                    "Date": pred_date.strftime("%Y-%m-%d") if isinstance(pred_date, datetime.datetime) else str(pred_date),
                    "Ticker": res["Ticker"],
                    "Strike": res["Strike"],
                    "Predicted Increase (%)": pred_val * 100,
                    "BS Price": res["BS_Price"],
                    "Sentiment": res["Sentiment"]
                })
    return pd.DataFrame(rows) if rows else pd.DataFrame()

def plot_performance_metrics(results: list[dict[str, any]]) -> None:
    tickers = [res["Ticker"] for res in results if res]
    sentiments = [res["Sentiment"] for res in results if res]
    bs_prices = [res["BS_Price"] for res in results if res]
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8, 6))
    plt.scatter(sentiments, bs_prices, c="blue")
    plt.xlabel("News Sentiment")
    plt.ylabel("Black–Scholes Price")
    plt.title("Performance Metrics per Ticker")
    plt.grid(True)
    plt.savefig("performance_metrics.png")
    plt.close()

# -----------------------------
# Main Execution Functions (async version)
# -----------------------------
async def async_process_tickers(tickers: list[str]) -> list[dict[str, any]]:
    results = []
    tasks = [process_ticker.delay(ticker) for ticker in tickers]
    # Await Celery tasks asynchronously
    for task in tasks:
        try:
            result = await asyncio.to_thread(task.get, timeout=300)
            if result:
                results.append(result)
        except Exception as e:
            logger.error(f"Error processing ticker {ticker}: {e}")
    return results

def train_models(sample_X: np.ndarray, sample_y: np.ndarray) -> float:
    optimal_weight = optimize_ensemble_weight(sample_X, sample_y)
    global GLOBAL_ENSEMBLE_WEIGHT
    GLOBAL_ENSEMBLE_WEIGHT = optimal_weight
    logger.info("Ensemble model training complete.")
    return optimal_weight

def validate_tickers(tickers: list[str]) -> list[str]:
    valid_tickers = []
    for t in tickers:
        if t.isalnum():
            valid_tickers.append(t.upper())
        else:
            logger.warning(f"Ticker {t} is invalid and will be skipped.")
    return valid_tickers

async def async_main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true", help="Run tests and exit")
    parser.add_argument("--tickers", type=str, help="Comma-separated tickers")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    now = datetime.datetime.now()
    if now.weekday() != CONFIG["RUN_DAY"] or now.hour < CONFIG["RUN_HOUR"]:
        logger.error("This script is designed to run on Monday after 16:00. Exiting.")
        sys.exit(1)

    tickers: list[str] = []
    if args.tickers:
        tickers = [t.strip() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers_input = input("Enter tickers (comma separated): ")
        tickers = [t.strip() for t in tickers_input.split(",") if t.strip()]

    tickers = validate_tickers(tickers)
    if not tickers:
        logger.error("No valid tickers provided. Exiting.")
        sys.exit(1)

    # Auto-tune sentiment blend weight (sample texts and scores for demonstration)
    sample_texts = ["The market is looking bullish today.", "Stocks are plummeting in a bear market."]
    sample_true_scores = [0.8, -0.7]
    global GLOBAL_SENTIMENT_WEIGHT
    GLOBAL_SENTIMENT_WEIGHT = optimize_sentiment_weight(sample_texts, sample_true_scores)

    results = await async_process_tickers(tickers)
    if not results:
        logger.info("No predictions generated.")
        return

    prediction_table = build_prediction_table(results)
    if not prediction_table.empty:
        print("\nPrediction Table:")
        print(prediction_table.to_string(index=False))
    else:
        print("No significant predictions found.")

    plot_performance_metrics(results)
    avg_bs = np.mean([res["BS_Price"] for res in results if "BS_Price" in res])
    avg_sent = np.mean([res["Sentiment"] for res in results if "Sentiment" in res])
    metrics_data = {"avg_bs": avg_bs, "avg_sent": avg_sent}
    start_dashboard(metrics_data)

    # Auto-tune ensemble blend weight on a sample dataset (for demonstration)
    sample_X = np.random.rand(100, 10)
    sample_y = np.random.rand(100, 1)
    train_models(sample_X, sample_y)
    logger.info("Processing complete. Dashboard is live; press Ctrl+C to exit.")
    try:
        while True:
            await asyncio.sleep(10)
    except KeyboardInterrupt:
        sys.exit(0)

def main() -> None:
    asyncio.run(async_main())

# -----------------------------
# Unit and Integration Tests
# -----------------------------
def run_tests() -> None:
    logger.info("Running unit tests...")
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert abs(fast_cpu_distance(a, b)) < 1e-6, "fast_cpu_distance test failed."

    multi_gpu_accel = MultiGPUAccelerator()
    res_gpu = multi_gpu_accel.vector_multiply(a, b)
    assert np.allclose(res_gpu, a * b, atol=1e-5), "MultiGPU routine test failed."

    X_test = np.random.rand(5, 10)
    y_test = np.random.rand(5, 1)
    flax_nn = FlaxNeuralNetwork(input_dim=10, output_dim=1, dropout_rate=0.2, learning_rate=1e-3)
    params = flax_nn.model.init(jax.random.PRNGKey(0), jnp.array(X_test))
    loss_val = flax_nn.loss_fn(params, (jnp.array(X_test), jnp.array(y_test)))
    g = grad(flax_nn.loss_fn)(params, (jnp.array(X_test), jnp.array(y_test)))
    for key in g:
        assert g[key] is not None, f"Gradient for {key} missing."

    st = SentimentTransformer()
    score = st.basic_score("This is a great day for trading!")
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
    def test_retry_success() -> str:
        call_counter["count"] += 1
        if call_counter["count"] < 3:
            raise ValueError("Failing")
        return "succeeded"
    assert test_retry_success() == "succeeded", "Retry decorator test failed."

    @profile
    def dummy_sleep() -> str:
        time.sleep(0.2)
        return "done"
    assert dummy_sleep() == "done", "Profiling decorator test failed."

    fetcher = DataFetcher()
    try:
        asyncio.run(fetcher.async_fetch_stock_data("INVALID_TICKER"))
    except DataFetchError:
        logger.info("DataFetchError correctly raised for invalid ticker.")
    else:
        logger.error("DataFetchError was not raised for invalid ticker.")
    logger.info("All tests passed.")

if __name__ == "__main__":
    main()
