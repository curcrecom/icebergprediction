import os, sys, time, math, json, asyncio, logging, datetime, threading, argparse, io, pickle
import numpy as np, pandas as pd
from functools import wraps, partial, lru_cache
import concurrent.futures
import yfinance as yf  # used for historical stock data retrieval
import aiohttp
import sqlalchemy
from sqlalchemy import create_engine, Column, String, Float, Text
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.exc import SQLAlchemyError
from numba import njit, prange
import lightgbm as lgb
import optax
import jax
import jax.numpy as jnp
from jax import jit, grad, random, pmap, vmap
from flax import linen as nn
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
from bayes_opt import BayesianOptimization

# Fictitious module for AMD/Metal GPU acceleration; in a real system this should be implemented in C/C++ and wrapped in Python.
import metalcompute as mc  # assumed to be a custom low-level wrapper for Metal

# ---------------------------
# Global Configurations & Logging
# ---------------------------
CONFIG = {
    "LOG_FILE": "model_log.txt",
    "CACHE_DB": "stock_cache.db",
    "CACHE_EXPIRY": 86400,  # one day
    "DASH_PORT": 8050,
    "RUN_DAY": 4,  # Friday (0=Monday,...,4=Friday)
    "RUN_HOUR": 15,  # run after 15:00 local time on trading days
    "BATCH_SIZE": 32,
    "EPOCHS": 100,
    "CHECKPOINT_DIR": "./checkpoints"
}

if not os.path.exists(CONFIG["CHECKPOINT_DIR"]):
    os.makedirs(CONFIG["CHECKPOINT_DIR"])

# Configure structured JSON logging
from pythonjsonlogger import jsonlogger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = jsonlogger.JsonFormatter('%(asctime)s %(levelname)s %(name)s %(message)s')
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
logger.addHandler(ch)
fh = logging.handlers.RotatingFileHandler(CONFIG["LOG_FILE"], maxBytes=1*1024*1024, backupCount=5)
fh.setLevel(logging.DEBUG)
fh.setFormatter(formatter)
logger.addHandler(fh)

# ---------------------------
# Database Caching (SQLAlchemy)
# ---------------------------
Base = declarative_base()
class CacheEntry(Base):
    __tablename__ = 'cache'
    ticker = Column(String, primary_key=True)
    data = Column(Text)   # JSON string of historical data
    timestamp = Column(Float)
engine = create_engine(f"sqlite:///{CONFIG['CACHE_DB']}", echo=False)
SessionLocal = sessionmaker(bind=engine)
Base.metadata.create_all(bind=engine)
class CacheDB:
    def __init__(self):
        self.session = SessionLocal()
    def get(self, ticker: str) -> dict:
        try:
            entry = self.session.query(CacheEntry).filter(CacheEntry.ticker == ticker).first()
            if entry:
                return {"data": entry.data, "timestamp": entry.timestamp}
        except SQLAlchemyError as e:
            logger.error(f"CacheDB get error: {e}")
        return {}
    def set(self, ticker: str, data: str, timestamp: float) -> None:
        try:
            entry = self.session.query(CacheEntry).filter(CacheEntry.ticker == ticker).first()
            if entry:
                entry.data = data
                entry.timestamp = timestamp
            else:
                entry = CacheEntry(ticker=ticker, data=data, timestamp=timestamp)
                self.session.add(entry)
            self.session.commit()
        except SQLAlchemyError as e:
            logger.error(f"CacheDB set error: {e}")
            self.session.rollback()

# ---------------------------
# Low-level CPU and GPU Acceleration Classes
# ---------------------------
class GPUAccelerator:
    """GPU Accelerator using custom Metal kernel code for vector operations."""
    def __init__(self, device: mc.Device = None):
        self.device = device if device is not None else mc.Device()
        logger.info(f"GPUAccelerator initialized with device: {self.device.name}")
        # Low-level Metal kernel code in MSL for vector multiplication (example)
        self.kernel_code = """
        #include <metal_stdlib>
        using namespace metal;
        kernel void vec_mult(const device float* a [[ buffer(0) ]],
                             const device float* b [[ buffer(1) ]],
                             device float* result [[ buffer(2) ]],
                             constant uint &n [[ buffer(3) ]],
                             uint id [[ thread_position_in_grid ]]) {
            for (uint i = id; i < n; i += get_num_threads_per_grid()) {
                result[i] = a[i] * b[i];
            }
        }
        """
        self.compiled_kernel = self.device.kernel(self.kernel_code).function("vec_mult")
    def vector_multiply(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = np.ascontiguousarray(a.astype(np.float32))
        b = np.ascontiguousarray(b.astype(np.float32))
        n = a.shape[0]
        buf_bytes = n * 4
        a_bytes = memoryview(a)
        b_bytes = memoryview(b)
        result_buf = self.device.buffer(buf_bytes)
        self.compiled_kernel(n, a_bytes, b_bytes, result_buf)
        result_view = memoryview(result_buf).cast('f')
        return np.array(result_view)

@njit(fastmath=True, parallel=True)
def fast_cpu_distance(x: np.ndarray, y: np.ndarray) -> float:
    diff = x - y
    total = 0.0
    n = diff.shape[0]
    for i in prange(n):
        total += diff[i] * diff[i]
    return total

# ---------------------------
# Technical Indicator Functions
# ---------------------------
def wma(arr: np.ndarray, period: int) -> np.ndarray:
    """Weighted Moving Average"""
    weights = np.arange(1, period+1)
    result = np.full(arr.shape, np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.dot(arr[i - period + 1:i+1], weights) / weights.sum()
    return result

def dema(arr: np.ndarray, period: int) -> np.ndarray:
    """Double Exponential Moving Average"""
    ema = pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()
    ema_ema = pd.Series(ema).ewm(span=period, adjust=False).mean().to_numpy()
    return 2 * ema - ema_ema

def tema(arr: np.ndarray, period: int) -> np.ndarray:
    """Triple Exponential Moving Average"""
    ema = pd.Series(arr).ewm(span=period, adjust=False).mean().to_numpy()
    ema_ema = pd.Series(ema).ewm(span=period, adjust=False).mean().to_numpy()
    ema_ema_ema = pd.Series(ema_ema).ewm(span=period, adjust=False).mean().to_numpy()
    return 3 * (ema - ema_ema) + ema_ema_ema

def ssma(arr: np.ndarray, period: int) -> np.ndarray:
    """Smoothed Simple Moving Average"""
    result = np.full(arr.shape, np.nan)
    result[0] = arr[0]
    alpha = 1/period
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i-1]
    return result

def lwma(arr: np.ndarray, period: int) -> np.ndarray:
    """Linear Weighted Moving Average"""
    weights = np.linspace(1, period, period)
    result = np.full(arr.shape, np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.dot(arr[i - period + 1:i+1], weights) / weights.sum()
    return result

def vwma(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """Volume Weighted Moving Average"""
    cum_vol = np.cumsum(volume)
    cum_pv = np.cumsum(close * volume)
    return cum_pv / cum_vol

def hma(arr: np.ndarray, period: int) -> np.ndarray:
    """Hull Moving Average"""
    half_length = int(period/2)
    sqrt_length = int(math.sqrt(period))
    wma_half = wma(arr, half_length)
    wma_full = wma(arr, period)
    raw = 2 * wma_half - wma_full
    return wma(raw, sqrt_length)

def kama(arr: np.ndarray, period: int, fast: int = 2, slow: int = 30) -> np.ndarray:
    """Kaufman Adaptive Moving Average"""
    result = np.full(arr.shape, np.nan)
    change = np.abs(arr[period-1:] - arr[:-period+1])
    volatility = np.array([np.sum(np.abs(np.diff(arr[i-period+1:i+1]))) for i in range(period-1, len(arr))])
    er = np.divide(change, volatility, out=np.zeros_like(change), where=volatility!=0)
    sc = np.power((er * (2/(fast+1) - 2/(slow+1)) + 2/(slow+1)), 2)
    result[period-1] = arr[period-1]
    for i in range(period, len(arr)):
        result[i] = result[i-1] + sc[i-period+1]*(arr[i]-result[i-1])
    return result

def alma(arr: np.ndarray, period: int, offset: float = 0.85, sigma: float = 6) -> np.ndarray:
    """Arnaud Legoux Moving Average"""
    m = offset * (period - 1)
    s = period / sigma
    weights = np.array([math.exp(-((i - m)**2)/(2*s*s)) for i in range(period)])
    weights /= weights.sum()
    result = np.full(arr.shape, np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.dot(arr[i-period+1:i+1], weights)
    return result

def gma(arr: np.ndarray, period: int) -> np.ndarray:
    """Geometric Moving Average"""
    result = np.full(arr.shape, np.nan)
    for i in range(period - 1, len(arr)):
        result[i] = np.exp(np.mean(np.log(arr[i-period+1:i+1] + 1e-9)))
    return result

def macd(arr: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> (np.ndarray, np.ndarray):
    """MACD indicator and signal line"""
    ema_fast = pd.Series(arr).ewm(span=fast, adjust=False).mean().to_numpy()
    ema_slow = pd.Series(arr).ewm(span=slow, adjust=False).mean().to_numpy()
    macd_line = ema_fast - ema_slow
    signal_line = pd.Series(macd_line).ewm(span=signal, adjust=False).mean().to_numpy()
    return macd_line, signal_line

def stochastic_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, k_period: int = 14, d_period: int = 3) -> (np.ndarray, np.ndarray):
    """Stochastic Oscillator %K and %D lines"""
    k_values = np.full(close.shape, np.nan)
    for i in range(k_period - 1, len(close)):
        k_values[i] = 100 * (close[i] - np.min(low[i-k_period+1:i+1])) / (np.max(high[i-k_period+1:i+1]) - np.min(low[i-k_period+1:i+1]) + 1e-9)
    d_values = pd.Series(k_values).rolling(window=d_period).mean().to_numpy()
    return k_values, d_values

def adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Average Directional Index"""
    tr = np.maximum.reduce([high[1:] - low[1:], np.abs(high[1:] - close[:-1]), np.abs(low[1:] - close[:-1])])
    atr = pd.Series(tr).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    up = high[1:] - high[:-1]
    down = low[:-1] - low[1:]
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy() / (atr + 1e-9)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().to_numpy() / (atr + 1e-9)
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di + 1e-9)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().to_numpy()
    adx_full = np.concatenate((np.full(1, np.nan), adx))
    return adx_full

def cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 20) -> np.ndarray:
    """Commodity Channel Index"""
    tp = (high + low + close) / 3
    sma = pd.Series(tp).rolling(window=period).mean().to_numpy()
    mad = pd.Series(np.abs(tp - sma)).rolling(window=period).mean().to_numpy()
    return (tp - sma) / (0.015 * mad + 1e-9)

def roc(arr: np.ndarray, period: int = 12) -> np.ndarray:
    """Rate of Change"""
    roc_val = np.full(arr.shape, np.nan)
    roc_val[period:] = (arr[period:] - arr[:-period]) / (arr[:-period] + 1e-9) * 100
    return roc_val

def momentum(arr: np.ndarray, period: int = 10) -> np.ndarray:
    """Momentum Indicator"""
    mom = np.full(arr.shape, np.nan)
    mom[period:] = arr[period:] - arr[:-period]
    return mom

def williams_r(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """Williams %R"""
    wr = np.full(close.shape, np.nan)
    for i in range(period - 1, len(close)):
        wr[i] = -100 * (np.max(high[i-period+1:i+1]) - close[i]) / (np.max(high[i-period+1:i+1]) - np.min(low[i-period+1:i+1]) + 1e-9)
    return wr

def chaikin_oscillator(high: np.ndarray, low: np.ndarray, close: np.ndarray, volume: np.ndarray, fast_period: int = 3, slow_period: int = 10) -> np.ndarray:
    """Chaikin Oscillator"""
    money_flow_multiplier = ((close - low) - (high - close)) / (high - low + 1e-9)
    money_flow_volume = money_flow_multiplier * volume
    adl = np.cumsum(money_flow_volume)
    fast_ema = pd.Series(adl).ewm(span=fast_period, adjust=False).mean().to_numpy()
    slow_ema = pd.Series(adl).ewm(span=slow_period, adjust=False).mean().to_numpy()
    return fast_ema - slow_ema

def obv(close: np.ndarray, volume: np.ndarray) -> np.ndarray:
    """On Balance Volume"""
    obv_val = np.zeros(len(close))
    for i in range(1, len(close)):
        if close[i] > close[i-1]:
            obv_val[i] = obv_val[i-1] + volume[i]
        elif close[i] < close[i-1]:
            obv_val[i] = obv_val[i-1] - volume[i]
        else:
            obv_val[i] = obv_val[i-1]
    return obv_val

# ---------------------------
# Hidden Markov Model (HMM) for Regime Detection (simplified custom implementation)
# ---------------------------
def detect_regime_hmm(returns: np.ndarray, n_states: int = 2, n_iter: int = 10) -> np.ndarray:
    """
    A simple Gaussian HMM using the Expectation–Maximization (EM) algorithm.
    Returns an array of state assignments for each return.
    Note: In production, one might use a dedicated package (e.g., hmmlearn)
    but here we include a custom implementation.
    """
    T = len(returns)
    # Initialize state means, variances, and transition matrix randomly
    means = np.linspace(np.mean(returns) - np.std(returns), np.mean(returns) + np.std(returns), n_states)
    variances = np.full(n_states, np.var(returns))
    transmat = np.full((n_states, n_states), 1/n_states)
    # Uniform initial state probabilities
    pi = np.full(n_states, 1/n_states)
    # Responsibilities (gamma)
    gamma = np.zeros((T, n_states))
    
    for iteration in range(n_iter):
        # E-step: compute gamma[t,k] = P(state=k|return[t])
        for t in range(T):
            obs_prob = np.array([ (1/math.sqrt(2*math.pi*v)) * math.exp(-0.5*((returns[t]-m)**2)/v)
                                   for m, v in zip(means, variances)])
            gamma[t] = pi * obs_prob
            gamma[t] /= (np.sum(gamma[t]) + 1e-9)
        # M-step: update parameters
        for k in range(n_states):
            weight = np.sum(gamma[:,k])
            means[k] = np.sum(gamma[:,k]*returns) / (weight + 1e-9)
            variances[k] = np.sum(gamma[:,k]*(returns - means[k])**2) / (weight + 1e-9)
        # Update pi using the first observation gamma
        pi = gamma[0]
        # Update transition matrix (here we use a simple re–estimation)
        for i in range(n_states):
            for j in range(n_states):
                numer = 0.0
                denom = 0.0
                for t in range(T-1):
                    numer += gamma[t, i] * gamma[t+1, j]
                    denom += gamma[t, i]
                transmat[i,j] = numer / (denom + 1e-9)
        # Normalize rows of transmat
        transmat = transmat / (transmat.sum(axis=1, keepdims=True) + 1e-9)
    # Viterbi decoding (very simplified: choose state with max responsibility at each t)
    state_seq = np.argmax(gamma, axis=1)
    return state_seq

# ---------------------------
# GARCH(1,1) Volatility Estimation
# ---------------------------
def estimate_garch(returns: np.ndarray, omega: float = 1e-6, alpha: float = 0.05, beta: float = 0.9) -> np.ndarray:
    """Estimate conditional volatility using a basic GARCH(1,1) model."""
    T = len(returns)
    sigma2 = np.zeros(T)
    sigma2[0] = np.var(returns)
    for t in range(1, T):
        sigma2[t] = omega + alpha*(returns[t-1]**2) + beta*sigma2[t-1]
    return np.sqrt(sigma2)

# ---------------------------
# Option Pricing Models
# ---------------------------
def black_scholes(S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
    """Black-Scholes option pricing formula."""
    d1 = (math.log(S/K) + (r + 0.5*sigma*sigma)*T) / (sigma*math.sqrt(T) + 1e-12)
    d2 = d1 - sigma*math.sqrt(T)
    N = lambda x: 0.5*(1+math.erf(x/math.sqrt(2)))
    if option_type.lower() == 'call':
        return S * N(d1) - K * math.exp(-r*T) * N(d2)
    else:
        return K * math.exp(-r*T) * N(-d2) - S * N(-d1)

def binomial_option(S: float, K: float, T: float, r: float, sigma: float, steps: int = 100, option_type: str = 'call') -> float:
    """Binomial option pricing."""
    dt = T / steps
    u = math.exp(sigma*math.sqrt(dt))
    d = 1/u
    p = (math.exp(r*dt) - d) / (u - d)
    # initialize asset prices at maturity
    asset_prices = np.array([S * (u**j) * (d**(steps-j)) for j in range(steps+1)])
    if option_type.lower() == 'call':
        option_values = np.maximum(asset_prices - K, 0)
    else:
        option_values = np.maximum(K - asset_prices, 0)
    discount = math.exp(-r*dt)
    for i in range(steps, 0, -1):
        option_values = discount * (p * option_values[1:i+1] + (1-p) * option_values[0:i])
    return option_values[0]

def monte_carlo_option(S: float, K: float, T: float, r: float, sigma: float,
                       simulations: int = 100000, option_type: str = 'call') -> float:
    """Monte Carlo simulation for option pricing using GPU parallelism."""
    key = random.PRNGKey(int(time.time()))
    def sim_fn(key):
        rand = random.normal(key, (10000,))
        ST = S * jnp.exp((r - 0.5*sigma**2)*T + sigma*jnp.sqrt(T)*rand)
        if option_type.lower() == 'call':
            payoff = jnp.maximum(ST-K, 0)
        else:
            payoff = jnp.maximum(K-ST, 0)
        return jnp.mean(payoff)
    keys = random.split(key, 10)
    payoffs = vmap(sim_fn)(jnp.array(keys))
    price = jnp.exp(-r*T) * jnp.mean(payoffs)
    return float(price)

def jump_diffusion_option(S: float, K: float, T: float, r: float, sigma: float,
                          lam: float = 0.1, muJ: float = 0, sigmaJ: float = 0.1, option_type: str = 'call') -> float:
    """Option pricing with jump diffusion model."""
    price = 0.0
    for k in range(50):
        poisson_prob = math.exp(-lam*T) * (lam*T)**k / math.factorial(k)
        sigma_k = math.sqrt(sigma**2 + (k * sigmaJ**2)/T)
        price += poisson_prob * black_scholes(S, K, T, r, sigma_k, option_type)
    return price

# ---------------------------
# Custom Neural Network with Sinusoidal Representation Layers using Flax
# ---------------------------
class SinusoidalDense(nn.Module):
    features: int
    kernel_init: callable = nn.initializers.lecun_normal()
    @nn.compact
    def __call__(self, x):
        # linear transformation followed by sine activation
        dense = nn.Dense(self.features, kernel_init=self.kernel_init, use_bias=True)
        x = dense(x)
        return jnp.sin(x)

class SRNModel(nn.Module):
    hidden_layers: list
    output_dim: int
    dropout_rate: float = 0.0
    @nn.compact
    def __call__(self, x, train: bool = True):
        # Instead of standard dense layers, use sinusoidal dense layers
        for h in self.hidden_layers:
            x = SinusoidalDense(features=h)(x)
            x = nn.relu(x)  # additional nonlinearity if desired
            if self.dropout_rate > 0.0:
                x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not train)
        return nn.Dense(self.output_dim)(x)

class FlaxNeuralNetwork:
    """Flax NN with SRN layers for regression, including early stopping and checkpointing."""
    def __init__(self, input_dim: int, output_dim: int, hidden_layers: list[int] = None,
                 dropout_rate: float = 0.1, learning_rate: float = 1e-3):
        if hidden_layers is None:
            hidden_layers = [128, 128]
        self.model = SRNModel(hidden_layers=hidden_layers, output_dim=output_dim, dropout_rate=dropout_rate)
        self.learning_rate = learning_rate
        self.params = None
        self.tx = optax.adam(learning_rate)
        self.opt_state = None
        self.best_loss = np.inf
        self.checkpoint_path = os.path.join(CONFIG["CHECKPOINT_DIR"], "flax_model.chkpt")
    def init(self, rng: jax.random.PRNGKey, sample_input: jnp.ndarray) -> None:
        self.params = self.model.init(rng, sample_input)
        self.opt_state = self.tx.init(self.params)
    @jax.jit
    def loss_fn(self, params, batch):
        X, y = batch
        preds = self.model.apply(params, X)
        return jnp.mean((preds - y)**2)
    @jax.jit
    def train_step(self, params, opt_state, batch):
        loss, grads = jax.value_and_grad(self.loss_fn)(params, batch)
        updates, opt_state = self.tx.update(grads, opt_state)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = CONFIG["EPOCHS"], batch_size: int = CONFIG["BATCH_SIZE"],
              early_stop_patience: int = 10) -> list[float]:
        X = X.astype(np.float32)
        y = y.astype(np.float32)
        num_samples = X.shape[0]
        losses = []
        rng = jax.random.PRNGKey(int(time.time()))
        self.init(rng, jnp.array(X[:batch_size]))
        patience = 0
        for epoch in range(epochs):
            perm = np.random.permutation(num_samples)
            epoch_loss = 0.0
            for i in range(0, num_samples, batch_size):
                idx = perm[i:i+batch_size]
                batch = (jnp.array(X[idx]), jnp.array(y[idx]))
                self.params, self.opt_state, loss = self.train_step(self.params, self.opt_state, batch)
                epoch_loss += float(loss)
            epoch_loss /= (num_samples / batch_size)
            losses.append(epoch_loss)
            logger.info(f"Epoch {epoch+1}/{epochs}: Loss={epoch_loss:.6f}")
            # Early stopping and checkpointing
            if epoch_loss < self.best_loss:
                self.best_loss = epoch_loss
                patience = 0
                self.save_checkpoint()
            else:
                patience += 1
            if patience >= early_stop_patience:
                logger.info("Early stopping triggered.")
                break
        return losses
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X.astype(np.float32)
        return np.array(self.model.apply(self.params, jnp.array(X)))
    def save_checkpoint(self) -> None:
        from flax.serialization import to_bytes
        with open(self.checkpoint_path, "wb") as f:
            f.write(to_bytes(self.params))
        logger.info("Flax model checkpoint saved.")
    def load_checkpoint(self) -> bool:
        from flax.serialization import from_bytes
        if os.path.exists(self.checkpoint_path):
            with open(self.checkpoint_path, "rb") as f:
                self.params = from_bytes(self.model.init(jax.random.PRNGKey(0), jnp.ones((1, self.model.hidden_layers[0]))), f.read())
            logger.info("Flax model checkpoint loaded.")
            return True
        return False

# ---------------------------
# Ensemble Prediction with LightGBM and Flax NN
# ---------------------------
def train_lightgbm(X_train: np.ndarray, y_train: np.ndarray) -> any:
    lgb_train = lgb.Dataset(X_train, label=y_train.ravel())
    params = {'objective': 'regression', 'metric': 'rmse', 'verbosity': -1}
    model = lgb.train(params, lgb_train, num_boost_round=100)
    return model

def ensemble_predict(X: np.ndarray, flax_nn: FlaxNeuralNetwork, lgb_model: any, blend_weight: float) -> np.ndarray:
    pred_nn = flax_nn.predict(X)
    pred_lgb = lgb_model.predict(X)
    return blend_weight * pred_nn + (1 - blend_weight) * pred_lgb

def optimize_ensemble_weight(X: np.ndarray, y: np.ndarray) -> float:
    def objective(w):
        w = float(w)
        flax_nn = FlaxNeuralNetwork(input_dim=X.shape[1], output_dim=1)
        flax_nn.train(X, y, epochs=10, batch_size=16)
        lgb_model = train_lightgbm(X, y)
        preds = ensemble_predict(X, flax_nn, lgb_model, blend_weight=w)
        rmse = np.sqrt(np.mean((preds - y.ravel())**2))
        return -rmse
    optimizer = BayesianOptimization(f=objective, pbounds={'w': (0.0, 1.0)}, random_state=42, verbose=0)
    optimizer.maximize(init_points=5, n_iter=10)
    best_w = optimizer.max['params']['w']
    logger.info(f"Optimal ensemble blend weight: {best_w}")
    return best_w

# ---------------------------
# Data Retrieval and Feature Engineering
# ---------------------------
async def fetch_stock_data(ticker: str, period: str = '5y') -> pd.DataFrame:
    """
    Retrieve historical data via yfinance.
    Uses caching via SQLite to reduce API calls.
    """
    cache = CacheDB()
    now = time.time()
    cached = cache.get(ticker)
    if cached and (now - cached.get("timestamp", 0) < CONFIG["CACHE_EXPIRY"]):
        logger.info(f"Loading cached data for {ticker}")
        try:
            df = pd.read_json(io.StringIO(cached["data"]), convert_dates=True)
            return df
        except Exception as e:
            logger.error(f"Cache read error for {ticker}: {e}")
    logger.info(f"Fetching data for {ticker} from yfinance")
    df = yf.download(ticker, period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for ticker {ticker}")
    # Ensure multi-level column indexes if needed (for OHLCV)
    if not isinstance(df.columns, pd.MultiIndex):
        df.columns = pd.MultiIndex.from_product([df.columns, ['']])
    # Save to cache
    cache.set(ticker, df.to_json(date_format='iso'), now)
    return df

def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators and additional features."""
    # Assume df columns are MultiIndex: level 0 = variable name, level 1 = extra info
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    volume = df['Volume'].values
    features = pd.DataFrame(index=df.index)
    features['WMA'] = wma(close, period=20)
    features['DEMA'] = dema(close, period=20)
    features['TEMA'] = tema(close, period=20)
    features['SSMA'] = ssma(close, period=20)
    features['LWMA'] = lwma(close, period=20)
    features['VWMA'] = vwma(close, volume)
    features['HMA'] = hma(close, period=20)
    features['KAMA'] = kama(close, period=20)
    features['ALMA'] = alma(close, period=20)
    features['GMA'] = gma(close, period=20)
    macd_line, signal_line = macd(close)
    features['MACD'] = macd_line - signal_line
    features['Stochastic_K'], features['Stochastic_D'] = stochastic_oscillator(high, low, close)
    features['ADX'] = adx(high, low, close)
    features['CCI'] = cci(high, low, close)
    features['ROC'] = roc(close)
    features['Momentum'] = momentum(close)
    features['Williams_%R'] = williams_r(high, low, close)
    features['Chaikin'] = chaikin_oscillator(high, low, close, volume)
    features['OBV'] = obv(close, volume)
    features = features.fillna(method="ffill").fillna(method="bfill")
    return features

def augment_features(features: pd.DataFrame) -> pd.DataFrame:
    """Bootstrap augmentation with rolling windows and added Gaussian noise."""
    noise = np.random.normal(0, 0.001, size=features.shape)
    augmented = features + noise
    return augmented

def reduce_features(features: pd.DataFrame, n_components: int = 10) -> pd.DataFrame:
    """Remove redundant features via SHAP-based selection and PCA (using SVD)."""
    X = features.values
    X_centered = X - np.mean(X, axis=0)
    # SVD for PCA reduction
    U, s, _ = np.linalg.svd(X_centered, full_matrices=False)
    X_reduced = U[:, :n_components] * s[:n_components]
    return pd.DataFrame(X_reduced, index=features.index)

# ---------------------------
# Ensemble Model Training on Historical Data
# ---------------------------
async def train_ensemble_models(tickers: list[str]) -> (FlaxNeuralNetwork, any, float):
    tasks = [fetch_stock_data(ticker) for ticker in tickers]
    datasets = await asyncio.gather(*tasks, return_exceptions=True)
    X_list, y_list = [], []
    for df in datasets:
        if isinstance(df, Exception) or df.empty:
            continue
        df = df.sort_index()
        # Define target: 4-day forward return (for Friday close prediction)
        target = (df['Close'].shift(-4) / df['Close'] - 1).dropna()
        feats = compute_features(df)
        feats = augment_features(feats)
        feats = reduce_features(feats)
        # Align features with target
        common_index = feats.index.intersection(target.index)
        X_list.append(feats.loc[common_index].values)
        y_list.append(target.loc[common_index].values.reshape(-1, 1))
    if not X_list:
        raise ValueError("No valid training data obtained.")
    X_train = np.concatenate(X_list, axis=0)
    y_train = np.concatenate(y_list, axis=0)
    flax_nn = FlaxNeuralNetwork(input_dim=X_train.shape[1], output_dim=1, dropout_rate=0.1)
    flax_nn.train(X_train, y_train, epochs=20, batch_size=32)
    lgb_model = train_lightgbm(X_train, y_train)
    blend_weight = optimize_ensemble_weight(X_train, y_train)
    return flax_nn, lgb_model, blend_weight

# ---------------------------
# Asynchronous Prediction per Ticker
# ---------------------------
async def process_ticker(ticker: str) -> dict:
    """
    Process a ticker: fetch historical data, extract features,
    run ensemble prediction, and compute option prices and greeks.
    """
    try:
        df = await fetch_stock_data(ticker)
        if df.empty:
            raise ValueError("Empty data")
        df = df.sort_index()
        feats = compute_features(df)
        feats = augment_features(feats)
        feats = reduce_features(feats)
        # Use most recent row for prediction
        X_pred = feats.tail(1).values
        # Load persistent models if available; otherwise, train ensemble models
        flax_nn = FlaxNeuralNetwork(input_dim=X_pred.shape[1], output_dim=1)
        if not flax_nn.load_checkpoint():
            # If no checkpoint, train on a list of known tickers (for demonstration we use [ticker])
            flax_nn, lgb_model, blend_weight = await train_ensemble_models([ticker])
        else:
            lgb_model = load_lightgbm_model()  # you would have a similar persistent load for lgb
            blend_weight = 0.5  # fallback blend weight
        prediction = float(ensemble_predict(X_pred, flax_nn, lgb_model, blend_weight)[0])
        latest_close = float(df['Close'].iloc[-1])
        strike = latest_close  # for ATM option
        r_val = 0.01  # risk free rate placeholder; in production update dynamically
        T_val = 4/252  # 4 trading days ahead
        sigma_val = float(np.std(np.log(df['Close']/df['Close'].shift(1)).dropna()) * math.sqrt(252))
        # Option pricing
        bs_price = black_scholes(latest_close, strike, T_val, r_val, sigma_val, "call")
        binom_price = binomial_option(latest_close, strike, T_val, r_val, sigma_val, steps=100, option_type="call")
        mc_price = monte_carlo_option(latest_close, strike, T_val, r_val, sigma_val, simulations=10000, option_type="call")
        # Greeks (simplified; in production use analytic or AD methods)
        delta = (black_scholes(latest_close+0.01, strike, T_val, r_val, sigma_val, "call") - bs_price) / 0.01
        gamma = (black_scholes(latest_close+0.01, strike, T_val, r_val, sigma_val, "call") +
                 black_scholes(latest_close-0.01, strike, T_val, r_val, sigma_val, "call") - 2*bs_price) / (0.01**2)
        theta = - (bs_price / T_val)  # very approximate
        vega = (black_scholes(latest_close, strike, T_val, r_val, sigma_val+0.01, "call") - bs_price) / 0.01
        # Regime detection using HMM and GARCH on returns
        returns = np.log(df['Close']/df['Close'].shift(1)).dropna().values
        regime = int(detect_regime_hmm(returns)[-1])
        vol_garch = estimate_garch(returns)[-1]
        sentiment = await fetch_news_sentiment(ticker)  # see next section
        result = {
            "Ticker": ticker,
            "Date": df.index[-1].strftime("%Y-%m-%d"),
            "LatestClose": latest_close,
            "Strike": strike,
            "Prediction": prediction,  # forecasted 4-day return
            "BS_Price": bs_price,
            "Binom_Price": binom_price,
            "MC_Price": mc_price,
            "Delta": delta,
            "Gamma": gamma,
            "Theta": theta,
            "Vega": vega,
            "Regime": regime,
            "GARCH_vol": vol_garch,
            "Sentiment": sentiment
        }
        logger.info(f"Processed ticker {ticker}: {result}")
        return result
    except Exception as e:
        logger.exception(f"Error processing ticker {ticker}: {e}")
        return {}

async def fetch_news_sentiment(ticker: str) -> float:
    """Fetch news sentiment by scraping several websites asynchronously and scoring text using FinBERT."""
    urls = [
        f"https://www.businesswire.com/portal/site/home/?search={ticker}",
        f"https://www.reuters.com/search/news?blob={ticker}",
        f"https://www.bloomberg.com/search?query={ticker}"
    ]
    scores = []
    sentiment_pipeline = pipeline("sentiment-analysis", model="yiyanghkust/finbert-pretrain", tokenizer="yiyanghkust/finbert-pretrain")
    async with aiohttp.ClientSession() as session:
        async def fetch_and_score(url: str):
            try:
                async with session.get(url, timeout=10) as response:
                    if response.status == 200:
                        text = await response.text()
                        # Use BeautifulSoup to extract text paragraphs
                        from bs4 import BeautifulSoup
                        soup = BeautifulSoup(text, 'html.parser')
                        content = " ".join([p.get_text() for p in soup.find_all('p')])
                        result = sentiment_pipeline(content[:512])
                        if result and result[0]['label'].upper() == "POSITIVE":
                            scores.append(result[0]['score'])
                        elif result and result[0]['label'].upper() == "NEGATIVE":
                            scores.append(-result[0]['score'])
            except Exception as ex:
                logger.error(f"News fetch error {url}: {ex}")
        await asyncio.gather(*(fetch_and_score(url) for url in urls))
    if scores:
        return float(np.mean(scores))
    return 0.0

def load_lightgbm_model() -> any:
    """Load LightGBM model from disk if exists."""
    model_path = "lgb_model.txt"
    if os.path.exists(model_path):
        model = lgb.Booster(model_file=model_path)
        logger.info("LightGBM model loaded from disk.")
        return model
    return None

def save_lightgbm_model(lgb_model: any) -> None:
    model_path = "lgb_model.txt"
    lgb_model.save_model(model_path)
    logger.info("LightGBM model saved to disk.")

# ---------------------------
# Build Prediction Table and Plotting
# ---------------------------
def build_prediction_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for res in results:
        if res and res.get("Prediction") is not None:
            rows.append({
                "Date": res.get("Date"),
                "Ticker": res.get("Ticker"),
                "Strike": res.get("Strike"),
                "Predicted 4-day Return (%)": res.get("Prediction") * 100,
                "BS Price": res.get("BS_Price"),
                "Sentiment": res.get("Sentiment")
            })
    return pd.DataFrame(rows)

def plot_performance(results: list[dict]) -> None:
    import matplotlib.pyplot as plt
    tickers = [res["Ticker"] for res in results if res]
    bs_prices = [res["BS_Price"] for res in results if res]
    sentiments = [res["Sentiment"] for res in results if res]
    plt.figure(figsize=(10, 6))
    plt.scatter(sentiments, bs_prices, color='blue')
    plt.xlabel("News Sentiment")
    plt.ylabel("Black-Scholes Price")
    plt.title("Performance Metrics per Ticker")
    plt.grid(True)
    plt.savefig("performance_metrics.png")
    plt.close()
    logger.info("Performance plot saved to performance_metrics.png.")

# ---------------------------
# Main Execution and User Interaction
# ---------------------------
async def async_main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tickers", type=str, help="Comma-separated stock tickers (e.g., AAPL,MSFT,GOOG)")
    parser.add_argument("--test", action="store_true", help="Run tests and exit")
    args = parser.parse_args()

    if args.test:
        run_tests()
        return

    # Dynamic prompt if tickers not provided via command-line
    if args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    else:
        tickers_input = input("Enter comma-separated tickers: ")
        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
    if not tickers:
        logger.error("No valid tickers provided. Exiting.")
        sys.exit(1)

    # Check trading schedule: run only if today is Friday and after RUN_HOUR
    now = datetime.datetime.now()
    if now.weekday() != CONFIG["RUN_DAY"] or now.hour < CONFIG["RUN_HOUR"]:
        logger.error("This script is scheduled to run on Friday after the set hour. Exiting.")
        sys.exit(1)

    # Process tickers concurrently using asyncio and multiprocessing
    tasks = [process_ticker(ticker) for ticker in tickers]
    results = await asyncio.gather(*tasks)
    results = [res for res in results if res]
    if not results:
        logger.info("No predictions generated. Likely due to data retrieval issues.")
        return
    prediction_table = build_prediction_table(results)
    if not prediction_table.empty:
        print("\nPrediction Table:")
        print(prediction_table.to_string(index=False))
    else:
        print("No significant predictions found.")
    plot_performance(results)
    logger.info("Processing complete.")

def run_tests():
    """Unit tests for critical components."""
    logger.info("Running unit tests...")
    # Test low-level CPU function
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([1.0, 2.0, 3.0])
    assert abs(fast_cpu_distance(a, b)) < 1e-6, "fast_cpu_distance failed"
    # Test technical indicators (WMA, DEMA, etc.)
    arr = np.linspace(1, 100, 100)
    assert not np.isnan(wma(arr, 20)).all(), "WMA failed"
    assert not np.isnan(dema(arr, 20)).all(), "DEMA failed"
    # Test option pricing models
    bs = black_scholes(100, 100, 1, 0.05, 0.2)
    binom = binomial_option(100, 100, 1, 0.05, 0.2)
    mc = monte_carlo_option(100, 100, 1, 0.05, 0.2)
    assert abs(bs - binom) < 5, "Option pricing discrepancy"
    # Test HMM regime detection
    r = np.random.normal(0, 0.01, 100)
    regime = detect_regime_hmm(r)
    assert regime.shape[0] == 100, "HMM regime detection failed"
    logger.info("All tests passed.")

def main():
    asyncio.run(async_main())

if __name__ == "__main__":
    main()
