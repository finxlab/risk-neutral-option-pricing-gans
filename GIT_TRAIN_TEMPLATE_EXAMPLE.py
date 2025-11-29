"""

An example script organized into clear stages: preprocessing -> model definition ->
model load/save -> training -> inference (generation).

Usage summary:
 - Load configuration from `configs/config.yaml` and use the `config` values.
 - Use `load_time_series` to read CSV and perform preprocessing .
 - Use `create_windows` to create rolling-window tensors for training.
 - Use `get_model` to obtain Generator/Discriminator and `train_model` to train.
 - Use `generate_samples` to produce many samples and inverse-scale them.

Reference license notes (one-line each — verify LICENSE in the original repo before reuse):
 - TimeGAN (jsyoon0823/TimeGAN): see the repository LICENSE for terms; verify before reuse.
 - QuantGAN / temporalCN (JamesSullivan/temporalCN): see the repository LICENSE for terms; verify before reuse.
 - SigCWGAN (SigCGANs/Conditional-Sig-Wasserstein-GANs): see the repository LICENSE for terms; verify before reuse.

"""

import os
import yaml
import pickle
from typing import Tuple, Optional

import ml_collections
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

import random
from numpy.lib.stride_tricks import sliding_window_view


def set_seed(seed: int):
    """Set random seeds for reproducibility (numpy, random, torch)."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def scaling(values: np.ndarray):
    """Scale time-series values per feature (zero mean, unit std).

    Args:
        values: array shape (T, n_vars)

    Returns:
        scaled: np.ndarray same shape as `values` (float32)
        scalers: dict with keys 'mean' and 'std' (each shape (n_vars,))
    """
    vals = np.asarray(values, dtype=float)
    mean = vals.mean(axis=0)
    std = vals.std(axis=0)
    std_safe = np.where(std <= 0, 1.0, std)
    scaled = (vals - mean) / std_safe
    scalers = {'mean': mean, 'std': std_safe}
    return scaled, scalers


def inverse_scaling(scaled, scalers: dict) -> np.ndarray:
    """Inverse the scaling applied by `scaling`.

    Accepts either a torch.Tensor or numpy array. Returns a numpy array with
    shape preserved.
    """
    # Convert torch tensors to numpy
    try:
        import torch

        if isinstance(scaled, torch.Tensor):
            arr = scaled.detach().cpu().numpy()
        else:
            arr = np.asarray(scaled)
    except Exception:
        arr = np.asarray(scaled)

    mean = np.asarray(scalers['mean'])
    std = np.asarray(scalers['std'])
    # expected shapes: arr (batch, n_vars, n_steps) or (T, n_vars)
    # reshape mean/std to broadcast over (batch, n_vars, n_steps)
    if arr.ndim == 3:
        mean_r = mean.reshape(1, -1, 1)
        std_r = std.reshape(1, -1, 1)
    elif arr.ndim == 2:
        mean_r = mean.reshape(1, -1)
        std_r = std.reshape(1, -1)
    else:
        # fallback: try to broadcast
        mean_r = mean
        std_r = std
    return arr * std_r + mean_r


def rolling_window(values: np.ndarray, n_steps: int) -> np.ndarray:
    """Create rolling windows over the time axis.

    Args:
        values: array shape (T, n_vars)
        n_steps: window length

    Returns:
        windows: np.ndarray shape (num_windows, n_vars, n_steps)
    """
    arr = np.asarray(values)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    # sliding_window_view gives shape (T - n_steps + 1, n_vars, n_steps)
    win = sliding_window_view(arr, window_shape=(n_steps,), axis=0)
    # win shape: (num_windows, 1, n_vars, n_steps) depending on numpy version
    # normalize to (num_windows, n_vars, n_steps)
    if win.ndim == 3:
        # newer numpy: (num_windows, n_vars, n_steps)
        windows = win
    else:
        # older: squeeze the extra axis
        windows = np.squeeze(win, axis=1)
    return windows.astype(np.float32)


def compute_avg_emd(real_data: np.ndarray, fake_data: np.ndarray, window: int = 100) -> float:
    """Compute a simple average EMD-like distance between real and fake.

    This is a lightweight approximation: for each variable we compute rolling
    window sums and compare the empirical distributions by sorting their
    quantiles and taking mean absolute difference. This is not a perfect EMD
    implementation but serves as a useful proxy for distributional distance.
    """
    real = np.asarray(real_data)
    fake = np.asarray(fake_data)
    # Expect shape (num_samples, n_vars, n_steps)
    if real.ndim != 3 or fake.ndim != 3:
        raise ValueError("real_data and fake_data must have shape (N, n_vars, n_steps)")
    n_vars = real.shape[1]
    dists = []
    for j in range(n_vars):
        real_series = real[:, j, :]
        fake_series = fake[:, j, :]
        # compute rolling-window sums along time axis for both
        try:
            real_windows = rolling_window(real_series.T, window).sum(axis=1).ravel()
            fake_windows = rolling_window(fake_series.T, window).sum(axis=1).ravel()
        except Exception:
            # fallback: aggregate whole series
            real_windows = real_series.sum(axis=1)
            fake_windows = fake_series.sum(axis=1)

        # compute comparable quantiles
        m = min(len(real_windows), len(fake_windows))
        if m == 0:
            continue
        qs = np.linspace(0, 1, m)
        r_q = np.quantile(real_windows, qs)
        f_q = np.quantile(fake_windows, qs)
        d = np.mean(np.abs(r_q - f_q))
        dists.append(d)
    if len(dists) == 0:
        return float('inf')
    return float(np.mean(dists))

# Try to import network/trainer classes from the reference implementation.
try:
    from src.baselines.networks.generators import UserGenerator
    from src.baselines.networks.discriminators import UserDiscriminator
    from src.baselines.trainer import GANTrainer
except Exception:
    # If not available, set to None and expect the caller to provide classes.
    UserGenerator = None
    UserDiscriminator = None
    GANTrainer = None


def load_config(config_path: str) -> ml_collections.ConfigDict:
    """Load a config file and return as a `ml_collections.ConfigDict`."""
    with open(config_path) as f:
        cfg = ml_collections.ConfigDict(yaml.safe_load(f))
    return cfg


def load_time_series(csv_path: str) -> pd.DataFrame:
    """Read a CSV file, drop NA and convert columns to numeric types.

    If a `Date` column exists, convert it to datetime and set it as the index.
    """
    df = pd.read_csv(csv_path)
    df = df.replace(0, np.nan).dropna()
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
        df.set_index('Date', inplace=True)
    df = df.apply(pd.to_numeric).astype(float)
    return df


def create_windows(values: np.ndarray, n_steps: int) -> torch.Tensor:
    """Create rolling windows from a numpy array and return as a torch Tensor.

    Returned shape is (num_samples, n_vars, n_steps).
    """
    windows = rolling_window(values, n_steps)
    return torch.from_numpy(windows).float()


def get_model(config: ml_collections.ConfigDict, generator_cls=None, discriminator_cls=None):
    """Return instances of Generator and Discriminator.

    If `generator_cls`/`discriminator_cls` are provided they are used; otherwise
    attempt to use `UserGenerator`/`UserDiscriminator` from the repository.

    References (for implementation inspiration only — not drop-in code):
    - TimeGAN: https://github.com/jsyoon0823/TimeGAN
    - QuantGAN / temporalCN: https://github.com/JamesSullivan/temporalCN
    - SigCWGAN: https://github.com/SigCGANs/Conditional-Sig-Wasserstein-GANs

    Notes:
    - The above links are references/inspiration, not plug-and-play modules.    
    - If you use code from those repos, ensure the model's input/output API
        matches this pipeline (e.g., `G(noise)` should accept
        `(batch, noise_dim, n_steps)` and return `(batch, n_vars, n_steps)`),
        or adapt/wrap the implementation accordingly.
    - Prefer passing `generator_cls`/`discriminator_cls` explicitly or add
        compatible implementations under `src.baselines`.
    """
    gen_cls = generator_cls or UserGenerator
    dis_cls = discriminator_cls or UserDiscriminator
    if gen_cls is None or dis_cls is None:
        raise RuntimeError("No generator/discriminator class available. Provide `generator_cls`/`discriminator_cls` or add implementations under `src.baselines`.")
    G = gen_cls(config)
    D = dis_cls(config)
    return G, D


def save_model(model: torch.nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)


def load_model(model: torch.nn.Module, path: str, device: Optional[torch.device] = None):
    model.load_state_dict(torch.load(path, map_location=device))
    if device is not None:
        model.to(device)
    model.eval()


def train_model(G: torch.nn.Module, D: torch.nn.Module, train_dl: DataLoader, config: ml_collections.ConfigDict):
    """Train the GAN using `GANTrainer`.

    Raises an error if `GANTrainer` is not available in the environment.
    """
    if GANTrainer is None:
        raise RuntimeError("GANTrainer is not available. Provide a trainer or add src.baselines.trainer.")
    trainer = GANTrainer(G=G, D=D, train_dl=train_dl, config=config)
    trainer.fit()
    return trainer


def generate_samples(generator: torch.nn.Module, scalers, total_batch_size: int, n_steps: int, chunk_size: int, device: torch.device, config: ml_collections.ConfigDict) -> np.ndarray:
    """Generate samples in chunks to avoid OOM, then inverse-scale and return.

    `total_batch_size` is the total number of samples desired. `chunk_size` is the
    maximum number generated at once. The returned array is the inverse-scaled
    numpy array of generated samples.

    `config` is required to obtain `config.noise_dim`. Alternatively, you can
    pass a generator that exposes the required input size.
    """
    generator.to(device)
    generator.eval()

    num_chunks = int(np.ceil(total_batch_size / chunk_size))
    results = []
    with torch.no_grad():
        for i in range(num_chunks):
            this_batch = chunk_size if (i < num_chunks - 1) else (total_batch_size - chunk_size * (num_chunks - 1))
            noise = torch.randn(this_batch, config.noise_dim, n_steps, device=device)
            fake_chunk = generator(noise)
            results.append(fake_chunk.cpu())

    fake = torch.cat(results, dim=0)
    fake_data = inverse_scaling(fake, scalers)
    return fake_data


def evaluate_generated(real_data: np.ndarray, fake_data: np.ndarray, window: int = 100) -> float:
    """Example evaluation: compute EMD-based distance using `compute_avg_emd`.

    Replace this with a different metric if required.
    """
    return compute_avg_emd(real_data, fake_data, window)


def main(config_path: str = 'configs/config.yaml'):
    # Load configuration and set random seed
    config = load_config(config_path)
    set_seed(config.seed)
    device = torch.device(config.device if hasattr(config, 'device') else 'cpu')

    # Example: iterate over multiple files (keeps the original notebook's pattern)
    for idx in range(0, 39):
        config.update({"asset": "samples"}, allow_val_change=True)
        config.update({"file_name": f"train{idx}"}, allow_val_change=True)

        # ---------------------------
        # 1) Preprocessing
        # ---------------------------
        csv_path = f"./data/trainset/{config.file_name}.csv"
        df = load_time_series(csv_path)
        values = df.values

        # Scaling (returns scaled values and fitted scalers)
        log_returns_scaled, scalers = scaling(values)
        windows = create_windows(log_returns_scaled, config.n_steps)

        # Build Dataset / DataLoader
        train_size = int(windows.shape[0] * 1.0)
        training_data = windows[:train_size]
        train_set = TensorDataset(training_data)
        train_dl = DataLoader(train_set, batch_size=config.batch_size, shuffle=True)

        # ---------------------------
        # 2) Model definition (Generator / Discriminator)
        # ---------------------------
        G, D = get_model(config)

        # ---------------------------
        # 3) Train the model
        # ---------------------------
        trainer = train_model(G, D, train_dl, config)

        # Save final models (example paths)
        save_dir = f"./results/models/{config.asset}_{config.file_name}"
        os.makedirs(save_dir, exist_ok=True)
        save_model(G, os.path.join(save_dir, 'Generator_final.pt'))
        save_model(D, os.path.join(save_dir, 'Discriminator_final.pt'))

        # ---------------------------
        # 4) Inference / generation — large-batch generation, filtering, saving
        # ---------------------------
        total_samples = getattr(config, 'inference_total', 10000)
        chunk_size = getattr(config, 'inference_chunk', 2000)
        n_steps = config.n_steps

        fake_data = generate_samples(G, scalers, total_samples, n_steps, chunk_size, device, config)

        # Simple range-based filtering (keeps the original script's approach)
        real_data = inverse_scaling(training_data.transpose(1, 2), scalers)

        out_dir = f"./outputs/{config.asset}"
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f'timeseries_{config.asset}_{config.file_name}.pkl'), 'wb') as f:
            pickle.dump(fake_data, f)

        # ---------------------------
        # 5) (Optional) Evaluation: EMD, etc.
        # ---------------------------
        try:
            emd = evaluate_generated(real_data, fake_data, window=100)
            print(f"[{config.file_name}] EMD: {emd:.4f}")
        except Exception:
            print("Evaluation failed — compute_avg_emd may not be available.")


if __name__ == '__main__':
    main()
