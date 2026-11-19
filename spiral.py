"""Synthetic LeJEPA experiment from Klindt, LeCun & Balestriero (arXiv:2605.26379).

This implementation follows the authors' public 2-D experiment conventions:
https://github.com/klindtlab/lejepa-identifiability/tree/main/experiments
The training interface exposes observations only. The known coordinates are
used after training to fit and independently test an affine recovery probe.
"""

import math

import torch
from torch import nn


def spiral_observation(latent):
    """Rotate each point by pi times its Euclidean radius (Appendix H.1)."""
    x, y = latent.unbind(dim=-1)
    angle = math.pi * torch.linalg.vector_norm(latent, dim=-1)
    cosine, sine = angle.cos(), angle.sin()
    return torch.stack((x * cosine - y * sine, x * sine + y * cosine), dim=-1)


class SpiralEncoder(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, width), nn.GELU(),
            nn.Linear(width, 2),
        )
        # Fixed, real-valued quadrature avoids complex accelerator operators.
        frequencies = torch.linspace(0.0, 3.0, 17)
        gaussian = torch.exp(-0.5 * frequencies.square())
        # Integrate an even integrand over [-3, 3] using doubled trapezoids.
        coefficients = torch.full_like(frequencies, 2.0 * 3.0 / 16)
        coefficients[0] *= 0.5
        coefficients[-1] *= 0.5
        self.register_buffer("frequencies", frequencies)
        self.register_buffer("gaussian", gaussian)
        self.register_buffer("quadrature", coefficients * gaussian)

    def forward(self, observations):
        return self.network(observations)


def make_encoder(width=256):
    """Construct the four-layer, two-output MLP from Appendix H.2."""
    if width < 1:
        raise ValueError("width must be positive")
    return SpiralEncoder(width)


def make_batch(batch_size, slices, generator, rho=0.95):
    """Return CPU float tensors: two observed views [2,B,2], directions [2,S].

    As in the authors' data.py, each view receives independent OU noise
    applied to the same hidden center. Its correlation with the center is
    rho; the correlation between the two views is rho**2.
    """
    if batch_size < 2 or slices < 1 or not 0 < rho < 1:
        raise ValueError("expected batch_size >= 2, slices >= 1 and 0 < rho < 1")
    centers = torch.randn(batch_size, 2, generator=generator, device="cpu")
    noise = torch.randn(2, batch_size, 2, generator=generator, device="cpu")
    latent_views = rho * centers + math.sqrt(1 - rho**2) * noise
    directions = torch.randn(2, slices, generator=generator, device="cpu")
    return spiral_observation(latent_views), nn.functional.normalize(directions, dim=0)


def objective(encoder, views, projections, weight=0.001):
    """Compute the scalar LeJEPA loss. Set weight=0 for the alignment control."""
    embeddings = encoder(views)
    # For two views, mean squared deviation from their center is 1/4 of MSE.
    alignment = 0.25 * (embeddings[0] - embeddings[1]).square().mean()
    if weight == 0:
        return alignment
    flat = embeddings.flatten(0, 1)
    angles = (flat @ projections).unsqueeze(-1) * encoder.frequencies
    real_error = angles.cos().mean(dim=0) - encoder.gaussian
    imaginary_error = angles.sin().mean(dim=0)
    squared_error = real_error.square() + imaginary_error.square()
    sigreg = (squared_error * encoder.quadrature).sum(dim=-1).mean() * flat.shape[0]
    return (1 - weight) * alignment + weight * sigreg


def _fit_and_predict(fit_features, fit_targets, test_features):
    """CPU/FP64 minimum-norm affine fit, including rank-deficient embeddings."""
    feature_mean = fit_features.mean(dim=0)
    target_mean = fit_targets.mean(dim=0)
    weights = torch.linalg.pinv(fit_features - feature_mean) @ (fit_targets - target_mean)
    return (test_features - feature_mean) @ weights + target_mean


def _r2(target, prediction):
    residual = (target - prediction).square().sum(dim=0)
    total = (target - target.mean(dim=0)).square().sum(dim=0)
    return 1 - residual / total


@torch.no_grad()
def evaluate(encoder, device, seed=1729, n_fit=2048, n_test=2048, n_plot=512):
    """Return JSON-ready metrics and the first n_plot test examples for plotting.

    The two probes (observations and embeddings) share a fit set; the R²,
    covariance statistics, and scatter coordinates all use a disjoint test set.
    """
    if min(n_fit, n_test) < 3 or n_plot < 0:
        raise ValueError("expected n_fit and n_test >= 3 and n_plot >= 0")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    true = torch.randn(n_fit + n_test, 2, generator=generator, device="cpu")
    observed = spiral_observation(true)
    was_training = encoder.training
    encoder.eval()
    try:
        embeddings = encoder(observed.to(device)).cpu().double()
    finally:
        encoder.train(was_training)
    true, observed = true.double(), observed.double()
    fit_target, test_target = true.split((n_fit, n_test))
    fit_obs, test_obs = observed.split((n_fit, n_test))
    fit_emb, test_emb = embeddings.split((n_fit, n_test))
    recovered = _fit_and_predict(fit_emb, fit_target, test_emb)
    baseline = _fit_and_predict(fit_obs, fit_target, test_obs)
    r2 = _r2(test_target, recovered)
    covariance = torch.cov(test_emb.T)
    metrics = {
        "r2": r2.mean().item(),
        "r2_by_coordinate": r2.tolist(),
        "raw_r2": _r2(test_target, baseline).mean().item(),
        "covariance_eigenvalues": torch.linalg.eigvalsh(covariance).tolist(),
        "total_variance": covariance.trace().item(),
    }
    scatter = {
        name: values[:n_plot].tolist()
        for name, values in (
            ("true", test_target), ("observed", test_obs),
            ("embedding", test_emb), ("recovered", recovered),
        )
    }
    return metrics, scatter
