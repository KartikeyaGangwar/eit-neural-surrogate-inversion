"""
Shape library for generating 2D primitive geometries.
======================================================

Generates 8 shape families as sampled 2D boundary points:
  - circle
  - ellipse
  - rectangle
  - star
  - banana
  - crescent
  - random_convex
  - random_concave

All functions return:
  (boundary_points: np.ndarray shape (n_samples, 2), metadata: dict)

No EIDORS dependency. Runs on Colab.
"""

from __future__ import annotations
import numpy as np
from typing import Tuple, Dict, Any, Optional


def generate_circle(
    radius: float = 0.35,
    center: Tuple[float, float] = (0.0, 0.0),
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a circular boundary."""
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    pts = np.stack([
        center[0] + radius * np.cos(t),
        center[1] + radius * np.sin(t),
    ], axis=1)
    return pts, {
        "shape_family": "circle",
        "shape_params": {"radius": radius, "center": list(center)},
    }


def generate_ellipse(
    a: float = 0.45,
    b: float = 0.25,
    center: Tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate an elliptical boundary."""
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    x = a * np.cos(t)
    y = b * np.sin(t)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pts = np.stack([
        center[0] + x * cos_a - y * sin_a,
        center[1] + x * sin_a + y * cos_a,
    ], axis=1)
    return pts, {
        "shape_family": "ellipse",
        "shape_params": {"a": a, "b": b, "center": list(center), "angle": angle},
    }


def generate_rectangle(
    width: float = 0.50,
    height: float = 0.30,
    center: Tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    m: float = 6.0,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a smooth rectangle via superellipse."""
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    cos_t = np.cos(t)
    sin_t = np.sin(t)
    x = (width / 2.0) * np.sign(cos_t) * (np.abs(cos_t)) ** (2.0 / m)
    y = (height / 2.0) * np.sign(sin_t) * (np.abs(sin_t)) ** (2.0 / m)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pts = np.stack([
        center[0] + x * cos_a - y * sin_a,
        center[1] + x * sin_a + y * cos_a,
    ], axis=1)
    return pts, {
        "shape_family": "rectangle",
        "shape_params": {
            "width": width,
            "height": height,
            "center": list(center),
            "angle": angle,
            "superellipse_m": m,
        },
    }


def generate_star(
    n_points: int = 5,
    outer_radius: float = 0.45,
    inner_radius: float = 0.20,
    center: Tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a smooth star-shaped boundary (concave)."""
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    r = inner_radius + (outer_radius - inner_radius) * 0.5 * (1.0 + np.cos(n_points * (t - angle)))
    pts = np.stack([
        center[0] + r * np.cos(t),
        center[1] + r * np.sin(t),
    ], axis=1)
    return pts, {
        "shape_family": "star",
        "shape_params": {
            "n_points": n_points,
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
            "center": list(center),
            "angle": angle,
        },
    }


def generate_banana(
    length: float = 0.45,
    curvature: float = 0.60,
    width: float = 0.15,
    center: Tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a curved banana/crescent-like boundary (concave)."""
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    x0 = length * np.cos(t)
    y0 = width * np.sin(t) + curvature * (x0 ** 2 - (length / 2.0) ** 2)
    cos_a, sin_a = np.cos(angle), np.sin(angle)
    pts = np.stack([
        center[0] + x0 * cos_a - y0 * sin_a,
        center[1] + x0 * sin_a + y0 * cos_a,
    ], axis=1)
    return pts, {
        "shape_family": "banana",
        "shape_params": {
            "length": length,
            "curvature": curvature,
            "width": width,
            "center": list(center),
            "angle": angle,
        },
    }


def generate_crescent(
    outer_radius: float = 0.40,
    inner_radius: float = 0.25,
    offset: float = 0.18,
    center: Tuple[float, float] = (0.0, 0.0),
    angle: float = 0.0,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a moon crescent boundary (concave)."""
    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    # Concave indentation near t = pi
    d_t = (t - np.pi) % (2.0 * np.pi) - np.pi
    r = outer_radius - offset * np.exp(-6.0 * d_t ** 2)
    t_rot = t + angle
    pts = np.stack([
        center[0] + r * np.cos(t_rot),
        center[1] + r * np.sin(t_rot),
    ], axis=1)
    return pts, {
        "shape_family": "crescent",
        "shape_params": {
            "outer_radius": outer_radius,
            "inner_radius": inner_radius,
            "offset": offset,
            "center": list(center),
            "angle": angle,
        },
    }


def generate_random_convex(
    n_vertices: int = 5,
    radius_range: Tuple[float, float] = (0.25, 0.45),
    center: Tuple[float, float] = (0.0, 0.0),
    rng: Optional[np.random.Generator] = None,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a smooth random convex shape via low-order Fourier series."""
    if rng is None:
        rng = np.random.default_rng()

    r0 = float(rng.uniform(*radius_range))
    # Small harmonic amplitudes ensuring r0 + a_k(1+k^2) > 0 (strict convexity)
    a2 = float(rng.uniform(-0.06, 0.06))
    phi2 = float(rng.uniform(0.0, 2.0 * np.pi))

    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    r = r0 + a2 * np.cos(2.0 * t + phi2)
    pts = np.stack([
        center[0] + r * np.cos(t),
        center[1] + r * np.sin(t),
    ], axis=1)
    return pts, {
        "shape_family": "random_convex",
        "shape_params": {
            "r0": r0,
            "a2": a2,
            "phi2": phi2,
            "center": list(center),
        },
    }


def generate_random_concave(
    n_vertices: int = 5,
    radius_range: Tuple[float, float] = (0.25, 0.45),
    center: Tuple[float, float] = (0.0, 0.0),
    rng: Optional[np.random.Generator] = None,
    n_samples: int = 1024,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """Generate a random concave shape with pronounced star-like reflex indentations.

    Uses a star-like Fourier modulation r(t) = r0 * (1 - depth * cos(k*t)^2)
    which guarantees clear concavity that is detected by both curvature and
    vertex-based methods after B-spline fitting.

    The depth parameter controls how concave: 0 = convex, 0.5 = moderate,
    0.7 = deep star-like.
    """
    if rng is None:
        rng = np.random.default_rng()

    r0 = float(rng.uniform(*radius_range))
    k = int(rng.choice([3, 4, 5]))
    # Depth in [0.4, 0.65]: ensures visually clear, FEM-valid concavity
    # r_min = r0 * (1 - depth), r_max = r0
    depth = float(rng.uniform(0.40, 0.65))
    phi_k = float(rng.uniform(0.0, 2.0 * np.pi))

    t = np.linspace(0.0, 2.0 * np.pi, n_samples, endpoint=False)
    # Cosine^2 gives smooth, star-like petal pattern:
    #   r_max = r0,  r_min = r0 * (1 - depth)
    r = r0 * (1.0 - depth * np.cos(k * t + phi_k) ** 2)
    pts = np.stack([
        center[0] + r * np.cos(t),
        center[1] + r * np.sin(t),
    ], axis=1)
    return pts, {
        "shape_family": "random_concave",
        "shape_params": {
            "r0": r0,
            "k_harmonics": k,
            "depth": depth,
            "phi_k": phi_k,
            "center": list(center),
        },
    }
