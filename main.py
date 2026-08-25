"""
Zero-Online-FEM EIT B-Spline Pipeline — Master Command-Line Interface (CLI)
===========================================================================
Unified entry point for physics validation, dataset generation, surrogate training,
deep ensemble benchmarking, multi-start Levenberg-Marquardt shape inversion,
measurement noise robustness, and publication figure generation.

Available Commands:
    # --- Physical Model Validations (Stage A - D) ---
    python main.py validate-a       # Pure Python analytic vs FD dSigma/dTheta test
    python main.py validate-b       # Stage 3: EIDORS J_sigma sensitivity validation (local FEM)
    python main.py validate-c       # Stage 4: J_theta full geometric Jacobian validation (local FEM)
    python main.py validate-d       # Stage D: Python vs MATLAB reference numerical parity check

    # --- Dataset Generation ---
    python main.py dataset          # Generate chunked HDF5 dataset (requires local EIDORS)

    # --- ML Training & Inference ---
    python main.py train-surrogate  # Train baseline VoltageSurrogate on HDF5 dataset
    python main.py invert           # Run zero-online-FEM deterministic shape inversion (9 targets)
    python main.py benchmark-ensemble # Run 5-member Deep Ensemble forward & inverse evaluation
    python main.py benchmark-noise  # Run 234-trial measurement noise robustness benchmark

    # --- Verification & Publication Figures ---
    python main.py validate-master  # Verify all quantitative manuscript claims & integrity
    python main.py generate-figures # Generate all 28 high-resolution publication figures
"""

from __future__ import annotations
import sys
import os
import argparse
from pathlib import Path

# Add project root and src to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import config as cfg


def cmd_validate_a(args: argparse.Namespace) -> None:
    from src.validation.val_A_conductivity_fd import main as val_a_main
    val_a_main()


def cmd_validate_b(args: argparse.Namespace) -> None:
    from src.validation.val_B_fem_jsigma import main as val_b_main
    val_b_main()


def cmd_validate_c(args: argparse.Namespace) -> None:
    from src.validation.val_C_geometry_jacobian import main as val_c_main
    val_c_main()


def cmd_validate_d(args: argparse.Namespace) -> None:
    from src.validation.val_D_matlab_comparison import main as val_d_main
    val_d_main()


def cmd_dataset(args: argparse.Namespace) -> None:
    from scripts.generate_dataset import main as gen_main
    gen_main()


def cmd_train_surrogate(args: argparse.Namespace) -> None:
    from scripts.train_surrogate import main as train_main
    train_main()


def cmd_invert(args: argparse.Namespace) -> None:
    from scripts.run_inversion import run_inversion_benchmark
    run_inversion_benchmark(quick_mode=args.quick)


def cmd_benchmark_ensemble(args: argparse.Namespace) -> None:
    from scripts.run_deep_ensemble import evaluate_ensemble
    evaluate_ensemble(quick_mode=args.quick)


def cmd_benchmark_noise(args: argparse.Namespace) -> None:
    from scripts.run_noise_robustness import run_noise_benchmark
    run_noise_benchmark(quick_mode=args.quick)


def cmd_validate_master(args: argparse.Namespace) -> None:
    from scripts.run_master_validation import main as master_val_main
    master_val_main()


def cmd_generate_figures(args: argparse.Namespace) -> None:
    from scripts.plot_computational_domain_setup import main as domain_fig_main
    from scripts.generate_journal_tpami_figures import main as journal_fig_main
    from scripts.generate_publication_figures import main as pub_fig_main
    domain_fig_main()
    journal_fig_main()
    pub_fig_main()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python main.py",
        description="Master CLI for Zero-Online-FEM B-Spline EIT Inverse Reconstruction Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # validate-a
    p_va = subparsers.add_parser("validate-a", help="Run Stage 1 analytic vs FD dSigma/dTheta validation")
    p_va.set_defaults(func=cmd_validate_a)

    # validate-b
    p_vb = subparsers.add_parser("validate-b", help="Run Stage 3 EIDORS J_sigma sensitivity validation (local FEM)")
    p_vb.set_defaults(func=cmd_validate_b)

    # validate-c
    p_vc = subparsers.add_parser("validate-c", help="Run Stage 4 J_theta geometry Jacobian validation (local FEM)")
    p_vc.set_defaults(func=cmd_validate_c)

    # validate-d
    p_vd = subparsers.add_parser("validate-d", help="Run Stage D Python vs MATLAB reference comparison")
    p_vd.set_defaults(func=cmd_validate_d)

    # dataset
    p_ds = subparsers.add_parser("dataset", help="Generate chunked HDF5 dataset via EIDORS/Octave (local FEM)")
    p_ds.add_argument("--output", "-o", type=str, default=str(PROJECT_ROOT / "data" / "dataset.h5"))
    p_ds.add_argument("--n-samples", "-n", type=int, default=10000)
    p_ds.add_argument("--chunk-size", "-c", type=int, default=cfg.DATASET_CHUNK_SIZE)
    p_ds.add_argument("--seed", "-s", type=int, default=cfg.DEFAULT_SEED)
    p_ds.add_argument("--eidors-path", type=str, default=None)
    p_ds.add_argument("--mesh-maxsz", type=float, default=cfg.MESH_MAXSZ)
    p_ds.add_argument("--optimized", action="store_true")
    p_ds.add_argument("--workers", "-w", type=int, default=1)
    p_ds.add_argument("--verbose", "-v", action="store_true", default=True)
    p_ds.set_defaults(func=cmd_dataset)

    # train-surrogate
    p_tr = subparsers.add_parser("train-surrogate", help="Train baseline VoltageSurrogate on HDF5 dataset")
    p_tr.add_argument("--dataset", type=str, default=str(PROJECT_ROOT / "data" / "dataset.h5"))
    p_tr.add_argument("--output", type=str, default=str(PROJECT_ROOT / "models" / "surrogate_10k_final.pt"))
    p_tr.add_argument("--steps", type=int, default=cfg.TRAIN_STEPS)
    p_tr.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE)
    p_tr.add_argument("--lr", type=float, default=1e-3)
    p_tr.add_argument("--seed", type=int, default=cfg.DEFAULT_SEED)
    p_tr.set_defaults(func=cmd_train_surrogate)

    # invert
    p_inv = subparsers.add_parser("invert", help="Run Zero-Online-FEM multi-start LM shape inversion (9 targets)")
    p_inv.add_argument("--quick", action="store_true", help="Run on 2 targets for fast testing")
    p_inv.set_defaults(func=cmd_invert)

    # benchmark-ensemble
    p_ens = subparsers.add_parser("benchmark-ensemble", help="Evaluate K=5 Deep Ensemble forward & inverse models")
    p_ens.add_argument("--quick", action="store_true", help="Run on 2 targets for fast testing")
    p_ens.set_defaults(func=cmd_benchmark_ensemble)

    # benchmark-noise
    p_noi = subparsers.add_parser("benchmark-noise", help="Run 234-trial measurement noise robustness benchmark")
    p_noi.add_argument("--quick", action="store_true", help="Run small subset of trials for fast smoke testing")
    p_noi.set_defaults(func=cmd_benchmark_noise)

    # validate-master
    p_mas = subparsers.add_parser("validate-master", help="Verify all manuscript claims and result consistency")
    p_mas.set_defaults(func=cmd_validate_master)

    # generate-figures
    p_fig = subparsers.add_parser("generate-figures", help="Generate all 28 publication figures (PNG and PDF)")
    p_fig.set_defaults(func=cmd_generate_figures)

    return parser


def main() -> None:
    parser = build_parser()
    if len(sys.argv) == 1:
        parser.print_help(sys.stderr)
        sys.exit(1)

    args = parser.parse_args()
    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help(sys.stderr)


if __name__ == "__main__":
    main()
