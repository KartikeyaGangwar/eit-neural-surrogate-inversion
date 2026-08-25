# ===========================================================================
# EIDORS-DEPENDENT MODULE
#
# This module requires:
#   - Octave installed and accessible on the system PATH
#   - EIDORS 3.12 startup.m accessible at config.EIDORS_PATH
#
# DO NOT import this module at the top level in Colab or any non-FEM context.
# Import only inside functions or scripts where FEM is explicitly needed.
#
# All other project modules (geometry, surrogate, inversion, experiments)
# work without this module.
# ===========================================================================
"""
EIDORS/Octave subprocess backend for EIT forward problems.

Wraps the local EIDORS/Octave installation to:
  1. Build and cache a circular 2D FEM model with 16 electrodes
  2. Solve the forward problem: theta -> voltage (1920,)
  3. Compute the geometry Jacobian: theta -> J_theta (1920, 64)
     via J_theta = J_sigma @ dSigma/dTheta

The Octave subprocess protocol:
  - Python writes a temporary .m script
  - Python calls: octave --no-gui --no-history <script.m>
  - Octave writes results to .mat files
  - Python reads .mat files via scipy.io.loadmat

FEM configuration (matches validated MATLAB reference):
  - Circular domain, radius 1.0 m
  - 16 surface electrodes (Complete Electrode Model)
  - 120 stimulation patterns: all pairs (i, j) with i < j
  - Measurement matrix: I_16 - ones(16,16)/16  (mean-subtracted)
  - fwd_solve_1st_order
  - system_mat_1st_order
  - normalize_measurements = 0
"""

from __future__ import annotations

import os
import sys
import subprocess
import tempfile
import textwrap
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import Optional, Tuple

# Allow import from project root without installation
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as cfg
from physics.fem_model import FEMModelConfig
from geometry.conductivity import compute_sigma_and_derivative


class FEMBackend:
    """
    EIDORS/Octave subprocess backend for circular EIT forward problems.

    Usage (local Windows machine with EIDORS installed)::

        backend = FEMBackend(eidors_path=r"C:\\path\\to\\eidors-v3.12-ng\\eidors")
        backend.build_model()
        voltage = backend.solve(theta)            # shape (1920,)
        voltage, J_theta = backend.solve_with_jacobian(theta)  # (1920,), (1920,64)

    The EIDORS path is read from config.EIDORS_PATH by default, which in turn
    reads from the EIDORS_PATH environment variable if set.
    """

    def __init__(
        self,
        eidors_path: Optional[str] = None,
        mesh_maxsz: Optional[float] = None,
        model_config: Optional[FEMModelConfig] = None,
        work_dir: Optional[Path] = None,
        verbose: bool = False,
    ) -> None:
        """
        Args:
            eidors_path:   Path to the EIDORS startup.m directory.
                           Defaults to config.EIDORS_PATH.
            mesh_maxsz:    Max element size [m] (e.g. 0.05). If provided,
                           overrides model_config.mesh_maxsz.
            model_config:  FEM model configuration. Defaults to FEMModelConfig().
            work_dir:      Directory for temporary Octave scripts and .mat files.
                           A temporary directory is created if None.
            verbose:       If True, print Octave stdout/stderr.
        """
        # Initialize attributes immediately for safe destruction if construction fails
        self._tmp: Optional[tempfile.TemporaryDirectory] = None
        self._model_built: bool = False
        self._n_elem: Optional[int] = None

        self.eidors_path = (eidors_path or cfg.EIDORS_PATH).replace("\\", "/")
        
        if model_config is None:
            self.model_cfg = FEMModelConfig()
        else:
            self.model_cfg = model_config

        if mesh_maxsz is not None:
            self.model_cfg.mesh_maxsz = float(mesh_maxsz)

        self.verbose = verbose

        if work_dir is None:
            self._tmp = tempfile.TemporaryDirectory(prefix="eit_fem_")
            self.work_dir = Path(self._tmp.name)
        else:
            self.work_dir = Path(work_dir)
            self.work_dir.mkdir(parents=True, exist_ok=True)

        # Cached file paths
        self._model_mat = self.work_dir / "cached_model.mat"
        self._centres_mat = self.work_dir / "centres.mat"

    # ------------------------------------------------------------------
    # PUBLIC INTERFACE
    # ------------------------------------------------------------------

    def build_model(self) -> None:
        """
        Build the static circular FEM model in Octave and cache to disk.

        This is the expensive one-time setup call. Subsequent solve() and
        jacobian() calls reuse the cached model.

        The model is built with:
          - Circular domain (radius 1.0, 256 boundary points for Netgen)
          - 16 electrodes (CEM, width 0.20 rad, z_contact 0.01)
          - 120 stimulation patterns
          - Mean-subtracted measurement matrix
          - fwd_solve_1st_order / system_mat_1st_order

        Raises:
            RuntimeError: if Octave fails.
        """
        script = self._make_build_script()
        script_path = self.work_dir / "build_model.m"
        script_path.write_text(script, encoding="utf-8")
        self._run_octave(script_path, label="build_model")
        self._model_built = True

        # Load N_elem for documentation purposes
        centres = self._load_centres()
        self._n_elem = centres.shape[0]
        if self.verbose:
            print(f"[FEMBackend] Model built. N_elem = {self._n_elem}")

    def get_element_centres(self) -> np.ndarray:
        """
        Return FEM element centroid coordinates.

        Returns:
            centres: shape (N_elem, 2)
        """
        self._require_model()
        return self._load_centres()

    def solve(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute EIT voltage measurements for given B-spline parameters.

        Args:
            theta: shape (64,)

        Returns:
            voltage: shape (1920,)
        """
        self._check_theta(theta)
        self._require_model()

        centres = self._load_centres()
        sigma, _ = compute_sigma_and_derivative(theta, centres)

        sigma_path = self.work_dir / "sigma_in.mat"
        sio.savemat(str(sigma_path), {"sigma": sigma.reshape(-1, 1)})

        script = self._make_solve_script(sigma_path)
        result_path = self.work_dir / "result_solve.mat"
        script_path = self.work_dir / "run_solve.m"
        script_path.write_text(script.format(result_path=result_path.as_posix()),
                                encoding="utf-8")
        self._run_octave(script_path, label="solve")

        data = sio.loadmat(str(result_path))
        voltage = np.array(data["voltage"]).flatten()
        assert voltage.shape == (cfg.N_MEAS,), \
            f"Expected voltage shape ({cfg.N_MEAS},), got {voltage.shape}"
        return voltage

    def jacobian(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute the geometry Jacobian J_theta = dV/dtheta.

        Args:
            theta: shape (64,)

        Returns:
            J_theta: shape (1920, 64)
        """
        _, J_theta = self.solve_with_jacobian(theta)
        return J_theta

    def solve_with_jacobian(
        self, theta: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute voltage and geometry Jacobian in a single Octave call.

        J_theta = J_sigma @ dSigma_dTheta

        where:
            J_sigma        = EIDORS calc_jacobian output, shape (N_meas, N_elem)
            dSigma_dTheta  = analytic derivative,        shape (N_elem, 64)
            J_theta        = composition,                shape (N_meas, 64)

        Memory at maxsz=0.05:
            N_elem ~ 2,000–4,000
            J_sigma ~ 45 MB float64  (handled inside Octave)
            J_theta ~ 1 MB float64   (always small)

        Args:
            theta: shape (64,)

        Returns:
            voltage:  shape (1920,)
            J_theta:  shape (1920, 64)
        """
        self._check_theta(theta)
        self._require_model()

        centres = self._load_centres()
        sigma, dSigma_dTheta = compute_sigma_and_derivative(theta, centres)

        sigma_path = self.work_dir / "sigma_in.mat"
        sio.savemat(str(sigma_path), {"sigma": sigma.reshape(-1, 1)})

        result_path = self.work_dir / "result_jac.mat"
        script = self._make_jacobian_script(sigma_path, result_path)
        script_path = self.work_dir / "run_jacobian.m"
        script_path.write_text(script, encoding="utf-8")
        self._run_octave(script_path, label="jacobian")

        data = sio.loadmat(str(result_path))
        voltage = np.array(data["voltage"]).flatten()          # (1920,)
        J_sigma = np.array(data["J_sigma"])                    # (1920, N_elem)

        # Chain rule: J_theta = J_sigma @ dSigma_dTheta
        J_theta = J_sigma @ dSigma_dTheta                     # (1920, 64)

        assert voltage.shape == (cfg.N_MEAS,)
        assert J_theta.shape == (cfg.N_MEAS, cfg.N_THETA)

        return voltage, J_theta

    # ------------------------------------------------------------------
    # OCTAVE SCRIPT GENERATORS
    # ------------------------------------------------------------------

    def _make_build_script(self) -> str:
        """Generate the Octave model-build script."""
        mc = self.model_cfg
        ep = self.eidors_path
        model_mat = self._model_mat.as_posix()
        centres_mat = self._centres_mat.as_posix()

        # Match MATLAB reference test_stage3_eidors_conductivity_jacobian.m lines 195-230:
        # Construct explicit circle boundary points (256 samples) and use [electrode_width electrode_rfnum].
        script = textwrap.dedent(f"""\
            run('{ep}/startup.m');

            % ---- Circular FEM domain boundary ({mc.n_boundary_points} points, radius {mc.domain_radius}) ----
            n_bp = {mc.n_boundary_points};
            b_th = linspace(0, 2*pi, n_bp + 1)';
            b_th(end) = [];
            circle_boundary = [{mc.domain_radius} * cos(b_th), {mc.domain_radius} * sin(b_th)];

            shape = {{circle_boundary, {mc.mesh_maxsz}}};
            elec_spec = [{mc.electrode_width}, {mc.electrode_rfnum}];
            mdl = ng_mk_2d_model(shape, {mc.n_elec}, elec_spec);

            % ---- Contact impedance ----
            for i = 1:{mc.n_elec}
                mdl.electrode(i).z_contact = {mc.z_contact};
            end

            % ---- Stimulation patterns: all pairs (i,j) with i<j ----
            n_elec = {mc.n_elec};
            meas_mat = eye(n_elec) - ones(n_elec) / n_elec;
            k = 1;
            clear stim;
            for i = 1:n_elec
                for j = (i+1):n_elec
                    sp = zeros(n_elec, 1);
                    sp(i) =  {mc.current_amplitude};
                    sp(j) = -{mc.current_amplitude};
                    stim(k).stim_pattern = sp;
                    stim(k).meas_pattern = meas_mat;
                    k = k + 1;
                end
            end
            mdl.stimulation = stim;

            % ---- Solver configuration ----
            mdl.solve           = @{mc.solver};
            mdl.system_mat      = @{mc.system_mat};
            mdl.normalize_measurements = {mc.normalize_measurements};

            % ---- Save model (Octave format supports function handles) ----
            save('{model_mat}', 'mdl');

            % ---- Compute and save element centres (v6 MAT format for SciPy loadmat) ----
            nodes = mdl.nodes;
            elems = mdl.elems;
            n_elem = size(elems, 1);
            centres = zeros(n_elem, 2);
            for k = 1:n_elem
                centres(k, :) = mean(nodes(elems(k, :), :), 1);
            end
            save('-v6', '{centres_mat}', 'centres');
            fprintf('BUILD COMPLETE: n_elem = %d\\n', n_elem);
        """)
        return script

    def _make_solve_script(self, sigma_path: Path) -> str:
        """Generate the Octave forward-solve script."""
        ep = self.eidors_path
        model_mat = self._model_mat.as_posix()
        sig_mat = sigma_path.as_posix()

        script = textwrap.dedent(f"""\
            run('{ep}/startup.m');
            load('{model_mat}', 'mdl');
            d = load('{sig_mat}');
            sigma = d.sigma(:);
            img = eidors_obj('image', 'fwd_solve', 'elem_data', sigma, 'fwd_model', mdl);
            data = fwd_solve(img);
            voltage = data.meas(:);
            save('-v6', '{{result_path}}', 'voltage');
            fprintf('SOLVE COMPLETE: n_meas = %d\\n', numel(voltage));
        """)
        return script

    def _make_jacobian_script(self, sigma_path: Path, result_path: Path) -> str:
        """Generate the Octave Jacobian script."""
        ep = self.eidors_path
        model_mat = self._model_mat.as_posix()
        sig_mat = sigma_path.as_posix()
        res_mat = result_path.as_posix()

        script = textwrap.dedent(f"""\
            run('{ep}/startup.m');
            load('{model_mat}', 'mdl');
            d = load('{sig_mat}');
            sigma = d.sigma(:);
            img = eidors_obj('image', 'jacobian', 'elem_data', sigma, 'fwd_model', mdl);
            data = fwd_solve(img);
            voltage = data.meas(:);
            J_sigma = calc_jacobian(img);
            save('-v6', '{res_mat}', 'voltage', 'J_sigma');
            fprintf('JACOBIAN COMPLETE: size(J_sigma) = [%d %d]\\n', size(J_sigma,1), size(J_sigma,2));
        """)
        return script

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _find_octave_executable() -> str:
        """Locate octave executable on system PATH or standard installation paths."""
        import shutil

        # 1. Environment variable
        env_octave = os.environ.get("OCTAVE_EXECUTABLE") or os.environ.get("OCTAVE_PATH")
        if env_octave and os.path.isfile(env_octave):
            return env_octave

        # 2. System PATH
        which_octave = shutil.which("octave")
        if which_octave:
            return which_octave

        # 3. Search standard Windows installation paths
        search_patterns = [
            r"C:\Program Files\GNU Octave\*\mingw64\bin\octave.exe",
            r"C:\Program Files (x86)\GNU Octave\*\mingw64\bin\octave.exe",
            r"C:\Octave\*\mingw64\bin\octave.exe",
        ]
        import glob
        for pattern in search_patterns:
            matches = glob.glob(pattern)
            if matches:
                # Return the highest version match
                matches.sort(reverse=True)
                return matches[0]

        # Fallback to simple command name
        return "octave"

    def _run_octave(self, script_path: Path, label: str = "") -> None:
        """Run octave --no-gui --no-history <script_path>."""
        octave_exe = self._find_octave_executable()
        cmd = [octave_exe, "--no-gui", "--no-history", str(script_path)]
        tag = f"[FEMBackend:{label}]" if label else "[FEMBackend]"
        if self.verbose:
            print(f"{tag} Running: {' '.join(cmd)}")
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            if self.verbose:
                if result.stdout:
                    print(f"{tag} stdout:\n{result.stdout}")
                if result.stderr:
                    print(f"{tag} stderr:\n{result.stderr}")
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"{tag} Octave failed (exit {exc.returncode}).\n"
                f"Script: {script_path}\n"
                f"stdout: {exc.stdout}\n"
                f"stderr: {exc.stderr}"
            ) from exc

    def _load_centres(self) -> np.ndarray:
        """Load element centroids from cached .mat file."""
        data = sio.loadmat(str(self._centres_mat))
        centres = np.array(data["centres"])
        assert centres.ndim == 2 and centres.shape[1] == 2, \
            f"Unexpected centres shape: {centres.shape}"
        return centres

    def _require_model(self) -> None:
        """Raise if build_model() has not been called."""
        if not self._model_built:
            raise RuntimeError(
                "FEM model not built. Call backend.build_model() first."
            )

    @staticmethod
    def _check_theta(theta: np.ndarray) -> None:
        """Validate theta shape."""
        if not isinstance(theta, np.ndarray):
            raise TypeError(f"theta must be np.ndarray, got {type(theta)}")
        if theta.shape != (cfg.N_THETA,):
            raise ValueError(
                f"theta must have shape ({cfg.N_THETA},), got {theta.shape}"
            )

    def __repr__(self) -> str:
        status = "built" if self._model_built else "not built"
        return (
            f"FEMBackend(eidors_path='{self.eidors_path}', "
            f"mesh_maxsz={self.model_cfg.mesh_maxsz}, "
            f"model={status}, n_elem={self._n_elem})"
        )

    def __del__(self) -> None:
        tmp = getattr(self, "_tmp", None)
        if tmp is not None:
            try:
                tmp.cleanup()
            except Exception:
                pass
