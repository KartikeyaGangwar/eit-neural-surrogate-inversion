"""
Script to generate reference_output.mat using Octave + EIDORS reference routines.

This produces the MATLAB/Octave reference dataset for Stage D validation:
- theta (64,)
- sigma (N_elem,)
- dSigma_dTheta (N_elem, 64)
- voltage (1920,)
- J_theta (1920, 64)
"""

from __future__ import annotations
import sys
import os
import subprocess
import tempfile
import numpy as np
import scipy.io as sio

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from geometry.parameters import sample_reference_geometry
from physics.fem_backend import FEMBackend
from config import EIDORS_PATH

def generate_reference_mat(output_mat: str = "reference_output.mat") -> None:
    """
    Generate authoritative reference MAT file using Octave and EIDORS.
    
    Args:
        output_mat: Output .mat filepath to save reference matrices.
    """
    print(f"Generating MATLAB/Octave reference output MAT file: {output_mat}")
    octave_exe = FEMBackend._find_octave_executable()
    if not octave_exe:
        raise RuntimeError("Octave executable not found.")

    theta = sample_reference_geometry()
    
    # Root dir containing polygon_conductivity_bspline.m
    root_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    with tempfile.TemporaryDirectory(prefix="ref_gen_") as tmpdir:
        theta_mat = os.path.join(tmpdir, "theta_in.mat")
        sio.savemat(theta_mat, {"theta": theta.reshape(-1, 1)})

        script_m = os.path.join(tmpdir, "gen_ref.m")
        res_mat = os.path.join(tmpdir, "res_ref.mat")

        root_path_esc = root_dir.replace("\\", "/").replace("'", "''")
        eidors_path_esc = EIDORS_PATH.replace("\\", "/").replace("'", "''")
        theta_mat_esc = theta_mat.replace("\\", "/").replace("'", "''")
        res_mat_esc = res_mat.replace("\\", "/").replace("'", "''")

        m_code = f"""
        addpath('{root_path_esc}');
        run('{eidors_path_esc}/startup.m');

        % 1. Circular geometry
        n_elec = 16;
        elec_w = 0.20;
        elec_rf = 10;
        z_contact = 0.01;
        maxsz = 0.05;

        n_pts = 256;
        t_bnd = linspace(0, 2*pi, n_pts+1)';
        t_bnd(end) = [];
        circle_bnd = [cos(t_bnd), sin(t_bnd)];

        shape = {{circle_bnd, maxsz}};
        elec_spec = [elec_w, elec_rf];
        mdl = ng_mk_2d_model(shape, n_elec, elec_spec);

        % CEM z_contact
        for i = 1:n_elec
            mdl.electrode(i).z_contact = z_contact;
        end

        % 120 stim patterns
        stim = [];
        k = 1;
        meas_mat = eye(n_elec) - ones(n_elec)/n_elec;
        for i = 1:n_elec
            for j = (i+1):n_elec
                sp = zeros(n_elec, 1);
                sp(i) = 1;
                sp(j) = -1;
                stim(k).stim_pattern = sp;
                stim(k).meas_pattern = meas_mat;
                k = k + 1;
            end
        end
        mdl.stimulation = stim;
        mdl.solve = @fwd_solve_1st_order;
        mdl.system_mat = @system_mat_1st_order;
        mdl.normalize_measurements = 0;

        nodes = mdl.nodes;
        elems = mdl.elems;
        n_elem = size(elems, 1);
        centres = zeros(n_elem, 2);
        for k = 1:n_elem
            centres(k, :) = mean(nodes(elems(k, :), :), 1);
        end

        % Load theta
        t_data = load('{theta_mat_esc}');
        theta = t_data.theta(:);

        % Compute sigma and dSigma_dTheta using reference MATLAB script
        sigma_bg = 1.0;
        sigma_inc = 1.0e-4;
        alpha = 80.0;

        [sigma, dSigma_dTheta, ~] = polygon_conductivity_bspline(centres, theta, sigma_bg, sigma_inc, alpha);

        % Compute voltage and J_sigma
        img = eidors_obj('image', 'solve', 'elem_data', sigma, 'fwd_model', mdl);
        data = fwd_solve(img);
        voltage = data.meas(:);

        J_sigma = calc_jacobian(img);
        J_theta = J_sigma * dSigma_dTheta;

        % Save in v6 format
        save('-v6', '{res_mat_esc}', 'theta', 'sigma', 'dSigma_dTheta', 'voltage', 'J_theta');
        """

        with open(script_m, "w") as f:
            f.write(m_code)

        print(f"Running Octave script to calculate reference outputs...")
        proc = subprocess.run([octave_exe, "--no-gui", "--no-history", script_m], capture_output=True, text=True)
        if proc.returncode != 0:
            print("Octave stdout:\n", proc.stdout)
            print("Octave stderr:\n", proc.stderr)
            raise RuntimeError("Octave reference generation failed.")

        res_data = sio.loadmat(res_mat)
        sio.savemat(output_mat, res_data)
        print(f"Successfully generated '{output_mat}'.")

if __name__ == "__main__":
    generate_reference_mat()
