function test_bspline_conductivity_derivative()
% ============================================================
% STAGE 1: ISOLATED CONDUCTIVITY DERIVATIVE VALIDATION
% ============================================================
clc; clear;

% Setup mock FEM element centers
N = 2000;
rng(42);
centres = (rand(N,2) - 0.5) * 2;

% Base non-convex geometry
control_points = [
     0.45   0.00;  0.43   0.10;  0.36   0.19;  0.25   0.27;
     0.12   0.31; -0.02   0.30; -0.15   0.25; -0.28   0.18;
    -0.38   0.07; -0.42  -0.06; -0.38  -0.18; -0.29  -0.27;
    -0.17  -0.31; -0.04  -0.30;  0.06  -0.24;  0.03  -0.13;
    -0.05  -0.06; -0.10   0.03; -0.07   0.12;  0.02   0.17;
     0.13   0.16;  0.22   0.10;  0.28   0.00;  0.25  -0.10;
     0.18  -0.17;  0.27  -0.22;  0.38  -0.19;  0.46  -0.12;
     0.49  -0.04;  0.48   0.02;  0.47   0.05;  0.45   0.00
];

% Strict compliance: theta = control_points(:)
theta = control_points(:);

sigma_bg = 1.0;
sigma_inc = 1e-4;
alpha = 80;

fprintf('STAGE 1: EVALUATING ANALYTIC dSigma_dTheta DIRECTLY...\n');
[~, dSigma_an, ~] = polygon_conductivity_bspline(centres, theta, sigma_bg, sigma_inc, alpha);

h_list = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6];
test_params = [1, 2, 17, 32, 48, 64];

fprintf('\nConductivity Central Finite Difference Errors:\n');
for k = test_params
    fprintf('\n--- Parameter %d ---\n', k);

    for h = h_list
        theta_p = theta; theta_p(k) = theta_p(k) + h;
        sigma_p = polygon_conductivity_bspline(centres, theta_p, sigma_bg, sigma_inc, alpha);

        theta_m = theta; theta_m(k) = theta_m(k) - h;
        sigma_m = polygon_conductivity_bspline(centres, theta_m, sigma_bg, sigma_inc, alpha);

        fd_col = (sigma_p - sigma_m) / (2*h);
        an_col = dSigma_an(:, k);

        % Normalize by FD column norm to match original logic
        denom = norm(fd_col, 'fro');
        if denom < 1e-14
            rel_error = norm(fd_col - an_col, 'fro');
        else
            rel_error = norm(fd_col - an_col, 'fro') / denom;
        end

        fprintf('h = %1.0e : Relative Error = %.12e\n', h, rel_error);
    end
end
fprintf('\nSTAGE 1 VALIDATION COMPLETE.\n');
end
