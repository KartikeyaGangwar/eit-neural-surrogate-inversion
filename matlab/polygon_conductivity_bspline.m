function [elem_sigma, dSigma_dTheta, boundary_points] = ...
    polygon_conductivity_bspline(centres, theta, sigma_bg, sigma_inc, alpha)

% ============================================================
% VECTORIZED CLOSED B-SPLINE CONDUCTIVITY & JACOBIAN
% ============================================================

n_control = 32;
n_theta = 2 * n_control;

if length(theta) ~= n_theta
    error('Expected %d parameters, got %d.', n_theta, length(theta));
end

N = size(centres,1);
px = centres(:,1);
py = centres(:,2);

n_boundary_samples = 1024;
mollifier_eps = 0.05;
eps2 = mollifier_eps^2;
series_threshold = 1e-10;

% ------------------------------------------------------------
% CORRECT PARAMETER RECONSTRUCTION
% theta = control_points(:) -> [X1...X32, Y1...Y32]'
% ------------------------------------------------------------
control_points = [theta(1:n_control), theta(n_control+1:n_theta)];

% ------------------------------------------------------------
% Vectorized B-spline generation
% ------------------------------------------------------------
u = linspace(0, n_control, n_boundary_samples + 1)';
u(end) = [];
du = n_control / n_boundary_samples;

segment = floor(u);
t = u - segment;

% Cubic B-spline basis
B0 = (1-t).^3 / 6;
B1 = (3*t.^3 - 6*t.^2 + 4) / 6;
B2 = (-3*t.^3 + 3*t.^2 + 3*t + 1) / 6;
B3 = t.^3 / 6;

% Derivatives of basis w.r.t t
dB0 = -(1-t).^2 / 2;
dB1 = (3*t.^2 - 4*t) / 2;
dB2 = (-3*t.^2 + 2*t + 1) / 2;
dB3 = t.^2 / 2;

% Control point indices (1-based, wrapped)
i0 = mod(segment, n_control) + 1;
i1 = mod(segment + 1, n_control) + 1;
i2 = mod(segment + 2, n_control) + 1;
i3 = mod(segment + 3, n_control) + 1;

% Boundary points
qx = B0.*control_points(i0,1) + B1.*control_points(i1,1) + B2.*control_points(i2,1) + B3.*control_points(i3,1);
qy = B0.*control_points(i0,2) + B1.*control_points(i1,2) + B2.*control_points(i2,2) + B3.*control_points(i3,2);
boundary_points = [qx, qy];

% Boundary tangents
qtx = dB0.*control_points(i0,1) + dB1.*control_points(i1,1) + dB2.*control_points(i2,1) + dB3.*control_points(i3,1);
qty = dB0.*control_points(i0,2) + dB1.*control_points(i1,2) + dB2.*control_points(i2,2) + dB3.*control_points(i3,2);

% ------------------------------------------------------------
% Divergence-Theorem Boundary Integral & Analytic Derivative
% ------------------------------------------------------------
H = zeros(N, 1);
dH_dTheta = zeros(N, n_theta);

for m = 1:n_boundary_samples

    qxm = qx(m); qym = qy(m);
    qtxm = qtx(m); qtym = qty(m);

    rx = qxm - px;
    ry = qym - py;

    s = rx.^2 + ry.^2;
    Ncross = rx .* qtym - ry .* qtxm;

    A = zeros(N,1);
    Aprime = zeros(N,1);

    small = s < series_threshold;
    normal = ~small;

    if any(small)
        ss = s(small);
        A(small) = 1/eps2 - ss/(2*eps2^2) + (ss.^2)/(6*eps2^3);
        Aprime(small) = -1/(2*eps2^2) + ss/(3*eps2^3);
    end

    if any(normal)
        sn = s(normal);
        e = exp(-sn/eps2);
        A(normal) = (1 - e) ./ sn;
        Aprime(normal) = ((1 + sn/eps2).*e - 1) ./ (sn.^2);
    end

    integrand = A .* Ncross;
    H = H + (du/(2*pi)) .* integrand;

    % Distribute derivatives strictly to the 4 active control points
    idx = [i0(m), i1(m), i2(m), i3(m)];
    B_vals = [B0(m), B1(m), B2(m), B3(m)];
    dB_vals = [dB0(m), dB1(m), dB2(m), dB3(m)];

    for j = 1:4
        cp = idx(j);
        bj = B_vals(j);
        dbj = dB_vals(j);

        % Derivative w.r.t X coordinate (index cp)
        ds_dx = 2 .* rx .* bj;
        dN_dx = bj .* qtym - ry .* dbj;
        dH_dTheta(:, cp) = dH_dTheta(:, cp) + (du/(2*pi)) .* (Aprime .* ds_dx .* Ncross + A .* dN_dx);

        % Derivative w.r.t Y coordinate (index n_control + cp)
        ds_dy = 2 .* ry .* bj;
        dN_dy = -bj .* qtxm + rx .* dbj;
        dH_dTheta(:, n_control + cp) = dH_dTheta(:, n_control + cp) + (du/(2*pi)) .* (Aprime .* ds_dy .* Ncross + A .* dN_dy);
    end
end

% ------------------------------------------------------------
% Bounded smooth occupancy mapping (tanh)
% ------------------------------------------------------------
occ_arg = alpha .* (H - 0.5);
s_map = 0.5 .* (1 + tanh(occ_arg));

ds_dH = 0.5 .* alpha .* (1 - tanh(occ_arg).^2);
ds_dTheta = ds_dH .* dH_dTheta;

% ------------------------------------------------------------
% Final Conductivity Mapping
% ------------------------------------------------------------
delta_sigma = sigma_inc - sigma_bg;
elem_sigma = sigma_bg + delta_sigma .* s_map;
dSigma_dTheta = delta_sigma .* ds_dTheta;

end
