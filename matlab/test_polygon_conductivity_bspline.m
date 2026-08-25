% ============================================================
% STAGE 2: CLOSED B-SPLINE GEOMETRY JACOBIAN (EIDORS)
% ============================================================
clear; clc;

% ============================================================
% 1. Start EIDORS
% ============================================================
run('C:/path/to/eidors-v3.12-ng/eidors/startup.m');
fprintf('\nEIDORS version: ');
disp(eidors_obj('eidors_version'));

% ============================================================
% 2. Fixed physical configuration
% ============================================================
n_elec = 16;
sigma_background = 1.0;
sigma_inclusion = 1.0e-4;
current_amplitude = 1.0;

electrode_width = 0.20;
electrode_rfnum = 10;
z_contact = 0.01;
domain_radius = 1.0;
mesh_size_list = [0.05];
alpha = 80;

n_control = 32;
n_theta = 2 * n_control;

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

theta = control_points(:);
if length(theta) ~= n_theta
    error('theta has %d parameters, expected %d.', length(theta), n_theta);
end

% ============================================================
% 3 & 4. Measurement Matrix & Stimulation
% ============================================================
measurement_matrix = eye(n_elec) - ones(n_elec) / n_elec;
stim_list = struct([]);
p = 0;
for i = 1:n_elec
    for j = i+1:n_elec
        p = p + 1;
        stim_list(p).stim_pattern = zeros(n_elec,1);
        stim_list(p).stim_pattern(i) = current_amplitude;
        stim_list(p).stim_pattern(j) = -current_amplitude;
        stim_list(p).meas_pattern = measurement_matrix;
    end
end

% ============================================================
% 5. FD validation configuration
% ============================================================
fd_test_parameters = [1, 2, 17, 32, 48, 64];
fd_step_list = [1e-2; 1e-3; 1e-4; 1e-5; 1e-6];

% ============================================================
% 7. Execution Loop
% ============================================================
for r = 1:length(mesh_size_list)
    maxsz = mesh_size_list(r);
    fprintf('\nGenerating Mesh (maxsz = %.6f)...\n', maxsz);

    n_boundary_points = 256;
    boundary_theta = linspace(0, 2*pi, n_boundary_points + 1)';
    boundary_theta(end) = [];
    circle_boundary = [domain_radius*cos(boundary_theta), domain_radius*sin(boundary_theta)];

    shape = {circle_boundary, maxsz};
    mdl = ng_mk_2d_model(shape, n_elec, [electrode_width electrode_rfnum]);

    for e = 1:n_elec
        mdl.electrode(e).z_contact = z_contact;
    end
    mdl.stimulation = stim_list;
    mdl.solve = @fwd_solve_1st_order;
    mdl.system_mat = @system_mat_1st_order;
    mdl.normalize_measurements = 0;

    nodes = mdl.nodes;
    elems = mdl.elems;
    centres = zeros(size(elems,1), 2);
    for k = 1:size(elems,1)
        centres(k,:) = mean(nodes(elems(k,:),:), 1);
    end

    % Base evaluation
    [sigma0, dSigma_dTheta, ~] = polygon_conductivity_bspline(centres, theta, sigma_background, sigma_inclusion, alpha);

    img0 = eidors_obj('image', 'base', 'elem_data', sigma0, 'fwd_model', mdl);
    fprintf('\nCalculating EIDORS conductivity Jacobian...\n');
    J_sigma = calc_jacobian(img0);

    fprintf('\nCalculating production geometry Jacobian (J_sigma * dSigma_dTheta)...\n');
    J_current = J_sigma * dSigma_dTheta;

    fprintf('\n========================================\n');
    fprintf('STAGE 2: EIDORS GEOMETRY FD CHECK\n');
    fprintf('========================================\n');

    for kk = 1:length(fd_test_parameters)
        k = fd_test_parameters(kk);
        fprintf('\n--- Parameter %d ---\n', k);

        for ih = 1:length(fd_step_list)
            h = fd_step_list(ih);

            % theta + h
            theta_plus = theta; theta_plus(k) = theta_plus(k) + h;
            sigma_plus = polygon_conductivity_bspline(centres, theta_plus, sigma_background, sigma_inclusion, alpha);
            img_plus = eidors_obj('image', 'plus', 'elem_data', sigma_plus, 'fwd_model', mdl);
            data_plus = fwd_solve(img_plus);

            % theta - h
            theta_minus = theta; theta_minus(k) = theta_minus(k) - h;
            sigma_minus = polygon_conductivity_bspline(centres, theta_minus, sigma_background, sigma_inclusion, alpha);
            img_minus = eidors_obj('image', 'minus', 'elem_data', sigma_minus, 'fwd_model', mdl);
            data_minus = fwd_solve(img_minus);

            % FD Column
            fd_column = (data_plus.meas - data_minus.meas) / (2*h);
            analytic_column = J_current(:,k);

            denom = norm(fd_column, 'fro');
            if denom < 1e-14
                column_error = norm(fd_column - analytic_column, 'fro');
            else
                column_error = norm(fd_column - analytic_column, 'fro') / denom;
            end

            fprintf('h = %1.0e : Relative FD Error = %.12e\n', h, column_error);
        end
    end
end
fprintf('\nSTAGE 2 TEST COMPLETED\n');
