% ============================================================
% STAGE 4
% VALID FEM MESH CONVERGENCE TEST
%
% Purpose:
%   Verify that the production geometry Jacobian
%
%       J_theta = J_sigma * dSigma_dTheta
%
%   converges as the FEM mesh is refined.
%
% IMPORTANT:
%   polygon_conductivity_bspline.m is NOT modified here.
%
%   Its production parameters remain:
%       n_boundary_samples = 1024
%       mollifier_eps      = 0.05
%       alpha              = 80
%
%   Only the FEM mesh size is refined.
%
% Mesh sequence:
%       0.025
%       0.0125
%       0.00625
%
% This is deliberately finer than the previous:
%       0.10, 0.05, 0.025
%
% ============================================================

clear;
clc;

fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4: VALID FEM MESH CONVERGENCE TEST\n');
fprintf('========================================\n');


% ============================================================
% 1. START EIDORS
% ============================================================

run('C:/path/to/eidors-v3.12-ng/eidors/startup.m');

fprintf('\nEIDORS version: ');
disp(eidors_obj('eidors_version'));


% ============================================================
% 2. PHYSICAL CONFIGURATION
% ============================================================

n_elec = 16;

sigma_background = 1.0;
sigma_inclusion  = 1.0e-4;

current_amplitude = 1.0;

electrode_width = 0.20;
electrode_rfnum = 10;

z_contact = 0.01;

domain_radius = 1.0;

alpha = 80;

n_control = 32;
n_theta = 2 * n_control;


% ============================================================
% 3. MESH SEQUENCE
%
% IMPORTANT:
%   eps = 0.05 inside polygon_conductivity_bspline.m
%
%   Therefore the previous 0.10 / 0.05 / 0.025 experiment
%   was too coarse for a clean convergence demonstration.
%
% ============================================================

mesh_size_list = [
    0.025
    0.0125
    0.00625
];

n_meshes = length(mesh_size_list);


fprintf('\nMesh levels:\n');

for r = 1:n_meshes
    fprintf('    %d : maxsz = %.6f\n', ...
        r, mesh_size_list(r));
end


% ============================================================
% 4. FIXED B-SPLINE GEOMETRY
%
% theta ordering:
%
%   [x1 ... x32 y1 ... y32]'
%
% This MUST match polygon_conductivity_bspline.m.
% ============================================================

control_points = [
     0.45   0.00;
     0.43   0.10;
     0.36   0.19;
     0.25   0.27;
     0.12   0.31;
    -0.02   0.30;
    -0.15   0.25;
    -0.28   0.18;
    -0.38   0.07;
    -0.42  -0.06;
    -0.38  -0.18;
    -0.29  -0.27;
    -0.17  -0.31;
    -0.04  -0.30;
     0.06  -0.24;
     0.03  -0.13;
    -0.05  -0.06;
    -0.10   0.03;
    -0.07   0.12;
     0.02   0.17;
     0.13   0.16;
     0.22   0.10;
     0.28   0.00;
     0.25  -0.10;
     0.18  -0.17;
     0.27  -0.22;
     0.38  -0.19;
     0.46  -0.12;
     0.49  -0.04;
     0.48   0.02;
     0.47   0.05;
     0.45   0.00
];

theta = control_points(:);

if length(theta) ~= n_theta
    error( ...
        'theta has %d parameters, expected %d.', ...
        length(theta), n_theta);
end


% ============================================================
% 5. FIXED MEASUREMENT MATRIX
% ============================================================

measurement_matrix = ...
    eye(n_elec) - ones(n_elec) / n_elec;


% ============================================================
% 6. STIMULATION PATTERNS
% ============================================================

stim_list = struct([]);

p = 0;

for i = 1:n_elec

    for j = i+1:n_elec

        p = p + 1;

        stim_list(p).stim_pattern = ...
            zeros(n_elec,1);

        stim_list(p).stim_pattern(i) = ...
            current_amplitude;

        stim_list(p).stim_pattern(j) = ...
            -current_amplitude;

        stim_list(p).meas_pattern = ...
            measurement_matrix;

    end

end


fprintf('\n');
fprintf('Electrodes          : %d\n', n_elec);
fprintf('Injection patterns  : %d\n', length(stim_list));
fprintf('Measurements        : %d\n', ...
    n_elec * length(stim_list));
fprintf('Sigma background    : %.6e\n', sigma_background);
fprintf('Sigma inclusion     : %.6e\n', sigma_inclusion);
fprintf('Contact impedance   : %.6e\n', z_contact);
fprintf('Smooth alpha        : %.6e\n', alpha);
fprintf('B-spline control pts: %d\n', n_control);
fprintf('Geometry parameters : %d\n', n_theta);


% ============================================================
% 7. STORAGE
% ============================================================

Jtheta_all = cell(n_meshes,1);
V_all      = cell(n_meshes,1);

node_count = zeros(n_meshes,1);
elem_count = zeros(n_meshes,1);

Jtheta_norm = zeros(n_meshes,1);
V_norm      = zeros(n_meshes,1);

Jtheta_max = zeros(n_meshes,1);


% ============================================================
% 8. MESH LOOP
% ============================================================

for r = 1:n_meshes

    maxsz = mesh_size_list(r);

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('MESH LEVEL %d / %d\n', r, n_meshes);
    fprintf('Maximum element size = %.6f\n', maxsz);
    fprintf('========================================\n');


    % --------------------------------------------------------
    % Generate outer circular FEM domain
    % --------------------------------------------------------

    fprintf('\nGenerating FEM mesh...\n');

    n_boundary_points = 256;

    boundary_theta = ...
        linspace(0, 2*pi, n_boundary_points + 1)';

    boundary_theta(end) = [];

    circle_boundary = [
        domain_radius .* cos(boundary_theta), ...
        domain_radius .* sin(boundary_theta)
    ];


    shape = {
        circle_boundary, ...
        maxsz
    };


    mdl = ng_mk_2d_model( ...
        shape, ...
        n_elec, ...
        [electrode_width electrode_rfnum]);


    % --------------------------------------------------------
    % Contact impedance
    % --------------------------------------------------------

    for e = 1:n_elec

        mdl.electrode(e).z_contact = ...
            z_contact;

    end


    % --------------------------------------------------------
    % EIT model configuration
    % --------------------------------------------------------

    mdl.stimulation = stim_list;

    mdl.solve = @fwd_solve_1st_order;

    mdl.system_mat = @system_mat_1st_order;

    mdl.normalize_measurements = 0;


    % --------------------------------------------------------
    % Mesh information
    % --------------------------------------------------------

    nodes = mdl.nodes;
    elems = mdl.elems;

    node_count(r) = size(nodes,1);
    elem_count(r) = size(elems,1);


    fprintf('\nMesh information:\n');
    fprintf('Nodes              : %d\n', ...
        node_count(r));

    fprintf('FEM elements       : %d\n', ...
        elem_count(r));

    fprintf('Electrodes         : %d\n', ...
        n_elec);


    % --------------------------------------------------------
    % FEM element centres
    % --------------------------------------------------------

    centres = zeros(size(elems,1),2);

    for k = 1:size(elems,1)

        centres(k,:) = ...
            mean(nodes(elems(k,:),:),1);

    end


    % --------------------------------------------------------
    % Conductivity
    % --------------------------------------------------------

    fprintf('\nEvaluating B-spline conductivity...\n');

    [sigma0, dSigma_dTheta, boundary_points] = ...
        polygon_conductivity_bspline( ...
            centres, ...
            theta, ...
            sigma_background, ...
            sigma_inclusion, ...
            alpha);


    % --------------------------------------------------------
    % Boundary diagnostic
    % --------------------------------------------------------

    fprintf('\nB-spline boundary:\n');

    fprintf('Boundary samples   : %d\n', ...
        size(boundary_points,1));

    fprintf( ...
        'Boundary x-range   : [%+.6f, %+.6f]\n', ...
        min(boundary_points(:,1)), ...
        max(boundary_points(:,1)));

    fprintf( ...
        'Boundary y-range   : [%+.6f, %+.6f]\n', ...
        min(boundary_points(:,2)), ...
        max(boundary_points(:,2)));


    % --------------------------------------------------------
    % Conductivity sanity check
    % --------------------------------------------------------

    fprintf('\nConductivity range:\n');

    fprintf('sigma min          : %.12e\n', ...
        min(sigma0));

    fprintf('sigma max          : %.12e\n', ...
        max(sigma0));

    fprintf('||dSigma/dTheta||  : %.12e\n', ...
        norm(dSigma_dTheta,'fro'));


    % --------------------------------------------------------
    % EIDORS image
    % --------------------------------------------------------

    img0 = eidors_obj( ...
        'image', ...
        'stage4_base', ...
        'elem_data', ...
        sigma0, ...
        'fwd_model', ...
        mdl);


    % --------------------------------------------------------
    % Forward solve
    % --------------------------------------------------------

    fprintf('\nSolving forward problem...\n');

    data0 = fwd_solve(img0);

    V = data0.meas;


    fprintf('Measurements       : %d\n', ...
        length(V));

    fprintf('||V||              : %.12e\n', ...
        norm(V,'fro'));


    % --------------------------------------------------------
    % Conductivity Jacobian
    % --------------------------------------------------------

    fprintf('\nCalculating EIDORS J_sigma...\n');

    J_sigma = calc_jacobian(img0);


    fprintf('J_sigma size       : %d x %d\n', ...
        size(J_sigma,1), ...
        size(J_sigma,2));


    % --------------------------------------------------------
    % Production geometry Jacobian
    %
    % J_theta =
    %
    %      J_sigma * dSigma/dTheta
    %
    % --------------------------------------------------------

    fprintf('\nCalculating production geometry Jacobian...\n');

    J_theta = ...
        J_sigma * dSigma_dTheta;


    fprintf('\nGeometry Jacobian:\n');

    fprintf('J_theta            : %d x %d\n', ...
        size(J_theta,1), ...
        size(J_theta,2));

    fprintf('||J_theta||_F      : %.12e\n', ...
        norm(J_theta,'fro'));

    fprintf('max |J_theta|      : %.12e\n', ...
        max(abs(J_theta(:))));


    % --------------------------------------------------------
    % Store results
    % --------------------------------------------------------

    Jtheta_all{r} = J_theta;

    V_all{r} = V;

    Jtheta_norm(r) = ...
        norm(J_theta,'fro');

    Jtheta_max(r) = ...
        max(abs(J_theta(:)));

    V_norm(r) = ...
        norm(V,'fro');


    % --------------------------------------------------------
    % Clear large temporary objects
    % --------------------------------------------------------

    clear J_sigma;
    clear dSigma_dTheta;
    clear img0;
    clear data0;

end


% ============================================================
% 9. BASIC MESH CONSISTENCY
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('BASIC MESH CONSISTENCY CHECK\n');
fprintf('========================================\n');

for r = 1:n_meshes

    fprintf( ...
        'maxsz = %.6f : nodes = %d, elements = %d\n', ...
        mesh_size_list(r), ...
        node_count(r), ...
        elem_count(r));

end


% ============================================================
% 10. FINEST MESH REFERENCE
% ============================================================

ref_idx = n_meshes;

J_ref = Jtheta_all{ref_idx};
V_ref = V_all{ref_idx};

J_ref_norm = norm(J_ref,'fro');
V_ref_norm = norm(V_ref,'fro');


fprintf('\n');
fprintf('========================================\n');
fprintf('REFERENCE MESH\n');
fprintf('========================================\n');

fprintf( ...
    'Reference maxsz = %.6f\n', ...
    mesh_size_list(ref_idx));

fprintf( ...
    'Reference ||J_theta|| = %.12e\n', ...
    J_ref_norm);

fprintf( ...
    'Reference ||V||       = %.12e\n', ...
    V_ref_norm);


% ============================================================
% 11. REFERENCE-NORMALIZED ERRORS
% ============================================================

J_reference_error = zeros(n_meshes,1);
V_reference_error = zeros(n_meshes,1);

for r = 1:n_meshes

    J_reference_error(r) = ...
        norm( ...
            Jtheta_all{r} - J_ref, ...
            'fro') / J_ref_norm;


    V_reference_error(r) = ...
        norm( ...
            V_all{r} - V_ref, ...
            'fro') / V_ref_norm;

end


fprintf('\n');
fprintf('========================================\n');
fprintf('J_THETA CONVERGENCE TO FINE REFERENCE\n');
fprintf('========================================\n');

for r = 1:n_meshes

    fprintf( ...
        'maxsz = %.6f : relative J_theta error = %.12e\n', ...
        mesh_size_list(r), ...
        J_reference_error(r));

end


fprintf('\n');
fprintf('========================================\n');
fprintf('FORWARD DATA CONVERGENCE TO FINE REFERENCE\n');
fprintf('========================================\n');

for r = 1:n_meshes

    fprintf( ...
        'maxsz = %.6f : relative V error = %.12e\n', ...
        mesh_size_list(r), ...
        V_reference_error(r));

end


% ============================================================
% 12. SUCCESSIVE MESH CHANGES
%
% Compare:
%
%   coarse -> medium
%   medium -> fine
%
% A healthy convergence sequence should generally show the
% second change becoming smaller.
% ============================================================

J_successive_change = zeros(n_meshes-1,1);
V_successive_change = zeros(n_meshes-1,1);

for r = 1:(n_meshes-1)

    J1 = Jtheta_all{r};
    J2 = Jtheta_all{r+1};

    V1 = V_all{r};
    V2 = V_all{r+1};


    J_successive_change(r) = ...
        norm(J2 - J1,'fro') / ...
        norm(J2,'fro');


    V_successive_change(r) = ...
        norm(V2 - V1,'fro') / ...
        norm(V2,'fro');

end


fprintf('\n');
fprintf('========================================\n');
fprintf('SUCCESSIVE J_THETA CHANGES\n');
fprintf('========================================\n');

for r = 1:(n_meshes-1)

    fprintf( ...
        '%.6f -> %.6f : relative change = %.12e\n', ...
        mesh_size_list(r), ...
        mesh_size_list(r+1), ...
        J_successive_change(r));

end


fprintf('\n');
fprintf('========================================\n');
fprintf('SUCCESSIVE FORWARD-DATA CHANGES\n');
fprintf('========================================\n');

for r = 1:(n_meshes-1)

    fprintf( ...
        '%.6f -> %.6f : relative change = %.12e\n', ...
        mesh_size_list(r), ...
        mesh_size_list(r+1), ...
        V_successive_change(r));

end


% ============================================================
% 13. COLUMN-WISE J_THETA CONVERGENCE
%
% This checks every one of the 64 geometry parameters.
%
% The reference is the finest mesh.
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('COLUMN-WISE J_THETA CONVERGENCE\n');
fprintf('REFERENCE = FINEST MESH\n');
fprintf('========================================\n');


column_error = ...
    zeros(n_meshes,n_theta);


for r = 1:n_meshes

    J_current = Jtheta_all{r};

    for k = 1:n_theta

        ref_col = J_ref(:,k);

        denom = norm(ref_col);

        if denom > 1e-14

            column_error(r,k) = ...
                norm( ...
                    J_current(:,k) - ref_col) / ...
                denom;

        else

            column_error(r,k) = ...
                norm( ...
                    J_current(:,k) - ref_col);

        end

    end

end


for r = 1:n_meshes

    fprintf('\nMesh maxsz = %.6f\n', ...
        mesh_size_list(r));

    fprintf( ...
        'Maximum column error = %.12e\n', ...
        max(column_error(r,:)));

    fprintf( ...
        'Median column error  = %.12e\n', ...
        median(column_error(r,:)));

end


% ============================================================
% 14. OVERALL CONVERGENCE INTERPRETATION
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4 CONVERGENCE INTERPRETATION\n');
fprintf('========================================\n');


if n_meshes >= 3

    J_decreasing = ...
        J_successive_change(end) < ...
        J_successive_change(end-1);

    V_decreasing = ...
        V_successive_change(end) < ...
        V_successive_change(end-1);


    fprintf('\nJ_theta:\n');

    fprintf( ...
        'coarse -> medium = %.12e\n', ...
        J_successive_change(1));

    fprintf( ...
        'medium -> fine   = %.12e\n', ...
        J_successive_change(2));


    fprintf('\nForward data:\n');

    fprintf( ...
        'coarse -> medium = %.12e\n', ...
        V_successive_change(1));

    fprintf( ...
        'medium -> fine   = %.12e\n', ...
        V_successive_change(2));


    if J_decreasing

        fprintf('\n');
        fprintf( ...
            'J_theta: SUCCESSIVE CHANGE DECREASED.\n');

        fprintf( ...
            'The geometry Jacobian shows mesh refinement convergence.\n');

    else

        fprintf('\n');
        fprintf( ...
            'J_theta: SUCCESSIVE CHANGE DID NOT DECREASE.\n');

        fprintf( ...
            'Do NOT claim mesh convergence yet.\n');

    end


    if V_decreasing

        fprintf( ...
            'Forward data: SUCCESSIVE CHANGE DECREASED.\n');

    else

        fprintf( ...
            'Forward data: SUCCESSIVE CHANGE DID NOT DECREASE.\n');

    end

end


% ============================================================
% 15. FINAL SUMMARY
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4 FINAL SUMMARY\n');
fprintf('========================================\n');

fprintf( ...
    'Geometry parameters : %d\n', ...
    n_theta);

fprintf( ...
    'B-spline control pts: %d\n', ...
    n_control);

fprintf( ...
    'Meshes tested       : %d\n', ...
    n_meshes);


for r = 1:n_meshes

    fprintf( ...
        'Mesh %.6f : %d nodes, %d elements\n', ...
        mesh_size_list(r), ...
        node_count(r), ...
        elem_count(r));

end


fprintf('\n');
fprintf( ...
    'Fine reference mesh : %.6f\n', ...
    mesh_size_list(ref_idx));

fprintf( ...
    'Fine ||J_theta||_F  : %.12e\n', ...
    J_ref_norm);

fprintf( ...
    'Fine ||V||          : %.12e\n', ...
    V_ref_norm);


fprintf('\n');
fprintf('J_theta reference errors:\n');

for r = 1:n_meshes

    fprintf( ...
        '    maxsz %.6f : %.12e\n', ...
        mesh_size_list(r), ...
        J_reference_error(r));

end


fprintf('\n');
fprintf('Forward-data reference errors:\n');

for r = 1:n_meshes

    fprintf( ...
        '    maxsz %.6f : %.12e\n', ...
        mesh_size_list(r), ...
        V_reference_error(r));

end


fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4 TEST COMPLETED\n');
fprintf('========================================\n');

