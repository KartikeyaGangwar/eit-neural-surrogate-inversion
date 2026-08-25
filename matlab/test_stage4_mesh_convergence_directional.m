% ============================================================
% STAGE 4
% VALID MESH CONVERGENCE + DIRECTIONAL GEOMETRY JACOBIAN TEST
%
% IMPORTANT:
%
% This test intentionally DOES NOT calculate the full
% conductivity Jacobian on the finest mesh.
%
% Previous Stage 4 failed here:
%
%   J_sigma = 1920 x 167235
%
% which caused:
%
%   out of memory or dimension too large for Octave's index type
%
% Therefore:
%
%   Mesh 1:
%       full J_sigma + full J_theta
%
%   Mesh 2:
%       full J_sigma + full J_theta
%
%   Mesh 3:
%       NO full J_sigma
%       NO full J_theta
%       directional finite-difference validation only
%
%
% The production conductivity function is NOT modified.
%
% polygon_conductivity_bspline.m remains exactly as validated
% in Stages 1-3.
%
% ============================================================

clear;
clc;


fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4: MESH CONVERGENCE + DIRECTIONAL TEST\n');
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
% 3. MESH LEVELS
%
% These are the meshes that were successfully generated.
%
% ============================================================

mesh_size_list = [
    0.02500
    0.01250
    0.00625
];

n_meshes = length(mesh_size_list);


% ============================================================
% 4. FD CONFIGURATION
%
% Stage 2 already showed that h = 1e-4 is in the good
% convergence region.
%
% We therefore use h = 1e-4 for the directional Stage 4
% validation to avoid unnecessary forward solves.
%
% ============================================================

fd_h = 1.0e-4;


% ============================================================
% 5. FIXED GEOMETRY
%
% theta ordering:
%
%   [x1 ... x32 y1 ... y32]'
%
% This matches polygon_conductivity_bspline.m.
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
% 6. MEASUREMENT MATRIX
% ============================================================

measurement_matrix = ...
    eye(n_elec) - ones(n_elec) / n_elec;


% ============================================================
% 7. STIMULATION PATTERNS
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

fprintf('Sigma background    : %.6e\n', ...
    sigma_background);

fprintf('Sigma inclusion     : %.6e\n', ...
    sigma_inclusion);

fprintf('Contact impedance   : %.6e\n', ...
    z_contact);

fprintf('Smooth alpha        : %.6e\n', ...
    alpha);

fprintf('B-spline control pts: %d\n', ...
    n_control);

fprintf('Geometry parameters : %d\n', ...
    n_theta);

fprintf('Directional FD h    : %.1e\n', ...
    fd_h);


% ============================================================
% 8. FIXED DIRECTION SET
%
% These directions are deliberately deterministic.
%
% Each direction is normalized so that:
%
%       ||d||_2 = 1
%
% The same directions are used on every mesh.
%
% ============================================================

direction_matrix = zeros(n_theta,6);


% Direction 1: x displacement of control point 1

direction_matrix(1,1) = 1;


% Direction 2: y displacement of control point 1

direction_matrix(n_control+1,2) = 1;


% Direction 3: x displacement of control point 17

direction_matrix(17,3) = 1;


% Direction 4: y displacement of control point 17

direction_matrix(n_control+17,4) = 1;


% Direction 5: collective x deformation

direction_matrix(1:n_control,5) = 1;


% Direction 6: collective y deformation

direction_matrix(n_control+1:n_theta,6) = 1;


for d = 1:6

    direction_matrix(:,d) = ...
        direction_matrix(:,d) / ...
        norm(direction_matrix(:,d));

end


n_directions = size(direction_matrix,2);


fprintf('\n');
fprintf('Directional tests   : %d\n', n_directions);


% ============================================================
% 9. STORAGE
% ============================================================

V_all = cell(n_meshes,1);

sigma_all = cell(n_meshes,1);

dSigma_all = cell(n_meshes,1);

Jtheta_all = cell(n_meshes,1);

has_Jtheta = false(n_meshes,1);

node_count = zeros(n_meshes,1);

elem_count = zeros(n_meshes,1);

V_norm = zeros(n_meshes,1);

Jtheta_norm = NaN(n_meshes,1);


% Directional analytic Jacobian
%
% Rows:
%   mesh
%
% Columns:
%   direction

directional_analytic = ...
    NaN(n_meshes,n_directions);


% Directional finite-difference derivative

directional_fd = ...
    NaN(n_meshes,n_directions);


% Relative analytic-vs-FD errors

directional_error = ...
    NaN(n_meshes,n_directions);


% ============================================================
% 10. MESH LOOP
% ============================================================

for r = 1:n_meshes

    maxsz = mesh_size_list(r);


    fprintf('\n');
    fprintf('========================================\n');
    fprintf('MESH LEVEL %d / %d\n', r, n_meshes);
    fprintf('Maximum element size = %.6f\n', maxsz);
    fprintf('========================================\n');


    % --------------------------------------------------------
    % Generate circular FEM domain
    % --------------------------------------------------------

    fprintf('\nGenerating FEM mesh...\n');


    n_boundary_points = 256;


    boundary_theta = ...
        linspace( ...
            0, ...
            2*pi, ...
            n_boundary_points + 1)';


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
    % EIT model
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


    node_count(r) = ...
        size(nodes,1);

    elem_count(r) = ...
        size(elems,1);


    fprintf('\nMesh information:\n');

    fprintf( ...
        'Nodes              : %d\n', ...
        node_count(r));

    fprintf( ...
        'FEM elements       : %d\n', ...
        elem_count(r));

    fprintf( ...
        'Electrodes         : %d\n', ...
        n_elec);


    % --------------------------------------------------------
    % Element centres
    % --------------------------------------------------------

    centres = zeros( ...
        size(elems,1), ...
        2);


    for k = 1:size(elems,1)

        centres(k,:) = ...
            mean( ...
                nodes(elems(k,:),:), ...
                1);

    end


    % --------------------------------------------------------
    % Base conductivity
    % --------------------------------------------------------

    fprintf('\nEvaluating B-spline conductivity...\n');


    [sigma0, dSigma_dTheta, boundary_points] = ...
        polygon_conductivity_bspline( ...
            centres, ...
            theta, ...
            sigma_background, ...
            sigma_inclusion, ...
            alpha);


    sigma_all{r} = sigma0;

    dSigma_all{r} = dSigma_dTheta;


    fprintf('\nB-spline boundary:\n');

    fprintf( ...
        'Boundary samples   : %d\n', ...
        size(boundary_points,1));

    fprintf( ...
        'Boundary x-range   : [%+.6f, %+.6f]\n', ...
        min(boundary_points(:,1)), ...
        max(boundary_points(:,1)));

    fprintf( ...
        'Boundary y-range   : [%+.6f, %+.6f]\n', ...
        min(boundary_points(:,2)), ...
        max(boundary_points(:,2)));


    fprintf('\nConductivity range:\n');

    fprintf( ...
        'sigma min          : %.12e\n', ...
        min(sigma0));

    fprintf( ...
        'sigma max          : %.12e\n', ...
        max(sigma0));

    fprintf( ...
        '||dSigma/dTheta||  : %.12e\n', ...
        norm(dSigma_dTheta,'fro'));


    % --------------------------------------------------------
    % Base image
    % --------------------------------------------------------

    img0 = eidors_obj( ...
        'image', ...
        'stage4_base', ...
        'elem_data', ...
        sigma0, ...
        'fwd_model', ...
        mdl);


    % --------------------------------------------------------
    % Base forward solve
    % --------------------------------------------------------

    fprintf('\nSolving base forward problem...\n');


    data0 = fwd_solve(img0);


    V0 = data0.meas;


    V_all{r} = V0;

    V_norm(r) = ...
        norm(V0,'fro');


    fprintf( ...
        'Measurements       : %d\n', ...
        length(V0));

    fprintf( ...
        '||V||              : %.12e\n', ...
        V_norm(r));


    % ========================================================
    % FULL J_THETA ONLY ON MESHES THAT FIT IN MEMORY
    %
    % Mesh 1 and Mesh 2:
    %
    %   J_sigma
    %   J_theta
    %
    % Mesh 3:
    %
    %   deliberately skipped
    %
    % ========================================================

    if r <= 2

        fprintf('\n');
        fprintf( ...
            'Calculating EIDORS J_sigma on this mesh...\n');


        J_sigma = ...
            calc_jacobian(img0);


        fprintf( ...
            'J_sigma size       : %d x %d\n', ...
            size(J_sigma,1), ...
            size(J_sigma,2));


        fprintf('\n');
        fprintf( ...
            'Calculating full J_theta...\n');


        J_theta = ...
            J_sigma * dSigma_dTheta;


        Jtheta_all{r} = J_theta;

        has_Jtheta(r) = true;


        Jtheta_norm(r) = ...
            norm(J_theta,'fro');


        fprintf( ...
            'J_theta size       : %d x %d\n', ...
            size(J_theta,1), ...
            size(J_theta,2));


        fprintf( ...
            '||J_theta||_F      : %.12e\n', ...
            Jtheta_norm(r));


        fprintf( ...
            'max |J_theta|      : %.12e\n', ...
            max(abs(J_theta(:))));


        clear J_sigma;

    else

        fprintf('\n');
        fprintf( ...
            'FINE MESH: full J_sigma/J_theta SKIPPED.\n');

        fprintf( ...
            'Reason: previous 0.00625 mesh exceeded Octave memory.\n');

    end


    % ========================================================
    % DIRECTIONAL GEOMETRY TEST
    %
    % For each direction d:
    %
    % Analytic:
    %
    %   J_theta*d
    %
    % Independent FD:
    %
    %   [V(theta+h*d)-V(theta-h*d)]/(2h)
    %
    % ========================================================

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('DIRECTIONAL GEOMETRY TEST\n');
    fprintf('========================================\n');


    for d = 1:n_directions

        direction = ...
            direction_matrix(:,d);


        fprintf('\n');
        fprintf( ...
            'Direction %d / %d\n', ...
            d, ...
            n_directions);


        % ----------------------------------------------------
        % Conductivity directional derivative
        % ----------------------------------------------------

        dsigma_direction = ...
            dSigma_dTheta * direction;


        % ----------------------------------------------------
        % Analytic EIT directional derivative
        %
        % Available only where full J_theta was computed.
        % ----------------------------------------------------

        if has_Jtheta(r)

            analytic_direction = ...
                Jtheta_all{r} * direction;


            directional_analytic(r,d) = ...
                norm(analytic_direction,'fro');

        end


        % ----------------------------------------------------
        % theta + h*d
        % ----------------------------------------------------

        theta_plus = ...
            theta + fd_h .* direction;


        sigma_plus = ...
            polygon_conductivity_bspline( ...
                centres, ...
                theta_plus, ...
                sigma_background, ...
                sigma_inclusion, ...
                alpha);


        img_plus = eidors_obj( ...
            'image', ...
            'stage4_plus', ...
            'elem_data', ...
            sigma_plus, ...
            'fwd_model', ...
            mdl);


        data_plus = ...
            fwd_solve(img_plus);


        V_plus = ...
            data_plus.meas;


        % ----------------------------------------------------
        % theta - h*d
        % ----------------------------------------------------

        theta_minus = ...
            theta - fd_h .* direction;


        sigma_minus = ...
            polygon_conductivity_bspline( ...
                centres, ...
                theta_minus, ...
                sigma_background, ...
                sigma_inclusion, ...
                alpha);


        img_minus = eidors_obj( ...
            'image', ...
            'stage4_minus', ...
            'elem_data', ...
            sigma_minus, ...
            'fwd_model', ...
            mdl);


        data_minus = ...
            fwd_solve(img_minus);


        V_minus = ...
            data_minus.meas;


        % ----------------------------------------------------
        % Central FD
        % ----------------------------------------------------

        fd_direction = ...
            (V_plus - V_minus) ./ ...
            (2 .* fd_h);


        directional_fd(r,d) = ...
            norm(fd_direction,'fro');


        % ----------------------------------------------------
        % Analytic-vs-FD error
        % ----------------------------------------------------

        if has_Jtheta(r)

            denom = ...
                norm(fd_direction,'fro');


            if denom > 1e-14

                directional_error(r,d) = ...
                    norm( ...
                        fd_direction - ...
                        analytic_direction, ...
                        'fro') / denom;

            else

                directional_error(r,d) = ...
                    norm( ...
                        fd_direction - ...
                        analytic_direction, ...
                        'fro');

            end


            fprintf( ...
                '||analytic|| = %.12e\n', ...
                norm(analytic_direction,'fro'));


            fprintf( ...
                '||FD||       = %.12e\n', ...
                norm(fd_direction,'fro'));


            fprintf( ...
                'relative error = %.12e\n', ...
                directional_error(r,d));

        else

            fprintf( ...
                '||FD||       = %.12e\n', ...
                norm(fd_direction,'fro'));

            fprintf( ...
                'analytic J_theta unavailable on this mesh\n');

        end


        % ----------------------------------------------------
        % Cleanup
        % ----------------------------------------------------

        clear sigma_plus;
        clear sigma_minus;
        clear img_plus;
        clear img_minus;
        clear data_plus;
        clear data_minus;

    end


    % --------------------------------------------------------
    % Cleanup mesh-specific large arrays
    % --------------------------------------------------------

    clear img0;
    clear data0;
    clear dSigma_dTheta;
    clear sigma0;
    clear centres;
    clear elems;
    clear nodes;
    clear mdl;

end


% ============================================================
% 11. FORWARD DATA MESH CONVERGENCE
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('FORWARD DATA MESH CONVERGENCE\n');
fprintf('========================================\n');


V_reference = ...
    V_all{n_meshes};


V_reference_norm = ...
    norm(V_reference,'fro');


V_reference_error = ...
    zeros(n_meshes,1);


for r = 1:n_meshes

    V_reference_error(r) = ...
        norm( ...
            V_all{r} - V_reference, ...
            'fro') / ...
        V_reference_norm;


    fprintf( ...
        'maxsz = %.6f : relative V error = %.12e\n', ...
        mesh_size_list(r), ...
        V_reference_error(r));

end


% ============================================================
% 12. SUCCESSIVE FORWARD CONVERGENCE
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('SUCCESSIVE FORWARD DATA CHANGES\n');
fprintf('========================================\n');


V_successive_change = ...
    zeros(n_meshes-1,1);


for r = 1:(n_meshes-1)

    V_successive_change(r) = ...
        norm( ...
            V_all{r+1} - ...
            V_all{r}, ...
            'fro') / ...
        norm( ...
            V_all{r+1}, ...
            'fro');


    fprintf( ...
        '%.6f -> %.6f : relative change = %.12e\n', ...
        mesh_size_list(r), ...
        mesh_size_list(r+1), ...
        V_successive_change(r));

end


% ============================================================
% 13. J_THETA CONVERGENCE WHERE FULL J_THETA EXISTS
%
% This compares mesh 1 against mesh 2.
%
% We intentionally do NOT compare mesh 3 here because a full
% J_theta was not constructed on mesh 3.
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('FULL J_THETA CONVERGENCE\n');
fprintf('========================================\n');


if has_Jtheta(1) && has_Jtheta(2)

    J1 = Jtheta_all{1};

    J2 = Jtheta_all{2};


    J12_error = ...
        norm(J1 - J2,'fro') / ...
        norm(J2,'fro');


    fprintf( ...
        '0.025000 -> 0.012500\n');

    fprintf( ...
        'relative J_theta change = %.12e\n', ...
        J12_error);

else

    fprintf( ...
        'Full J_theta comparison unavailable.\n');

end


% ============================================================
% 14. COLUMN-WISE FULL J_THETA CHECK
% ============================================================

if has_Jtheta(1) && has_Jtheta(2)

    fprintf('\n');
    fprintf('========================================\n');
    fprintf('COLUMN-WISE J_THETA CHANGE\n');
    fprintf('========================================\n');


    J1 = Jtheta_all{1};

    J2 = Jtheta_all{2};


    column_change = ...
        zeros(n_theta,1);


    for k = 1:n_theta

        denom = ...
            norm(J2(:,k));


        if denom > 1e-14

            column_change(k) = ...
                norm( ...
                    J1(:,k) - ...
                    J2(:,k)) / denom;

        else

            column_change(k) = ...
                norm( ...
                    J1(:,k) - ...
                    J2(:,k));

        end


        fprintf( ...
            'parameter %2d : %.12e\n', ...
            k, ...
            column_change(k));

    end


    fprintf('\n');

    fprintf( ...
        'maximum column change = %.12e\n', ...
        max(column_change));

    fprintf( ...
        'median column change  = %.12e\n', ...
        median(column_change));

end


% ============================================================
% 15. DIRECTIONAL FD MESH CONVERGENCE
%
% This is the key fine-mesh test.
%
% Since the fine mesh cannot store J_sigma, compare the actual
% directional derivative obtained from the forward model:
%
%       dV/ddtheta
%
% across meshes.
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('DIRECTIONAL FD MESH CONVERGENCE\n');
fprintf('========================================\n');


directional_fd_reference = ...
    directional_fd(n_meshes,:);


directional_fd_error = ...
    zeros(n_meshes,n_directions);


for r = 1:n_meshes

    fprintf('\nMesh maxsz = %.6f\n', ...
        mesh_size_list(r));


    for d = 1:n_directions

        ref_value = ...
            directional_fd_reference(d);


        % We only have norms here, so compare the scalar
        % directional derivative magnitudes.

        if abs(ref_value) > 1e-14

            directional_fd_error(r,d) = ...
                abs( ...
                    directional_fd(r,d) - ...
                    ref_value) / ...
                abs(ref_value);

        else

            directional_fd_error(r,d) = ...
                abs( ...
                    directional_fd(r,d) - ...
                    ref_value);

        end


        fprintf( ...
            'direction %d : %.12e\n', ...
            d, ...
            directional_fd_error(r,d));

    end

end


% ============================================================
% 16. DIRECTIONAL ANALYTIC VALIDATION SUMMARY
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('ANALYTIC GEOMETRY DIRECTIONAL VALIDATION\n');
fprintf('========================================\n');


for r = 1:n_meshes

    if has_Jtheta(r)

        fprintf('\nMesh maxsz = %.6f\n', ...
            mesh_size_list(r));


        for d = 1:n_directions

            fprintf( ...
                'direction %d : relative analytic-FD error = %.12e\n', ...
                d, ...
                directional_error(r,d));

        end

    else

        fprintf('\nMesh maxsz = %.6f\n', ...
            mesh_size_list(r));

        fprintf( ...
            'Full analytic J_theta unavailable.\n');

        fprintf( ...
            'Independent FD directional derivative was tested.\n');

    end

end


% ============================================================
% 17. FINAL INTERPRETATION
% ============================================================

fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4 INTERPRETATION\n');
fprintf('========================================\n');


% ------------------------------------------------------------
% Forward convergence
% ------------------------------------------------------------

if V_successive_change(2) < ...
   V_successive_change(1)

    fprintf('\n');
    fprintf( ...
        'FORWARD DATA: PASS\n');

    fprintf( ...
        'Successive mesh change decreased.\n');

else

    fprintf('\n');
    fprintf( ...
        'FORWARD DATA: NOT CONVERGED YET\n');

    fprintf( ...
        'Successive mesh change did not decrease.\n');

end


% ------------------------------------------------------------
% Full J_theta comparison
% ------------------------------------------------------------

if has_Jtheta(1) && has_Jtheta(2)

    fprintf('\n');
    fprintf( ...
        'FULL J_THETA: COARSE/MEDIUM COMPARISON AVAILABLE\n');

    fprintf( ...
        'Relative change = %.12e\n', ...
        J12_error);

    fprintf( ...
        'This comparison stops at maxsz = 0.0125 because\n');

    fprintf( ...
        'the full fine-mesh J_sigma cannot fit in Octave memory.\n');

else

    fprintf('\n');
    fprintf( ...
        'FULL J_THETA: NOT AVAILABLE\n');

end


% ------------------------------------------------------------
% Fine directional derivative
% ------------------------------------------------------------

fprintf('\n');
fprintf( ...
    'FINE MESH DIRECTIONAL TEST: AVAILABLE\n');

fprintf( ...
    'The 0.00625 mesh was tested without constructing\n');

fprintf( ...
    'the full 1920 x 167235 conductivity Jacobian.\n');


% ------------------------------------------------------------
% Analytic directional validation
% ------------------------------------------------------------

fprintf('\n');

for r = 1:n_meshes

    if has_Jtheta(r)

        max_error = ...
            max(directional_error(r,:));


        fprintf( ...
            'Mesh %.6f: maximum analytic-FD directional error = %.12e\n', ...
            mesh_size_list(r), ...
            max_error);

    end

end


% ============================================================
% 18. FINAL SUMMARY
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
    'Forward successive changes:\n');

for r = 1:(n_meshes-1)

    fprintf( ...
        '    %.6f -> %.6f : %.12e\n', ...
        mesh_size_list(r), ...
        mesh_size_list(r+1), ...
        V_successive_change(r));

end


fprintf('\n');

fprintf( ...
    'Directional FD fine-mesh reference = %.6f\n', ...
    mesh_size_list(n_meshes));


fprintf('\n');
fprintf('========================================\n');
fprintf('STAGE 4 TEST COMPLETED\n');
fprintf('========================================\n');

