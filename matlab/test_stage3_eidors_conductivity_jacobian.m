% ============================================================
% STAGE 3
% DIRECT VALIDATION OF EIDORS CONDUCTIVITY JACOBIAN
%
% PURPOSE
%
% Validate:
%
%       J_sigma = dV / dSigma
%
% independently from the B-spline geometry derivative.
%
% We DO NOT perturb all FEM elements.
%
% Instead we test several normalized conductivity directions:
%
%       dSigma_direction
%
% and compare:
%
%       J_sigma * dSigma_direction
%
% against independent central finite differences:
%
%       [ V(sigma + h*d) - V(sigma - h*d) ] / (2h)
%
% This is a directional derivative test.
%
% ============================================================


clear;
clc;


% ============================================================
% 1. START EIDORS
% ============================================================

run( ...
    'C:/path/to/eidors-v3.12-ng/eidors/startup.m');


fprintf('\nEIDORS version: ');
disp(eidors_obj('eidors_version'));


% ============================================================
% 2. PHYSICAL CONFIGURATION
% ============================================================

n_elec = 16;


sigma_background = ...
    1.0;


sigma_inclusion = ...
    1.0e-4;


current_amplitude = ...
    1.0;


electrode_width = ...
    0.20;


electrode_rfnum = ...
    10;


z_contact = ...
    0.01;


domain_radius = ...
    1.0;


maxsz = ...
    0.05;


alpha = ...
    80;


% ============================================================
% 3. B-SPLINE GEOMETRY
%
% Same geometry used in Stage 1 and Stage 2.
% ============================================================

control_points = [

     0.45   0.00
     0.43   0.10
     0.36   0.19
     0.25   0.27
     0.12   0.31
    -0.02   0.30
    -0.15   0.25
    -0.28   0.18
    -0.38   0.07
    -0.42  -0.06
    -0.38  -0.18
    -0.29  -0.27
    -0.17  -0.31
    -0.04  -0.30
     0.06  -0.24
     0.03  -0.13
    -0.05  -0.06
    -0.10   0.03
    -0.07   0.12
     0.02   0.17
     0.13   0.16
     0.22   0.10
     0.28   0.00
     0.25  -0.10
     0.18  -0.17
     0.27  -0.22
     0.38  -0.19
     0.46  -0.12
     0.49  -0.04
     0.48   0.02
     0.47   0.05
     0.45   0.00

];


theta = ...
    control_points(:);


% ============================================================
% 4. STIMULATION PROTOCOL
% ============================================================

measurement_matrix = ...
    eye(n_elec) - ...
    ones(n_elec) / n_elec;


stim_list = ...
    struct([]);


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


fprintf('\n========================================\n');
fprintf('STAGE 3: EIDORS CONDUCTIVITY JACOBIAN\n');
fprintf('========================================\n');

fprintf('Electrodes         : %d\n', n_elec);
fprintf('Injection patterns : %d\n', length(stim_list));
fprintf('Mesh maxsz         : %.6f\n', maxsz);


% ============================================================
% 5. CREATE OUTER CIRCULAR FEM MODEL
% ============================================================

n_boundary_points = ...
    256;


boundary_theta = ...
    linspace( ...
        0, ...
        2*pi, ...
        n_boundary_points + 1)';


boundary_theta(end) = [];


circle_boundary = [

    domain_radius * cos(boundary_theta), ...
    domain_radius * sin(boundary_theta)

];


shape = {
    circle_boundary, ...
    maxsz
};


fprintf('\nGenerating FEM mesh...\n');


mdl = ...
    ng_mk_2d_model( ...
        shape, ...
        n_elec, ...
        [electrode_width electrode_rfnum]);


% ============================================================
% 6. CONTACT IMPEDANCE
% ============================================================

for e = 1:n_elec

    mdl.electrode(e).z_contact = ...
        z_contact;

end


% ============================================================
% 7. STIMULATION
% ============================================================

mdl.stimulation = ...
    stim_list;


mdl.solve = ...
    @fwd_solve_1st_order;


mdl.system_mat = ...
    @system_mat_1st_order;


mdl.normalize_measurements = ...
    0;


% ============================================================
% 8. FEM ELEMENT CENTRES
% ============================================================

nodes = ...
    mdl.nodes;


elems = ...
    mdl.elems;


N = ...
    size(elems,1);


centres = ...
    zeros(N,2);


for k = 1:N

    centres(k,:) = ...
        mean( ...
            nodes(elems(k,:),:), ...
            1);

end


fprintf('\nMesh information:\n');

fprintf( ...
    'Nodes              : %d\n', ...
    size(nodes,1));


fprintf( ...
    'FEM elements       : %d\n', ...
    N);


fprintf( ...
    'Electrodes         : %d\n', ...
    length(mdl.electrode));


% ============================================================
% 9. CREATE BASE B-SPLINE CONDUCTIVITY
% ============================================================

fprintf('\nEvaluating B-spline conductivity...\n');


sigma0 = ...
    polygon_conductivity_bspline( ...
        centres, ...
        theta, ...
        sigma_background, ...
        sigma_inclusion, ...
        alpha);


% ============================================================
% 10. BASE IMAGE
% ============================================================

img0 = ...
    eidors_obj( ...
        'image', ...
        'Stage 3 conductivity Jacobian test', ...
        'elem_data', ...
        sigma0, ...
        'fwd_model', ...
        mdl);


% ============================================================
% 11. BASE FORWARD SOLVE
% ============================================================

fprintf('\nSolving base forward problem...\n');


data0 = ...
    fwd_solve(img0);


V0 = ...
    data0.meas;


fprintf( ...
    'Measurements       : %d\n', ...
    numel(V0));


% ============================================================
% 12. CALCULATE EIDORS ANALYTIC J_sigma
% ============================================================

fprintf('\nCalculating EIDORS J_sigma...\n');


J_sigma = ...
    calc_jacobian(img0);


fprintf( ...
    'J_sigma size       : %d x %d\n', ...
    size(J_sigma,1), ...
    size(J_sigma,2));


if size(J_sigma,2) ~= N

    error( ...
        ['Unexpected J_sigma dimensions. ', ...
         'Expected %d conductivity columns, got %d.'], ...
        N, ...
        size(J_sigma,2));

end


% ============================================================
% 13. CONDUCTIVITY DIRECTION TESTS
%
% We test several independent directions.
%
% Each direction is normalized to unit Euclidean norm.
%
% This avoids testing only a special FEM-element direction.
% ============================================================

n_directions = ...
    6;


fprintf('\nGenerating conductivity test directions...\n');


rng(12345);


D = ...
    zeros(N,n_directions);


% ------------------------------------------------------------
% Direction 1
%
% Uniform conductivity perturbation.
% ------------------------------------------------------------

d = ...
    ones(N,1);


d = ...
    d / norm(d);


D(:,1) = ...
    d;


% ------------------------------------------------------------
% Direction 2
%
% Random direction.
% ------------------------------------------------------------

d = ...
    randn(N,1);


d = ...
    d / norm(d);


D(:,2) = ...
    d;


% ------------------------------------------------------------
% Direction 3
%
% Second independent random direction.
% ------------------------------------------------------------

d = ...
    randn(N,1);


d = ...
    d / norm(d);


D(:,3) = ...
    d;


% ------------------------------------------------------------
% Direction 4
%
% Localized perturbation near one region.
% ------------------------------------------------------------

centre_target = ...
    [0.15 0.05];


distance_to_target = ...
    sqrt( ...
        (centres(:,1) - centre_target(1)).^2 + ...
        (centres(:,2) - centre_target(2)).^2);


d = ...
    exp( ...
        -(distance_to_target / 0.15).^2);


d = ...
    d / norm(d);


D(:,4) = ...
    d;


% ------------------------------------------------------------
% Direction 5
%
% Smooth spatial sinusoidal perturbation.
% ------------------------------------------------------------

d = ...
    sin( ...
        4 * centres(:,1)) .* ...
    cos( ...
        3 * centres(:,2));


d = ...
    d / norm(d);


D(:,5) = ...
    d;


% ------------------------------------------------------------
% Direction 6
%
% Random signed direction.
% ------------------------------------------------------------

d = ...
    randn(N,1);


d = ...
    d / norm(d);


D(:,6) = ...
    d;


% ============================================================
% 14. FD STEP SIZES
% ============================================================

h_list = [

    1e-2
    1e-3
    1e-4
    1e-5
    1e-6

];


% ============================================================
% 15. VALIDATION
% ============================================================

fprintf('\n========================================\n');
fprintf('DIRECT J_sigma DIRECTIONAL VALIDATION\n');
fprintf('========================================\n');


for id = 1:n_directions


    d = ...
        D(:,id);


    fprintf('\n');
    fprintf( ...
        '========== Direction %d / %d ==========\n', ...
        id, ...
        n_directions);


    % --------------------------------------------------------
    % Analytic directional derivative
    %
    %       J_sigma * d
    % --------------------------------------------------------

    analytic_direction = ...
        J_sigma * d;


    fprintf( ...
        '||J_sigma*d|| = %.12e\n', ...
        norm(analytic_direction));


    for ih = 1:length(h_list)


        h = ...
            h_list(ih);


        % ----------------------------------------------------
        % sigma + h*d
        % ----------------------------------------------------

        sigma_plus = ...
            sigma0 + ...
            h .* d;


        img_plus = ...
            eidors_obj( ...
                'image', ...
                'Stage3 plus', ...
                'elem_data', ...
                sigma_plus, ...
                'fwd_model', ...
                mdl);


        data_plus = ...
            fwd_solve(img_plus);


        % ----------------------------------------------------
        % sigma - h*d
        % ----------------------------------------------------

        sigma_minus = ...
            sigma0 - ...
            h .* d;


        img_minus = ...
            eidors_obj( ...
                'image', ...
                'Stage3 minus', ...
                'elem_data', ...
                sigma_minus, ...
                'fwd_model', ...
                mdl);


        data_minus = ...
            fwd_solve(img_minus);


        % ----------------------------------------------------
        % Central FD directional derivative
        % ----------------------------------------------------

        fd_direction = ...
            ( ...
                data_plus.meas - ...
                data_minus.meas ...
            ) / ...
            (2*h);


        % ----------------------------------------------------
        % Relative error
        % ----------------------------------------------------

        numerator = ...
            norm( ...
                fd_direction - ...
                analytic_direction);


        denominator = ...
            norm(analytic_direction);


        if denominator < 1e-14

            relative_error = ...
                numerator;

        else

            relative_error = ...
                numerator / ...
                denominator;

        end


        fprintf( ...
            'h = %.1e : relative error = %.12e\n', ...
            h, ...
            relative_error);


    end

end


% ============================================================
% 16. SUMMARY
% ============================================================

fprintf('\n\n========================================\n');
fprintf('STAGE 3 SUMMARY\n');
fprintf('========================================\n');


fprintf('\n');

fprintf( ...
    'J_sigma dimensions = %d x %d\n', ...
    size(J_sigma,1), ...
    size(J_sigma,2));


fprintf( ...
    'Conductivity directions tested = %d\n', ...
    n_directions);


fprintf( ...
    'FD steps tested = %d\n', ...
    length(h_list));


fprintf('\n');

fprintf( ...
    'Interpretation:\n');


fprintf( ...
    'For a correct J_sigma, the relative error should\n');


fprintf( ...
    'decrease as h becomes smaller until numerical\n');


fprintf( ...
    'solver precision begins to dominate.\n');


fprintf('\n');

fprintf('STAGE 3 COMPLETED\n');

fprintf('========================================\n');
