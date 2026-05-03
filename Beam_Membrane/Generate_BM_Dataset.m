%% Generate_BM_Dataset.m
% Generates a large dataset from the Beam-Membrane vocal fold model.
% Outputs a CSV file for transfer learning experiments (BCM -> BM).
%
% Usage:
%   1. Open MATLAB
%   2. cd to this directory (Beam+Membrane-Sean/)
%   3. Run: Generate_BM_Dataset
%
% Output: dataset_BM.csv with columns:
%   a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL
%
% Based on Randomly_Generating_Data_Membrane_Beam_Model.m

clc; clear all; close all;

%% ==================== CONFIGURATION ====================
N_s = 5000;              % Number of simulations
SAVE_EVERY = 50;         % Save progress every N sims
OUTPUT_CSV = 'dataset_BM.csv';

% Set to true to also save waveform data (Am, p_contact) for each sim.
% Warning: this creates large files (~28KB per sim = ~140MB total).
SAVE_WAVEFORMS = false;

%% ==================== SETUP ====================
% Add PhonationModelsCode2 to path (needed for MuscleControlModel)
addpath(genpath('PhonationModelsCode2'));

% Simulation parameters
T = 1;                   % Time duration [s]
c = 350;                 % Sound velocity [m/s]
delta = (10^-2) * 0.25;  % Standard tube length (WRA)
factor = 4;              % Time step subdivision factor
dt = (delta/c) / factor; % Time step
fs = floor(1/dt);        % Sample frequency

% Steady-state time bounds
t1 = 0.5;
t2 = 1;

% Load stiffness, damping, material, and geometric parameters
Stiffness_Damping_Membrane_Beam

%% ==================== PARAMETER RANGES ====================
% Muscle activation ranges
a_TA_min = 0;     a_TA_max = 1;
a_LCA_min = 0.5;  a_LCA_max = 1;
a_PCA_min = 0;    a_PCA_max = 0;   % Fixed at 0
a_CT_min = 0;     a_CT_max = 1;

% Subglottal pressure range [Pa]
Ps_min = 600;
Ps_max = 1000;

% Separation ratio
sr = 1.2;

% Finite difference discretization
Nx = 20;
Ny = 10;

% Convergence angle
Theta_conv = 0.05;

%% ==================== INITIAL CONDITIONS ====================
Wb0 = zeros(Nx, 1);
Wbt0 = zeros(Nx, 1);
[Y, X] = meshgrid(linspace(0,1,Ny), linspace(0,1,Nx));
Wm0 = 1e-9 * sin(pi*X) .* sin(pi*Y);  % Nonzero to enable mucosal wave
Wmt0 = zeros(Nx, Ny);

%% ==================== PRE-ALLOCATE ====================
results = NaN(N_s, 8);  % [a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL]
col_names = {'a_LCA', 'a_IA', 'a_PCA', 'a_CT', 'a_TA', 'Ps', 'F0', 'SPL'};

if SAVE_WAVEFORMS
    if ~exist('waveforms', 'dir')
        mkdir('waveforms');
    end
end

%% ==================== MAIN LOOP ====================
fprintf('================================================================\n');
fprintf('  BEAM-MEMBRANE DATA GENERATION\n');
fprintf('  Target: %d simulations\n', N_s);
fprintf('================================================================\n\n');

tic;
valid_count = 0;
skip_count = 0;
error_count = 0;

for i = 1:N_s
    % Randomly sample parameters
    a_LCA = a_LCA_min + (a_LCA_max - a_LCA_min) * rand;
    a_IA = a_LCA;        % a_IA always equals a_LCA
    a_PCA = a_PCA_min;    % Fixed at 0
    a_CT = a_CT_min + (a_CT_max - a_CT_min) * rand;
    a_TA = a_TA_min + (a_TA_max - a_TA_min) * rand;
    Ps_val = Ps_min + (Ps_max - Ps_min) * rand;

    ActVec = [a_LCA, a_IA, a_PCA, a_CT, a_TA];

    try
        % Posturing simulation
        [eps0, Theta_G] = Posturing_Simulation2(ActVec);
        Theta_G = 0.5 * Theta_G;

        % Stretched VF length
        L = (1 + eps0) * L0;

        % Internal moments and stresses
        [Mu_c_beam, M0_beam, F_beam, d_MUC, d_LIG, d_TA, ...
         b_MUC, b_LIG, b_TA, A_MUC, A_LIG, A_TA, sigma_MUC, ...
         rho_MUC, rho_LIG, rho_TA, Sm, eta] = ...
            Membrane_Beam_Parameters4(b0, A_MUC_0, A_LIG_0, A_TA_0, ...
                rho_MUC_0, rho_LIG_0, rho_TA_0, eps0, a_TA, Sm0, eta0);

        % Check physical validity
        if F_beam < 0 || sigma_MUC < 0
            results(i, :) = [a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps_val, NaN, NaN];
            skip_count = skip_count + 1;
        else
            % Dynamic simulation
            [t_sim, Am, p_contact, ~, ~, ~] = ...
                Membrane_Beam_Solver_MyImplementation2( ...
                    Wb0, Wm0, Wbt0, Wmt0, T, dt, Nx, Ny, L, b_MUC, ...
                    Theta_G, Theta_0, Theta_conv, rho_MUC, d_MUC, ...
                    rho_TA, A_TA, rho_LIG, A_LIG, sigma_MUC, ...
                    F_beam, M0_beam, Mu_c_beam, S_m, eta, ...
                    K_c, C_c, K_f, C_f, K_contact, Kr, Kr_r, ...
                    Ps_val, sr);

            % Extract F0 and SPL
            [Fp0, ~, ~, ~, ~, ~, ~, ~, ~, ~, ~, SPL_val, ~, ~] = ...
                Measures_Estimates(t_sim, factor, Am, p_contact, ...
                    Ps_val, sr, Nx, t1, t2);

            results(i, :) = [a_LCA, a_IA, a_PCA, a_CT, a_TA, ...
                             Ps_val, Fp0, SPL_val];
            valid_count = valid_count + 1;

            % Save waveform data if requested
            if SAVE_WAVEFORMS
                wf_file = sprintf('waveforms/sim_%05d.mat', i);
                wf_a_CT = a_CT; wf_a_TA = a_TA;
                wf_a_LCA = a_LCA; wf_Ps = Ps_val;
                save(wf_file, 'Am', 'p_contact', 't_sim', ...
                     'wf_a_CT', 'wf_a_TA', 'wf_a_LCA', 'wf_Ps', '-v7');
            end
        end

    catch ME
        results(i, :) = [a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps_val, NaN, NaN];
        error_count = error_count + 1;
        if error_count <= 10
            fprintf('  [ERROR] Sim %d: %s\n', i, ME.message);
        end
    end

    % Progress reporting and incremental save
    if mod(i, SAVE_EVERY) == 0
        elapsed = toc;
        rate = i / elapsed;
        remaining = (N_s - i) / rate;
        fprintf('  %4d/%d (%.0f%%) | valid=%d skip=%d err=%d | %.1f sim/min | ~%.0f min left\n', ...
            i, N_s, 100*i/N_s, valid_count, skip_count, error_count, ...
            rate*60, remaining/60);

        % Save intermediate CSV
        T_results = array2table(results(1:i, :), 'VariableNames', col_names);
        writetable(T_results, OUTPUT_CSV);
    end
end

%% ==================== FINAL OUTPUT ====================
elapsed = toc;

fprintf('\n================================================================\n');
fprintf('  COMPLETE\n');
fprintf('================================================================\n');
fprintf('  Total simulations: %d\n', N_s);
fprintf('  Valid (F0+SPL):    %d\n', valid_count);
fprintf('  Skipped (tension): %d\n', skip_count);
fprintf('  Errors:            %d\n', error_count);
fprintf('  Time:              %.1f minutes (%.1f sim/min)\n', ...
    elapsed/60, N_s/elapsed*60);

% Save full results (includes NaN rows)
T_results = array2table(results, 'VariableNames', col_names);
writetable(T_results, OUTPUT_CSV);
fprintf('\n  Full dataset:  %s (%d rows)\n', OUTPUT_CSV, N_s);

% Save clean version (NaN rows removed)
valid_mask = ~isnan(results(:, 7)) & ~isnan(results(:, 8));
T_clean = array2table(results(valid_mask, :), 'VariableNames', col_names);
clean_csv = strrep(OUTPUT_CSV, '.csv', '_clean.csv');
writetable(T_clean, clean_csv);
fprintf('  Clean dataset: %s (%d rows)\n', clean_csv, sum(valid_mask));

% Quick stats
if sum(valid_mask) > 0
    clean = results(valid_mask, :);
    fprintf('\n  Parameter ranges (valid sims):\n');
    fprintf('    a_CT:  [%.3f, %.3f]\n', min(clean(:,4)), max(clean(:,4)));
    fprintf('    a_TA:  [%.3f, %.3f]\n', min(clean(:,5)), max(clean(:,5)));
    fprintf('    a_LCA: [%.3f, %.3f]\n', min(clean(:,1)), max(clean(:,1)));
    fprintf('    Ps:    [%.0f, %.0f] Pa\n', min(clean(:,6)), max(clean(:,6)));
    fprintf('    F0:    [%.1f, %.1f] Hz\n', min(clean(:,7)), max(clean(:,7)));
    fprintf('    SPL:   [%.1f, %.1f] dB\n', min(clean(:,8)), max(clean(:,8)));
end

fprintf('\n================================================================\n');
fprintf('  Next step: copy dataset_BM_clean.csv to your FluidFlow\n');
fprintf('  project Beam_Membrane/ directory, then run the Python\n');
fprintf('  transfer learning scripts.\n');
fprintf('================================================================\n');
