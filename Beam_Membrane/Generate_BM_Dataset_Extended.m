%% Generate_BM_Dataset_Extended.m
% Regenerates the BM dataset with THREE extra physiological output columns
% (ACFL, PC, CPP) while staying ROW-ALIGNED to the existing dataset_BM.csv.
%
% The original dataset used unseeded rand(), so we cannot reproduce the same
% 5000 points by re-randomizing. Instead we READ dataset_BM.csv and loop over
% its EXACT parameter rows (a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps), so the new
% outputs line up 1:1 with all existing transfer/TabPFN results.
%
% Usage (run from the Beam+Membrane-Sean MATLAB folder, where the solver +
% PhonationModelsCode2 live; copy this file, Measures_Estimates_Ext.m and
% CPP_Estimate.m into that folder, and place dataset_BM.csv alongside):
%   >> Generate_BM_Dataset_Extended
%
% Output: dataset_BM_extended.csv with columns
%   a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps, F0, SPL, ACFL, PC, CPP, PC_loc
% (PC_loc is the optional collision-pressure location; drop it if unwanted.)

clc; clear all; close all;

%% ==================== CONFIGURATION ====================
INPUT_CSV  = 'dataset_BM.csv';            % existing params + F0/SPL
OUTPUT_CSV = 'dataset_BM_extended.csv';
SAVE_EVERY = 50;
F0_TOL = 1.0;     % Hz   sanity tolerance vs existing F0
SPL_TOL = 0.5;    % dB   sanity tolerance vs existing SPL

%% ==================== SETUP (identical to Generate_BM_Dataset.m) ========
addpath(genpath('PhonationModelsCode2'));

T = 1; c = 350;
delta = (10^-2) * 0.25;
factor = 4;
dt = (delta/c) / factor;
fs = floor(1/dt);
t1 = 0.5; t2 = 1;

Stiffness_Damping_Membrane_Beam       % loads stiffness/damping/material/geometry

sr = 1.2;
Nx = 20; Ny = 10;
Theta_conv = 0.05;

Wb0 = zeros(Nx, 1);
Wbt0 = zeros(Nx, 1);
[Y, X] = meshgrid(linspace(0,1,Ny), linspace(0,1,Nx));
Wm0 = 1e-9 * sin(pi*X) .* sin(pi*Y);
Wmt0 = zeros(Nx, Ny);

%% ==================== READ EXISTING ROWS ====================
T_in = readtable(INPUT_CSV);
N = height(T_in);
fprintf('Loaded %s : %d rows (looping over EXACT existing params)\n', INPUT_CSV, N);

% has F0/SPL to compare against?
have_ref = ismember('F0', T_in.Properties.VariableNames) && ...
           ismember('SPL', T_in.Properties.VariableNames);

col_names = {'a_LCA','a_IA','a_PCA','a_CT','a_TA','Ps','F0','SPL','ACFL','PC','CPP','PC_loc'};
results = NaN(N, numel(col_names));

%% ==================== MAIN LOOP ====================
tic;
valid_count = 0; skip_count = 0; error_count = 0;
max_f0_diff = 0; max_spl_diff = 0;

for i = 1:N
    a_LCA = T_in.a_LCA(i);
    a_IA  = T_in.a_IA(i);
    a_PCA = T_in.a_PCA(i);
    a_CT  = T_in.a_CT(i);
    a_TA  = T_in.a_TA(i);
    Ps_val = T_in.Ps(i);
    ActVec = [a_LCA, a_IA, a_PCA, a_CT, a_TA];

    base = [a_LCA, a_IA, a_PCA, a_CT, a_TA, Ps_val];

    try
        [eps0, Theta_G] = Posturing_Simulation2(ActVec);
        Theta_G = 0.5 * Theta_G;
        L = (1 + eps0) * L0;

        [Mu_c_beam, M0_beam, F_beam, d_MUC, d_LIG, d_TA, ...
         b_MUC, b_LIG, b_TA, A_MUC, A_LIG, A_TA, sigma_MUC, ...
         rho_MUC, rho_LIG, rho_TA, Sm, eta] = ...
            Membrane_Beam_Parameters4(b0, A_MUC_0, A_LIG_0, A_TA_0, ...
                rho_MUC_0, rho_LIG_0, rho_TA_0, eps0, a_TA, Sm0, eta0);

        if F_beam < 0 || sigma_MUC < 0
            results(i, :) = [base, NaN, NaN, NaN, NaN, NaN, NaN];
            skip_count = skip_count + 1;
        else
            [t_sim, Am, p_contact, ~, ~, ~] = ...
                Membrane_Beam_Solver_MyImplementation2( ...
                    Wb0, Wm0, Wbt0, Wmt0, T, dt, Nx, Ny, L, b_MUC, ...
                    Theta_G, Theta_0, Theta_conv, rho_MUC, d_MUC, ...
                    rho_TA, A_TA, rho_LIG, A_LIG, sigma_MUC, ...
                    F_beam, M0_beam, Mu_c_beam, S_m, eta, ...
                    K_c, C_c, K_f, C_f, K_contact, Kr, Kr_r, ...
                    Ps_val, sr);

            % NOTE: extended measures -> F0, SPL, PC(#10), PC_loc(#11), ACFL(#15), CPP(#16)
            [Fp0,~,~,~,~,~,~,~,~,PC,PC_loc,SPL_val,~,~,ACFL,CPP] = ...
                Measures_Estimates_Ext(t_sim, factor, Am, p_contact, ...
                    Ps_val, sr, Nx, t1, t2);

            results(i, :) = [base, Fp0, SPL_val, ACFL, PC, CPP, PC_loc];
            valid_count = valid_count + 1;

            % sanity: does recomputed F0/SPL match the stored values?
            if have_ref && ~isnan(T_in.F0(i)) && ~isnan(T_in.SPL(i))
                max_f0_diff  = max(max_f0_diff,  abs(Fp0 - T_in.F0(i)));
                max_spl_diff = max(max_spl_diff, abs(SPL_val - T_in.SPL(i)));
            end
        end

    catch ME
        results(i, :) = [base, NaN, NaN, NaN, NaN, NaN, NaN];
        error_count = error_count + 1;
        if error_count <= 10
            fprintf('  [ERROR] row %d: %s\n', i, ME.message);
        end
    end

    if mod(i, SAVE_EVERY) == 0
        elapsed = toc; rate = i/elapsed;
        fprintf('  %4d/%d (%.0f%%) | valid=%d skip=%d err=%d | %.1f/min | dF0max=%.3f dSPLmax=%.3f\n', ...
            i, N, 100*i/N, valid_count, skip_count, error_count, rate*60, ...
            max_f0_diff, max_spl_diff);
        writetable(array2table(results(1:i,:), 'VariableNames', col_names), OUTPUT_CSV);
    end
end

%% ==================== OUTPUT + SANITY ====================
writetable(array2table(results, 'VariableNames', col_names), OUTPUT_CSV);

fprintf('\n================ DONE ================\n');
fprintf('  rows written:      %d  (input had %d)\n', size(results,1), N);
fprintf('  valid / skip / err: %d / %d / %d\n', valid_count, skip_count, error_count);

if have_ref
    fprintf('\n  REPRODUCIBILITY CHECK vs existing F0/SPL:\n');
    fprintf('    max |dF0|  = %.4f Hz  (tol %.2f) -> %s\n', max_f0_diff, F0_TOL, ...
        ternary(max_f0_diff<=F0_TOL,'PASS','*** FAIL: rows not aligned / pipeline differs ***'));
    fprintf('    max |dSPL| = %.4f dB  (tol %.2f) -> %s\n', max_spl_diff, SPL_TOL, ...
        ternary(max_spl_diff<=SPL_TOL,'PASS','*** FAIL ***'));
end

vm = ~isnan(results(:,9)) & ~isnan(results(:,10));
if any(vm)
    fprintf('\n  New-output ranges (valid rows):\n');
    fprintf('    ACFL: [%.3f, %.3f]   (BCM source: [30, 1190])\n', min(results(vm,9)),  max(results(vm,9)));
    fprintf('    PC  : [%.3f, %.3f]   (BCM source: [0, 4000])\n',   min(results(vm,10)), max(results(vm,10)));
    cppv = results(vm,11); cppv = cppv(~isnan(cppv));
    if ~isempty(cppv)
        fprintf('    CPP : [%.3f, %.3f]   (BCM source: [-12, 46])\n', min(cppv), max(cppv));
    end
    fprintf('\n  Physicality: PC>0 ? %s   ACFL>0 ? %s\n', ...
        ternary(all(results(vm,10)>=0),'yes','NO'), ternary(all(results(vm,9)>=0),'yes','NO'));
    fprintf('\n  >> If ACFL/PC/CPP ranges differ a lot from the BCM source ranges,\n');
    fprintf('     the DEFINITIONS/UNITS differ -- reconcile before using for transfer.\n');
end

fprintf('\n  Wrote: %s\n', OUTPUT_CSV);

function out = ternary(cond, a, b)
    if cond, out = a; else, out = b; end
end
