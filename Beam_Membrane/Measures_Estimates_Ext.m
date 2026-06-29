function [Fp0,xp,P,xv,V,Amin,Amax,Aavg,indGC,p_contact_max,p_contact_max_location,SPL,score_sf,score_p,ACFL,CPP] = Measures_Estimates_Ext(t,factor,Am,p_contact, Ps,sr,Nx,t1,t2)
% Measures_Estimates_Ext
% Drop-in extension of Measures_Estimates that ALSO returns:
%   ACFL : AC glottal flow  = peak-to-peak of the glottal volume-flow
%          waveform Ug over the steady-state window [t1,t2]
%   CPP  : cepstral peak prominence from the radiated pressure pout
% PC (peak collision pressure) is already returned as p_contact_max (#10);
% its location is p_contact_max_location (#11).
%
% Everything up to and including SPL is IDENTICAL to the original
% Measures_Estimates.m. The only additions are fs, ACFL and CPP at the end.

% finding indices corresponding to the steady-state bounds t1,t2
[indmin,indmax]=FindingIndices(t,t1,t2);

[f, P1] = OneSidedSpectrum(t(indmin:indmax), Am(indmin:indmax));
PSD=P1.^2;
Spos = max(PSD(:), 1e-20);
SF = exp(mean(log(Spos))) / mean(Spos);
score_sf=1-SF;

[~,indF0]=max(P1);
F0=f(indF0);
Fp0=F0;

if F0>0
    [P,xp]=findpeaks(Am(indmin:indmax),t(indmin:indmax),'MinPeakDistance',0.8/F0);
    [V,xv]=findpeaks(-Am(indmin:indmax),t(indmin:indmax),'MinPeakDistance',0.8/F0);
    PP=P(2:end);
    dxp=diff(xp);
    X=[PP;dxp];
    mu = mean(X, 2);
    diffs = X - mu;
    s = sqrt(mean(sum(diffs.^2, 1)));
    score_p=exp(-1000*s);
else
    xp=[]; P=[]; xv=[]; V=[]; score_p=NaN;
end

[Amin, indLocal] = min(Am(indmin:indmax));
Amax = max(Am(indmin:indmax));
Aavg = mean(Am(indmin:indmax));
indGC = indLocal + indmin - 1;

A=p_contact(indmin:indmax,:,:);
[p_contact_max, linIdx] = max(A(:));
[~, ix,~] = ind2sub(size(A), linIdx);
p_contact_max_location=(ix-1)/(Nx-1);

pout = WRA_Solver(t,factor,Am,Ps,sr);
p0 = 20e-6;
prms = rms(pout(indmin:indmax));
SPL = 20*log10(prms/p0);

% ==================== NEW OUTPUTS ====================
fs = 1/(t(2)-t(1));                 % sample frequency from the time vector

% --- ACFL: AC (peak-to-peak) glottal volume flow over steady state ---
% Glottal volume flow Ug is the Bernoulli source flow that drives WRA_Solver
% (see WRA_Solver.m lines 16-17): q = sqrt(2*Ps/rho) .* (SR .* Am).
rho_air = 1.14;                     % kg/m^3, matches WRA_Solver
ACFL_UNIT = 1e6;                    % m^3/s -> cm^3/s (mL/s) to match BCM source units.
                                    % Am is in m^2 -> Ug in m^3/s; BCM ACFL is in cm^3/s.
                                    % Verify the printed ACFL range lands near BCM [30,1190].
Ug = sqrt(2*Ps/rho_air) .* (sr .* Am(:));
Ug_ss = Ug(indmin:indmax);
ACFL = (max(Ug_ss) - min(Ug_ss)) * ACFL_UNIT;

% --- CPP: cepstral peak prominence from radiated pressure ---
CPP = CPP_Estimate(pout(indmin:indmax), fs, F0);

end
