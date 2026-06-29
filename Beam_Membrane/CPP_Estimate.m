function CPP = CPP_Estimate(x, fs, F0)
% CPP_Estimate  Cepstral Peak Prominence (dB) of a steady-state pressure signal.
%
%   CPP = CPP_Estimate(x, fs, F0)
%     x  : steady-state radiated-pressure waveform (pout over [t1,t2])
%     fs : sample frequency [Hz]
%     F0 : fundamental frequency estimate [Hz] (used to center the search band)
%
% Method (Hillenbrand-style power cepstrum):
%   cepstrum = real(ifft(log(|fft(x)|.^2)))
%   find the peak in the quefrency band around 1/F0, fit a linear regression
%   baseline across that band, and report (peak - baseline) on a dB scale.
%
% IMPORTANT consistency note: CPP is highly definition-dependent (log base,
% power vs magnitude, dB scaling, quefrency band, baseline fit). The BCM
% source (Parra/JASA) CPP is in roughly [-12, 46] dB. If this routine's scale
% differs, MATCH the BCM source's exact CPP algorithm before using CPP for
% BCM->BM transfer -- otherwise source and target CPP are not the same quantity.

    x = x(:);
    x = x - mean(x);
    N = numel(x);
    if N < 32 || ~isfinite(F0) || F0 <= 0 || all(x == 0)
        CPP = NaN;
        return;
    end

    % power cepstrum
    X = fft(x);
    logspec = log(abs(X).^2 + 1e-20);
    ceps = real(ifft(logspec));
    ceps_db = 10*log10(ceps.^2 + 1e-20);     % put cepstrum on a dB scale

    q = (0:N-1)' / fs;                        % quefrency (s)

    % search band: voiced F0 range ~[60, 500] Hz  ->  quefrency [1/500, 1/60] s
    qlo = 1/500; qhi = 1/60;
    band = (q >= qlo) & (q <= qhi);
    if ~any(band)
        CPP = NaN; return;
    end
    cb = ceps_db(band);
    qb = q(band);

    % cepstral peak near 1/F0
    [pk, idx] = max(cb);

    % linear-regression baseline across the band; CPP = peak above baseline
    p = polyfit(qb, cb, 1);
    baseline = polyval(p, qb(idx));
    CPP = pk - baseline;
end
