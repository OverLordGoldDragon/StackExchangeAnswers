% Answer to https://dsp.stackexchange.com/q/86181/50076
%% Configure ################################################################
fs = 400;             % (in Hz)  anything works
duration = 5;         % (in sec) anything works
padtype = 'reflect';  % anything (supported) works

% get power between these frequencies
freq_min = 50;   % (in Hz)
freq_max = 150;  % (in Hz)

%% Obtain transform #########################################################
% make signal & filterbank
% assume this is Amperes passing through 1 Ohm resistor; P = I^2*R, E = P * T
% check actual physical units for your specific application and adjust accordingly
x = randn(1, fs * duration);
fb = cwtfilterbank('Wavelet', 'amor', 'SignalLength', N, 'VoicesPerOctave', 11, ...
                   'SamplingFrequency', fs, 'Boundary', padtype);

% transform, get frequencies
[Wx, freqs] = cwt(x);

% fetch coefficients according to `freq_min_max`
Wx_spec = Wx((freq_min < freqs) + (freqs < freq_max), :);

%% "Adjustments" ############################################################
% See "Practical adjustments" in the answer
% We shouldn't have to do this - the wavelets should be normalized such that
% it's automatically accounted for.

% fetch wavelets in freq domain, compute ET & ES transfer funcs, fethch maxima
psi_fs = fb.wavelets;  % fetch wavelets in freq domain
ET_tfn = sum(abs(psi_fs).^2, 1);
ES_tfn = abs(sum(psi_fs, 1)).^2;
ET_adj = max(ET_tfn);
ES_adj = max(ES_tfn);

%% Compute energy & power ###################################################
% compute energy & power (discrete)
ET_disc = sum(abs(Wx_spec).^2, 'all') / ET_adj;
ES_disc = sum(abs(real(sum(Wx_spec, 1))).^2, 'all') / ES_adj;
PT_disc = ET_disc / length(x);
PS_disc = ES_disc / length(x);

% compute energy & power(physical); estimate underlying continuous waveform via
% Riemann integration
sampling_period = 1 / fs;
ET_phys = ET_disc * sampling_period * duration;
ES_phys = ES_disc * sampling_period * duration;
PT_phys = ET_phys / duration;
PS_phys = ES_phys / duration;

% repeat for original signal
Ex_disc = sum(abs(x).^2, 'all');
Px_disc = Ex_disc / length(x);
Ex_phys = Ex_disc * sampling_period * duration;
Px_phys = Ex_phys / duration;

%% Report ###################################################################
s = ['Between %d and %d Hz, DISCRETE:\n'...
     '%.6g -- energy of transform\n'...
     '%.6g -- energy of signal\n'...
     '%.6g -- power of transform\n'...
     '%.6g -- power of signal\n\n'];
fprintf(s, freq_min, freq_max, ET_disc, ES_disc, PT_disc, PS_disc)


s = ['Between %d and %d Hz, PHYSICAL (via Riemann integration):\n'...
     '%.6g Joules -- energy of transform\n'...
     '%.6g Joules -- energy of signal\n'...
     '%.6g Watts  -- power of transform\n'...
     '%.6g Watts  -- power of signal\n\n'];
fprintf(s, freq_min, freq_max, ET_phys, ES_phys, PT_phys, PS_phys)

s = ['Original signal:\n'...
     '%.6g -- energy (discrete)\n'...
     '%.6g -- power  (discrete)\n'...
     '%.6g Joules -- energy (physical)\n'...
     '%.6g Watts  -- power  (physical)\n\n'];
fprintf(s, Ex_disc, Px_disc, Ex_phys, Px_phys)
