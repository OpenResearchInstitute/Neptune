% This is the test bench for the FlexLink OFDM Modulator / Demodulator
% Set up the test bench
clc;
clear variables;
close all;

% ----------------------------------------
% Configuration
% ----------------------------------------
NumberOfdmSymbols = 2;
TxSampleRate      = 20e6;

% Se the signal to noise ratio. I include this to make the plots look more interesting
SNRdB             = 40;  

% Set the bandwidth
bLteBw            = true;
if(bLteBw == true); NumSubcarriers = 913;
else;               NumSubcarriers = 841;
end

% ----------------------------------------------------
% Create the Transmit Resource Grid (all QPSK)
% ----------------------------------------------------
I               = (.7071/0.5) * (randi([0 1],NumSubcarriers,NumberOfdmSymbols) - 0.5);
Q               = (.7071/0.5) * (randi([0 1],NumSubcarriers,NumberOfdmSymbols) - 0.5);
TxResourceGrid  = I + 1j*Q;

% ----------------------------------------------------
% Run the FlexLink Ofdm Moduator
% ----------------------------------------------------
TxOutputSequence = OfdmModulator(TxResourceGrid, TxSampleRate);


% ----------------------------------------------------
% Add some noise
% ----------------------------------------------------
SNR_Linear       = 10^(SNRdB/10);
SignalPower      = mean(TxOutputSequence.*conj(TxOutputSequence));
NoisePower       = SignalPower/SNR_Linear;

I_Noise          = 0.7071*sqrt(NoisePower)*randn(size(TxOutputSequence));
Q_Noise          = 0.7071*sqrt(NoisePower)*randn(size(TxOutputSequence));
Noise            = I_Noise + 1j*Q_Noise;
MeasNoisePower   = mean(Noise .* conj(Noise));
MeasSnrdB        = 10*log10(SignalPower/MeasNoisePower);
disp(['Measured SNR (dB): ', num2str(MeasSnrdB)]);

RxInputSequence = TxOutputSequence + Noise;
% RxInputSequence = TxOutputSequence;


% --------------------------------------------
% Decimate to 20MHz if we generated everything at 40MHz
% --------------------------------------------
if(TxSampleRate == 40e6)
    RxInputSequence = 2*RxInputSequence(1, 1:2:end);
end

% ---------------------------------------------------------
% Demodulate the received, noisy sequence
% ---------------------------------------------------------
% Processing preamble B points us to the start of the IFFT portion
% of the OFDM symbol. Here I set this instance manually.
StartSample        = 116;

RxResourceGrid = OfdmDemodulator(RxInputSequence, StartSample, bLteBw);

% The time has arrived to see whether all this actually works
Error       = TxResourceGrid - RxResourceGrid;
MaxAbsError = max(Error(:)); 

disp(['Maximum Abs Error Magnitude = ',  num2str(MaxAbsError)]);

figure(1)
subplot(1,2,1)
plot(real(TxResourceGrid(:)), imag(TxResourceGrid(:)), 'b.'); grid on;
xlabel('I')
ylabel('Q')
title('Constellation of TX Resource Elements');
subplot(1,2,2)
plot(real(RxResourceGrid(:)), imag(RxResourceGrid(:)), 'b.'); grid on;
xlabel('I')
ylabel('Q')
title('Constellation of RX Resource Elements');