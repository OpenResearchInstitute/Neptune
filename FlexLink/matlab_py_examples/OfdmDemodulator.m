function ResourceGrid = OfdmDemodulator(InputSequence, StartSample, bLteBw)

% Note that the sample rate is 20MHz (there is no choice as there was in the modulator)
% Therefore,
SampleRate        = 20e6;   % 20MHz
FFT_Size          = 1024;   % IFFT length (1024/20MHz) = 51.2 microseconds 
CP_Length         = 116;    % CP length (116/20MHz)    =  5.8 microseconds
OfdmSymbolLength  = CP_Length + FFT_Size;
SubcarrierSpacing = SampleRate / FFT_Size; 

% An OFDM symbol is the concatenation of the CP and the IFFT portion
% The StartSample input argument is the estimated position of the first sample
% of the Ifft portion of the first valid OFDM symbol in the FlexLink Packet.
% The start time is found via the correlation against PreambleB.
% We will start 1 microsecond inside the CP to avoid intersymbol 
% interference due to channel precusors.
OneMicrosecondsInSamples = 20;
assert(StartSample > OneMicrosecondsInSamples, "Improper input sequence.");

% Determine the number of available OFDM symbols we can demodulator
NumAvailableOfdmSymbols = floor( (length(InputSequence))/OfdmSymbolLength);

if(bLteBw == true)          % LTE BW
    NumSubcarriers         = 913;   
    PosSubcarrierIndices   = 456:912;    % Indices in resource grid
    NegSubcarrierIndices   = 0:455;      % Indices in resoruce grid
    PosFftIndices          = 0:456;      % Indices in FFT input buffer
    NegFftIndices          = 568:1023;   % Indices in FFT input buffer
else                        % WLAN BW
    NumSubcarriers         = 841; 
    PosSubcarrierIndices   = 420:840;    % Indices in resource grid
    NegSubcarrierIndices   = 0:419;      % Indices in resoruce grid
    PosFftIndices          = 0:420;      % Indices in FFT input buffer
    NegFftIndices          = 604:1023;   % Indices in FFT input buffer
end      

% Phase compensation for 1 microsecond advance
TonesIEEE      = -(NumSubcarriers - 1)/2 : 1 : (NumSubcarriers - 1)/2;
OneMicroSecond = OneMicrosecondsInSamples / SampleRate;   % = 1e-6
Compensation   = exp(1j*2*pi*OneMicroSecond*TonesIEEE*SubcarrierSpacing).'; 

% Start the OFDM Demodulation Process (l is the 0 based OFDM symbol index)
ResourceGrid   = zeros(NumSubcarriers, NumAvailableOfdmSymbols);
StartIndex     = StartSample - OneMicrosecondsInSamples;
Range          = StartIndex : StartIndex + FFT_Size - 1;
for l = 0:NumAvailableOfdmSymbols - 1
    FftInputBuffer   = InputSequence(1, Range + 1).';
    FftOutputBuffer  = fft(FftInputBuffer);

    % Place the compensated FftOutput into the resource grid
    ResourceGrid(PosSubcarrierIndices + 1, l + 1) = FftOutputBuffer(PosFftIndices + 1, 1);
    ResourceGrid(NegSubcarrierIndices + 1, l + 1) = FftOutputBuffer(NegFftIndices + 1, 1);
    ResourceGrid(:, l+1)                          = ResourceGrid(:, l+1) .* Compensation;

    Range                                         = Range + OfdmSymbolLength;
end
