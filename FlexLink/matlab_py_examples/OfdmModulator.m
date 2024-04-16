function OutputSequence = OfdmModulator(ResourceGrid, OutputSampleRate)

% Determine the dimensions of the resource grid
[NumSubcarriers, NumOfdmSymbols] = size(ResourceGrid);

% Error checking (NumSubcarriers = 913 or 841 for the LTE or WLAN bandwidth)
assert(NumSubcarriers == 913    || NumSubcarriers == 841,    "Invalid number of subcarriers.");
assert(OutputSampleRate == 20e6 || OutputSampleRate == 40e6, "Invalid output sample rate.");

% Pick the IFFT size
if(OutputSampleRate == 20e6) 
    IFFT_Size = 1024;   % IFFT length (1024/20MHz) = 51.2 microseconds 
    CP_Length = 116;    % CP length   (116/20MHz)  =  5.8 microseconds
else
    IFFT_Size = 2048;   % IFFT length (2048/40MHz) = 51.2 microseconds 
    CP_Length = 232;    % CP length   (232/40MHz)  =  5.8 microseconds
end

OfdmSymbolLength = CP_Length + IFFT_Size; 

% Define the positive and negative frequency subcarriers in the resource grid
% k = 420 and 456 are the center DC carrier and thus belong to the positive frequencies
bIsLteBandwidth = NumSubcarriers == 913;
if(bIsLteBandwidth)
    PosSubcarrierIndices = 456:912; NegSubcarrierIndices = 0:455;
else
    PosSubcarrierIndices = 420:840; NegSubcarrierIndices = 0:419;
end

% Define the positive and negative frequency indices in the IFFT input buffer
if(bIsLteBandwidth)
    PosIfftIndices    = 0:456;
    if(IFFT_Size == 1024); NegIfftIndices = 568:1023;
    else;                  NegIfftIndices = 1592:2047; end
else
    PosIfftIndices    = 0:420;
    if(IFFT_Size == 1024); NegIfftIndices =  604:1023; 
    else;                  NegIfftIndices =  1628:2047; end
end

% Start the OFDM Modulation Process (l is the 0 based OFDM symbol index)
OutputSequence      = zeros(1, NumOfdmSymbols * OfdmSymbolLength);
IfftInputBuffer     = zeros(IFFT_Size, 1);
OutputSequenceIndex = 1; % Due to MatLab indexing (otherwise = 0)
for l = 0:NumOfdmSymbols - 1
    OfdmSymbol   = zeros(1, OfdmSymbolLength);
    IfftInputBuffer(PosIfftIndices + 1, 1) = ResourceGrid(PosSubcarrierIndices + 1, l+1);
    IfftInputBuffer(NegIfftIndices + 1, 1) = ResourceGrid(NegSubcarrierIndices + 1, l+1);
    IfftOutputBuffer                       = ifft(IfftInputBuffer);
    % Fretch the Cyclic Prefix
    CyclicPrefix     = IfftOutputBuffer(IFFT_Size - CP_Length + 1:end, 1);
    % Place the cyclic prefix at the start of the OFDM symbol
    OfdmSymbol(1, 1:CP_Length)   = CyclicPrefix.';
    % Place the IFFT portion next to the cyclic prefix
    OfdmSymbol(1, CP_Length+1:end) = IfftOutputBuffer.';
    % Place the OfdmSymbol into the output buffer
    OutputSequence(1,OutputSequenceIndex:OutputSequenceIndex + OfdmSymbolLength - 1) = OfdmSymbol;
    % Update the index into the output sequence
    OutputSequenceIndex = OutputSequenceIndex + OfdmSymbolLength;
end
