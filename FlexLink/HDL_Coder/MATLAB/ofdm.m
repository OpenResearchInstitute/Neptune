%% Introduction
%MATLAB script from Contemporary Communications Systems Using MATLAB by
%John G. Proakis 
%transcribed from the book 25 September 2023
%parameterized for Neptune 25 September 2023

%% Script of Illustrative Example from Book
%In the example from the book, K = 10 and N = 20.
%A channel that has bandwidth B is subdivided in to K channels.
%N-point inverse DFT (IDFT) yields a real-valued sequence, where 1/square root of N
%is simply a scale factor. The sequence xn , 0 >= n <= N-1 corresponds
%to the samples of the multicarrier OFDM signat xt, consisting of K=N/2
%subscarriers. T is the signal duration, or signal interval.

%The modulator in an OFDM system can be implemented by computing the
%inverse DFT (7.6.4). The demodulator that recovers the information symbols
%from the received signal samples is implemented by computing the DFT
%(7.6.7). 

%Below, we are using the 16 point QAMsignal constellation shown in a
%previous figure, we select pseudorandomly each of the information symbols
%X1, X2, ... ,X9. With T=100 seconds, we generate the transmitted signal
%waveform and then plot it. We compute the IDFT values. We show that x(t),
%evaulatied at Tn/N, N = 0 to N-1, corresponds to the IDFT values. Finally,
%using the IDFT values, we computer the DFT. This means we've recovered the
%information symbols Xk, 1 <= k <= 9, from the samples of x(t),
%where t = n*T/N, 0 <= n <= N-1

clear all

%echo on turns on echoing for statements in all script files. When you turn on echoing, 
% MATLAB® displays each line in the file in the Command Window as it runs. Normally, 
% the statements in a script are not displayed during execution. Statement echoing is 
% useful for debugging and for demonstrations.
echo on

%number of information symbols. We have 9 information symbols and one zero
%valued symbol, at DC.
K = 10;

%we create N = 2K symbols by defining X(N-k) = X*k
%The sequence Xn, 0 <= n <= N-1, corresponds to the samples of the
%multicarrier OFDM signal x(t), consisting of K = N/2 subcarriers. 
%The information symbol X0 is set to zero. Since x(t) must be a real-valued
%signal, its N-point DFT (Xk) must satsify the symmetry property.
%Therefore, from the K information Symbols Kx we create N = 2K symbols.
N = 2*K;

%sample interval in seconds
%the symbol rate Ts = 1/T on each of the subchannels is equal to the frequency
%separation Bc of adjacent subcarriers. Bc = bandwidth/K. T = K*Ts. 
%The book clearly listed this as seconds, but the math doesn't check out.
T = 100;

%Testing T = 100 seconds. Tcheck should equal T. It doesn't. 
Ts = 1/T;
Tcheck = K*Ts;

%a is a 1 by 36 matrix of random numbers
a = rand(1,36);

%subtract 0.5 from all elements of the matrix a, then apply the signum function
%to the result. This "splits" the values into randomly distributed 1 and -1
a = sign(a-0.5);

%we create a 9 by 4 matrix out of a with the reshape function.
b = reshape(a,9,4);

%Generate the 16 QAM points
%take the columns of b and multiply them as follows. This makes one column
%of complex numbers. This is 9 by 1.
XXX = 2*b(:,1) + b(:,2) +j*(2*b(:,3) + b(:,4));

%Transpose XXX to convert it from column to row and store in XX. This is 1 by 9.
XX = XXX';

%modify X to put in a leading 0, the contents of XX, another zero, and then
%use the complex conjugate function conj() to get the complex conjugate of each element 
% of XX in reverse order, and append this result to X
%Doing this we get a 1 by 20 size matrix. It is double the number of
%symbols, with a leading zero and a zero in between the two halves. The
%center DC component is set to zero.
%Why are we doing this? Symmetry property.
%X is the set of complex frequencies at this point? That's what they have
%to be in order to take the IDFT and get a complex time series. 
X = [0 XX 0 conj(XX(9:-1:1))];

%set up xt to be a 1 by 101 matrix of zeros. 
xt = zeros(1,101);

%create multicarrier OFDM signal samples consisting of K = N/2 subcarriers
%(7.6.5)
%For each tick in time t, 0 to 100, calculate x(t)
%take the symbol k from X and multiply it by exp(j*2*pi*k*t/T) and sum them
%up. There are 0 to 19 symbols which is the entire symmetric set around
%zero of the information symbols. And a leading zero symbol.
%T = 100 seconds sure looks and feels like a misprint. 
%So, for 100 times in a row, we sum up the entire set of information
%symbols using exp(j*2*pi*k*t/T) so k goes 0 to 19 for each t/T

for t=0:100
    for k=0:N-1
        xt(1,t+1)=xt(1,t+1)+1/sqrt(N)*X(k+1)*exp(j*2*pi*k*t/T);
        echo off
    end
end


echo on

%set up xn to be a 1 by N matrix of zeros
xn = zeros(1,N);

%compute the N-point inverse DFT (IDFT) of X to get a real-valued sequence (7.6.4)
for n = 0:N-1
    for k = 0:N-1
        xn(n+1) = xn(n+1) + 1/sqrt(N)*X(k+1)*exp(j*2*pi*n*k/N);
        echo off
    end
end

%let's look at different x(k)
%we are wanting x(n) to be DC at x(0) - that's the lowest frequency
%x(n) runs from x(0) to x(N-1)
%N is the length of the symbol and will be 1024 for Neptune
%x(0) to x(N-1) then set first coefficient to 1 and the rest x(1) to x(N-1) to zero. DC
%coefficient is set to 1. When we do the IDFT, x(0) ends up 1/N which is a
%constant value. This is what we wanted to see for the DC term. In other words we 
%proved it was a constant.
%at k = 1, we have one sine wave, which is the lowest frequency. This is I
%think 1/T. The highest is at k = N-1, and this is fs/2. 
echo on

%show that xt corresponds to the IDFT values.

%using the IDFT values, compute the DFT using (7.6.7) and recover the
%information symbols

%pause %Press any key to see a plot of x(t)

%plot 0 to 100 in the x axis, and the absolute value of x(t) in the y axis.
% Y = abs( X ) returns the absolute value of each element in input X . 
% If X is complex, abs(X) returns the complex magnitude. 
figure
plot([0:100], abs(xt))

%check the difference between xn and samples of x(t)
for n = 0:N-1
    index_check = T/N*n+1
    d(n+1) = xt(T/N*n+1) - xn(1+n);
    echo off
end

echo on 

%The norm function calculates several different types of matrix norms: 
% e = norm(d) returns the largest singular value of d, which is the set of
% differences between xt and xn
e = norm(d)

%set up a 1 by 10 matrix of zeros (is this all we need?)
Y = zeros(1,10);

%Compute the DFT using (7.6.7) (we only had 9 values, remember? each of
%these values was calculated using all of the DFT points)
for k = 1:9
    for n = 0:N-1
        Y(1,k+1) = Y(1,k+1) + 1/sqrt(N)*xn(n+1)*exp(-j*2*pi*k*n/N);
        echo off
    end
end

echo on

%get the difference between the first ten elements of Y and X, then get the
%largest singular value of this list of differences. 
dd = Y(1:10) - X(1:10);
ee = norm(dd);

echo off

%% Script Adapted for Neptune Mandatory Specification

%see https://github.com/OpenResearchInstitute/Neptune/
%In the Neptune Mandatory specification, K = 901 and FFT size = 1024.
%Unlike the Proakis book, these aren't simple multiples of each other.
%FFT sizes are most frequently powers of two because the math works out
%better that way in terms of efficiency in the calculations. 
%The Neptune specification has optional configurations, but we're going to stick with
%the mandatory configuration for the first run through.  

%A channel that has bandwidth B is subdivided in to K channels.
%N-point DFT (IDFT) yields a real-valued sequence, where 1/square root of N
%is simply a scale factor. The sequence x of n , 0 >= n <= N-1 corresponds
%to the samples of the multicarrier OFDM signat x of t, consisting of K
%subscarriers. T is the signal duration, or signal interval, 

%Note:
%for HDL Coder, we're going to probably need to use the exp block in simulink.
% https://www.mathworks.com/help/simulink/slref/mathfunction.html

%Below, we are using the 16 point QAMsignal constellation shown in a
%previous figure, we select pseudorandomly each of the information symbols
%X1, X2, ... ,X9. With T=100 seconds, we generate the transmitted signal
%waveform and then plot it. We compute the IDFT values. We show that x(t),
%evaulatied at Tn/N, N = 0 ro N-1, corresponds to the IDFT values. Finally,
%using the IDFT values, we computer the DFT. This means we've recovered the
%information symbols Kx, 1 <= k <= 9, from the samples of x(t). 


%echo on turns on echoing for statements in all script files. When you turn on echoing, 
% MATLAB® displays each line in the file in the Command Window as it runs. Normally, 
% the statements in a script are not displayed during execution. Statement echoing is 
% useful for debugging and for demonstrations.
echo on

%set the mandatory bandwidth
BW = 20e6;
%BW = 40e6; %Trueflight

%Set the master clock frequency assumed in the specification
Fs = 20.48e6; %FlexLink
%Fs = 40e6; %Trueflight

%set the IFFT size (this is the number of points in the DFT (IDFT).
IFFTsize = 1024; %FlexLink
%IFFTsize = 2048; %Trueflight

%set mandatory subcarrier spacing
SS = Fs/IFFTsize;

%the specification lists 901 subcarriers. 
Kspec = 901;

%number of virtual carriers in the specification is calculated
Nvc = IFFTsize - Kspec;

%calculate the RF bandwidth from the subcarrier spacing times the number of
%channels give in the specification
%****This matches the number in the spec, which is 18 MHz
RFBW = Kspec*SS;

%sample interval (in seconds?) we need to make it 5x the value of IFFTsize 
%which is what the book example did 
%in order for the error checking at the end to work out. 
T = 5*IFFTsize; 

%instead of the 20 channels in the example in the book, we now have 901. 
%working backwards, and using a similar construction technique from the
%book, and using 16QAM, we are going to need to construct an n by 4 matrix.
%To make the 20 channels, we had a zero value, 9 random complex numbers,
%another zero value, and then the complex conjugates of those 9 random
%complex numbers in reverse order. I am not entirely clear yet on why we
%did that, since it doesn't seem to enter into the math. 

%a is a 1 by 4*n matrix of random numbers, where n = roughly half of K.
%A pattern of "0 n 0 n" does not sum up to an odd number. So there's no 
% solution, but we can add one more zero at the end like this "0 n 0 n 0"
% and it then evens out. Does this wreck any of the math? 

%Of the 901 information bearing subcarriers, 
% 451 are mapped into the positive frequency inputs at IFFT[0] through IFFT[450], 
% whereas the remaining 450 are mapped into the negative frequency inputs 
% at IFFT[574] through IFFT[1023]. 

%create the data that goes into the information bearing subcarriers
%we'll need 4x the number of information bearing subcarriers to start with
a = rand(1,Kspec*4);

%subtract 0.5 from all elements of the matrix a, then apply the signum function
%to the result. This "splits" the values into randomly distributed 1 and -1
a = sign(a-0.5);

%we create a Kspec by 4 matrix out of a with the reshape function.
b = reshape(a,Kspec,4);

%Generate the 16 QAM points
%take the columns of b and multiply them as follows. This makes one column
%of complex numbers that are 16QAM encoded
XXX = 2*b(:,1) + b(:,2) +j*(2*b(:,3) + b(:,4));

%Transpose this matrix to convert from column to row.
XX = XXX';

%modify X to put in a leading 0 for k = 0 (the DC component), 
% then the first 451 values of XX, insert the virtual carriers,
%and then the 450 remaining values of XX. 
%We now have something that can be processed with a 1024 point IDFT.
X = [0 XX(1:451) zeros(1,Nvc-1) XX(452:901)];

%Create a vector of tones for Simulink implementation of IFFT 
Xsim = X';
%convert to time series for IFFT input in Simulink implementation
Xsim = timeseries(Xsim, 1/Fs);

%set up xt to be a 1 by T+1 matrix of zeros
xt = zeros(1,T+1);

%create multicarrier OFDM signal samples   
%(7.6.5)

for t = 0:T
    for k = 0:IFFTsize-1
         xt(1, t+1) = xt(1, t+1) + 1/sqrt(IFFTsize)*X(k+1)*exp(j*2*pi*k*t/T);
        echo off
    end
end

echo on

%set up xn to be a 1 by N matrix of zeros
xn = zeros(1,IFFTsize);

%compute the N-point inverse DFT (IDFT) of X to get a real-valued sequence (7.6.4)
%for n = 0:IFFTsize-1 
for n = 1:IFFTsize-1
    for k = 0:IFFTsize-1
        xn(n+1) = xn(n+1) + 1/sqrt(IFFTsize)*X(k+1)*exp(j*2*pi*n*k/IFFTsize);
        echo off
    end
end


echo on

%show that xt corresponds to the IDFT values.

%using the IDFT values, compute the DFT using (7.6.7) and recover the
%information symbols

%pause %Press any key to see a plot of x(t)

%plot 0 to 100 in the x axis, and the absolute value of x(t) in the y axis.
% Y = abs( X ) returns the absolute value of each element in input X . 
% If X is complex, abs(X) returns the complex magnitude. 
figure('Name','Neptune x(t)');
plot([0:T], abs(xt))

%check the difference between xn and samples of x(t)
for n = 0:IFFTsize-1
    d(n+1) = xt(T/IFFTsize*n+1) - xn(1+n);
    echo off
end


echo on

%The norm function calculates several different types of matrix norms: 
% e = norm(d) returns the largest singular value of d, which is the set of
% differences between xt and xn
e = norm(d)

%set up a 1 by 10 matrix of zeros
Y = zeros(1,IFFTsize);

%Compute the DFT using (7.6.7)
for k = 1:IFFTsize-1
    for n = 0:IFFTsize-1
        Y(1,k+1) = Y(1,k+1) + 1/sqrt(IFFTsize)*xn(n+1)*exp(-j*2*pi*k*n/IFFTsize);
        echo off
    end
end

echo on

%get the difference between the elements of Y and X, then get the
%largest singular value of this list of differences.
dd = Y(1:IFFTsize) - X(1:IFFTsize);
ee = norm(dd)

%plot the error, leaving out k = 0.
figure('Name','Plot the Difference Between Input Tones and the DFT');
plot(2:IFFTsize, abs(dd(2:IFFTsize)), "LineWidth", 4)


%% AGC Burst Creation
% This is a Zadoff-Chu sequence, similar to the one in 3GPP LTE
% It has essentially zero autocorrelation off of the zero lag position.
% Commonly used for synchronization in cellular protocols. 

% Sequence length
Nzc = 887;

% Root index
u = 34;

% Zadoff Chu sequence function
AgcBurst1 = zadoffChuSeq(u,Nzc);

% Visualization
 figure('Name', 'Neptune Zadoff-Chu Sequence I vs Q');
 plot(AgcBurst1);
 figure('Name', 'First 100 values of Neptune Zadoff-Chu Sequence')
 plot([1:100],real(AgcBurst1(1:100)));

% We now take the Nzc = 887-point discrete Fourier transform.
% A second argument to fft specifies a number of points n for 
% the transform, representing DFT length.

AgcBurst2 = fft(AgcBurst1, 887);
AgcBurst2 = AgcBurst2/sqrt(887);

% By definition, we will have a certain number of positive 
% frequency subcarriers, m = 0, 1, ... ,ScPositive-1 and a certain 
% number of negative frequency subcarriers, 
% m = Nzc - ScNegative, ... , Nzc-1.

ScPositive = ceil(Nzc/2);
ScNegative = floor(Nzc/2);

% We set up an AGC IFFT input buffer of length 1024 

AGC_IFFT_input = zeros(IFFTsize,1);

% We map AgcBurst2 to the IFFT_input as so:
% Note that MATLAB uses 1-based indexing 
% and the specification uses 0-based indexing.
% We need to ensure that index zero (DC) is set to zero. 

AGC_IFFT_input(1:ScPositive) = AgcBurst2(1:ScPositive);
AGC_IFFT_input(IFFTsize - ScNegative:IFFTsize-1) = AgcBurst2(Nzc - ScNegative:Nzc-1);

% visualization
 figure('Name', 'Our IFFT Input');
 plot([1:IFFTsize],real(AGC_IFFT_input));

% Execute the IFFT and retain the first 102 samples.
% The AgcBurst shall occupy 5 microseconds, 
% which at 20.48MHz yields 102 samples.
% 
% we use the floor function to make this an integer

AgcBurst3 = ifft(AGC_IFFT_input,IFFTsize);
%AgcBurst3 = AgcBurst3*sqrt(IFFTsize);
AgcBurst3 = AgcBurst3*sqrt(Nzc);

% obtain the length of the AGC burst
AgcBurstLength = floor(5e-6*Fs);

% set up a zero'd out array of the calculated length
AgcBurst = zeros(AgcBurstLength,1);
% leaving the first value set to zero, copy over values from AgcBurst3
AgcBurst(2:AgcBurstLength) = AgcBurst3(2:AgcBurstLength);
% set the first value to zero
AgcBurst(1) = complex(0,0);

% visualization
figure('Name', 'Neptune AGC Burst')
plot([1:size(AgcBurst,1)], real(AgcBurst))

% convert to signed fixed point sfix16_en15
AgcBurst = fi(AgcBurst,1,16,15)

%% Preamble A Creation
% either 250 or 25 micro seconds depending on whether the
% transmitter wants the receiver to do frequency offset 
% acquisition and removal or not. 
% 22120448 Hz sample rate for transmitter

% equation
long_t_samples = floor(250e-06*Fs); % obtain number of samples
short_t_samples = floor(25e-06*Fs); % obtain number of samples
long_t = 0:1/Fs:(long_t_samples-1)/Fs; % set up an array samples long
short_t = 0:1/Fs:(short_t_samples-1)/Fs; % set up an array samples long

Tone_Frequency_1 = 4*160e3 % tone frequency 1
Tone_Frequency_2 = 12*160e3 % tone frequency 2

% real and imaginary components of the two tones for long Preamble
Tone1 = exp( 1*j*2*pi*Tone_Frequency_1*long_t);
Tone2 = exp( 1*j*2*pi*Tone_Frequency_2*long_t); 
Tone3 = exp(-1*j*2*pi*Tone_Frequency_1*long_t); 
Tone4 = exp(-1*j*2*pi*Tone_Frequency_2*long_t); 

% add components together and normalize
PreambleAlong = 0.25*(Tone1 + Tone2 + Tone3 + Tone4);
figure("Name", "Long Preamble A")
plot(long_t, PreambleAlong)

% we need the length to set up the counter to read from LUT
PreambleAlongLength = length(PreambleAlong);
% first value is punctured to zero
PreambleAlong(1) = complex(0,0);
% convert to fixed point
PreambleAlong = fi(PreambleAlong,1,16,15)

% real and imaginary components of the two tones for long Preamble
Tone1 = exp( 1*j*2*pi*Tone_Frequency_1*short_t);
Tone2 = exp( 1*j*2*pi*Tone_Frequency_2*short_t); 
Tone3 = exp(-1*j*2*pi*Tone_Frequency_1*short_t); 
Tone4 = exp(-1*j*2*pi*Tone_Frequency_2*short_t); 

% add components together and normalize
PreambleAshort = 0.25*(Tone1 + Tone2 + Tone3 + Tone4);
figure("Name", "Short Preamble A")
plot(short_t, PreambleAshort)

% we need the length to set up the counter to read from LUT
PreambleAshortLength = length(PreambleAshort);
% first value is punctured to zero
PreambleAshort(1) = complex(0,0);
% convert to fixed point
PreambleAshort = fi(PreambleAshort,1,16,15)




%% Preamble B Creation
% This is a Zadoff-Chu sequence, similar to the one in 3GPP LTE
% It has essentially zero autocorrelation off of the zero lag position.
% Commonly used for synchronization in cellular protocols. 

% Sequence length
NzcB = 331;

% Root index
u = 34;

% Zadoff Chu sequence function
PreambleB1 = zadoffChuSeq(u,NzcB);

% Visualization
 figure('Name', 'Neptune Zadoff-Chu Sequence I vs Q');
 plot(PreambleB1);
 figure('Name', 'First 100 values of Neptune Zadoff-Chu Sequence')
 plot([1:100],real(PreambleB1(1:100)));

% We now take the Nzc = 331-point discrete Fourier transform.
% A second argument to fft specifies a number of points n for 
% the transform, representing DFT length.

PreambleB2 = fft(PreambleB1, 331);
PreambleB2 = PreambleB2/sqrt(331);

% By definition, we will have a certain number of positive 
% frequency subcarriers, m = 0, 1, ... ,ScPositiveB-1 and a certain 
% number of negative frequency subcarriers, 
% m = NzcB - ScNegativeB, ... , NzcB-1.

ScPositiveB = ceil(NzcB/2);
ScNegativeB = floor(NzcB/2);

% We set up an AGC IFFT inputB buffer of length 1024 

AGC_IFFT_inputB = zeros(IFFTsize,1);

% We map PreambleB2 to the IFFT_input as so:
% Note that MATLAB uses 1-based indexing 
% and the specification uses 0-based indexing.
% We need to ensure that the first value (DC) is set to zero.
% We need to have a leading zero before that for quiescent case.


AGC_IFFT_inputB(2:ScPositiveB+1) = PreambleB2(1:ScPositiveB);
AGC_IFFT_inputB(IFFTsize+1 - ScNegativeB:IFFTsize) = PreambleB2(NzcB +1 - ScNegativeB:NzcB);

% visualization
 figure('Name', 'Our IFFT Input');
 plot([1:IFFTsize],real(AGC_IFFT_inputB));

% Execute the IFFT and retain all 1024 samples
% I think we mean to use 1106 action item to check this

PreambleB3 = ifft(AGC_IFFT_inputB,IFFTsize);
PreambleB3 = PreambleB3*sqrt(1024);

% get the size of what we've calculated
PreambleBLength = size(PreambleB3,1);

% we need the first value to be 0 in circuit
PreambleB = zeros(PreambleBLength,1);
PreambleB(2:PreambleBLength) = PreambleB3(2:PreambleBLength);
PreambleB(1) = complex(0,0); % puncture what would otherwise be lost.

% visualization
figure('Name', 'Neptune Preamble B')
plot([1:size(PreambleB,1)], real(PreambleB))


PreambleB = fi(PreambleB,1,16,15)



%% Create Push to Talk (PTT)

PTT_1 = repmat(0, 1024, 1);
PTT_2 = repmat(1, 1024*7, 1);
PTT_3 = repmat(0, 1024, 1);
PTT = cat(1, PTT_1, PTT_2, PTT_3);
PTT = logical(PTT);
%PTT = timetable(PTT,'SampleRate',Fs)
PTT = timeseries(PTT, 1/Fs)

%% Utility Space
% create a complex zero in signed fixed point sfix16_en15
a = complex(fi(0,1,16,15), fi(0,1,16,15));
%% Save Workspace


% Save entire workspace as ofdm_neptune_section_workspace.mat
save('ofdm_neptune_section_workspace.mat')


%% Load Neptune Workspace and open Simulink Models

% load the workspace created by the Neptune section of this script
load('ofdm_neptune_section_workspace.mat')

% open the simulink model under development
open_system("neptune_IDFT_HDL_Coder_input")

