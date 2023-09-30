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

pause %Press any key to see a plot of x(t)

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
B = 20e6;

%Set the master clock frequency assumed in the specification
Fs = 20.48e6;

%set the IFFT size (this is the number of points in the DFT (IDFT).
% This value was N in the book. We aren't really going to use this? It's
% 1024 before calculating a 1024 FFT is easier than calcuating a 901 FFT? 
IFFTsize = 1024;

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

pause %Press any key to see a plot of x(t)

%plot 0 to 100 in the x axis, and the absolute value of x(t) in the y axis.
% Y = abs( X ) returns the absolute value of each element in input X . 
% If X is complex, abs(X) returns the complex magnitude. 
figure
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

%get the difference between the first ten elements of Y and X, then get the
%largest singular value of this list of differences. (10?)
dd = Y(1:IFFTsize) - X(1:IFFTsize);
ee = norm(dd)

%plot the error, leaving out k = 0. 
plot(2:1024, abs(dd(2:1024)), "LineWidth", 4)