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
%evaulatied at Tn/N, N = 0 ro N-1, corresponds to the IDFT values. Finally,
%using the IDFT values, we computer the DFT. This means we've recovered the
%information symbols Kx, 1 <= k <= 9, from the samples of x(t). 

%echo on turns on echoing for statements in all script files. When you turn on echoing, 
% MATLAB® displays each line in the file in the Command Window as it runs. Normally, 
% the statements in a script are not displayed during execution. Statement echoing is 
% useful for debugging and for demonstrations.
echo on

%number of channels
K = 10;

%number of points in the DFT (IDFT)
N = 2*K;

%sample interval (in seconds?)
T = 100;

%a is a 1 by 36 matrix of random numbers
a = rand(1,36);

%subtract 0.5 from all elements of the matrix a, then apply the signum function
%to the result. This "splits" the values into randomly distributed 1 and -1
a = sign(a-0.5);

%we create a 9 by 4 matrix out of a with the reshape function.
b = reshape(a,9,4);

%Generate the 16 QAM points
%take the columns of b and multiply them as follows. This makes one column
%of complex numbers. 
XXX = 2*b(:,1) + b(:,2) +j*(2*b(:,3) + b(:,4));

%Transpose this matrix to convert from column to row.
XX = XXX';

%modify X to put in a leading 0, the contents of XX, another zero, and then
%use the complex conjugate function conj() to get the complex conjugate of each element 
% of XX in reverse order, and append this result to X
%Doing this we get a 1 by 20 size matrix. It is double the number of channels.
X = [0 XX 0 conj(XX(9:-1:1))];

%set up xt to be a 1 by 101 matrix of zeros
xt = zeros(1,101);

%create multicarrier OFDM signal samples consisting of K = N/2 subcarriers
%(7.6.5)
for t = 0:100
    for k = 0:N-1 
        xt(1, t+1) = xt(1, t+1) + 1/sqrt(N)*X(k+1)*exp(j*2*pi*k*t/T);
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


echo on

%show that xt corresponds to the IDFT values.

%using the IDFT values, compute the DFT using (7.6.7) and recover the
%information symbols

pause %Press any key to see a plot of x(t)

%plot 0 to 100 in the x axis, and the absolute value of x(t) in the y axis.
% Y = abs( X ) returns the absolute value of each element in input X . 
% If X is complex, abs(X) returns the complex magnitude. 
plot([0:100], abs(xt))

%check the difference between xn and samples of x(t)
for n = 0:N-1
    d(n+1) = xt(T/N*n+1) - xn(1+n);
    echo off
end

echo on

%The norm function calculates several different types of matrix norms: 
% e = norm(d) returns the largest singular value of d, which is the set of
% differences between xt and xn
e = norm(d)

%set up a 1 by 10 matrix of zeros
Y = zeros(1,10);

%Compute the DFT using (7.6.7)
for k = 1:9
    for n = 0:N-1
        Y(1,k+1) = Y(1,k+1) + 1/sqrt(N)*xn(n+1)*exp(-j*2*pi*k*n/N);
        echo off
    end
end

echo on

%get the difference between the first ten elements of Y and X, then get the
%largest singular value of this list of differences. 
dd = Y(1:10) - X(1:10)
ee = norm(dd)


%% Script Adapted for Neptune Mandatory Specification

% this is not working  yet

%see https://github.com/OpenResearchInstitute/Neptune/
%In the Neptune Mandatory specification, K = 901 and FFT size = 1024.
%Unlike the Proakis book, these aren't simple multiples of each other. 
%The specification has optional configurations, but we're going to stick with
%the mandatory configuration for the first run through.  

%A channel that has bandwidth B is subdivided in to K channels.
%N-point DFT (IDFT) yields a real-valued sequence, where 1/square root of N
%is simply a scale factor. The sequence x of n , 0 >= n <= N-1 corresponds
%to the samples of the multicarrier OFDM signat x of t, consisting of K=N/2
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

clear all

%echo on turns on echoing for statements in all script files. When you turn on echoing, 
% MATLAB® displays each line in the file in the Command Window as it runs. Normally, 
% the statements in a script are not displayed during execution. Statement echoing is 
% useful for debugging and for demonstrations.
echo on

%set the mandatory bandwidth
B = 20e6;

%Set the master clock frequency assumed in the specification
Fs = 20.48e6

%set the IFFT size
IFFTsize = 1024;

%set mandatory subcarrier spacing
SS = Fs/IFFTsize

%number of subcarriers (channels)
K = B/SS

%however, the specification lists 901 subcarriers (need to figure this out)
Kspec = 901

%number of points in the DFT (IDFT)
N = 2*K;

%sample interval (in seconds?)
T = 100;

%a is a 1 by 36 matrix of random numbers
a = rand(1,36);

%subtract 0.5 from all elements of the matrix a, then apply the signum function
%to the result. This "splits" the values into randomly distributed 1 and -1
a = sign(a-0.5);

%we create a 9 by 4 matrix out of a with the reshape function.
b = reshape(a,9,4);

%Generate the 16 QAM points
%take the columns of b and multiply them as follows. This makes one column
%of complex numbers. 
XXX = 2*b(:,1) + b(:,2) +j*(2*b(:,3) + b(:,4));

%Transpose this matrix to convert from column to row.
XX = XXX';

%modify X to put in a leading 0, the contents of XX, another zero, and then
%use the complex conjugate function conj() to get the complex conjugate of each element 
% of XX in reverse order, and append this result to X
X = [0 XX 0 conj(XX(9:-1:1))];

%set up xt to be a 1 by 101 matrix of zeros
xt = zeros(1,101);

%create multicarrier OFDM signal samples consisting of K = N/2 subcarriers
%(7.6.5)
for t = 0:100
    for k = 0:N-1
        xt(1, t+1) = xt(1, t+1) + 1/sqrt(N)*X(k+1)*exp(j*2*pi*k*t/T)
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


echo on

%show that xt corresponds to the IDFT values.

%using the IDFT values, compute the DFT using (7.6.7) and recover the
%information symbols

pause %Press any key to see a plot of x(t)

%plot 0 to 100 in the x axis, and the absolute value of x(t) in the y axis.
% Y = abs( X ) returns the absolute value of each element in input X . 
% If X is complex, abs(X) returns the complex magnitude. 
plot([0:100], abs(xt))

%check the difference between xn and samples of x(t)
for n = 0:N-1
    d(n+1) = xt(T/N*n+1) - xn(1+n);
    echo off
end

echo on

%The norm function calculates several different types of matrix norms: 
% e = norm(d) returns the largest singular value of d, which is the set of
% differences between xt and xn
e = norm(d)

%set up a 1 by 10 matrix of zeros
Y = zeros(1,10);

%Compute the DFT using (7.6.7)
for k = 1:9
    for n = 0:N-1
        Y(1,k+1) = Y(1,k+1) + 1/sqrt(N)*xn(n+1)*exp(-j*2*pi*k*n/N);
        echo off
    end
end

echo on

%get the difference between the first ten elements of Y and X, then get the
%largest singular value of this list of differences. 
dd = Y(1:10) - X(1:10)
ee = norm(dd)
