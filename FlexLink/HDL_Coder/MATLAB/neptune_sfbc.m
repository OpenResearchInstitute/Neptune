function [tx1, tx2] = neptune_sfbc(in)
%Neptune space frequency block code
%   for two transmit antennas, space frequency block coding is implemented.
%   "in" is arriving as a 2 by 1 column vector. 
persistent hTDEnc;
if isempty(hTDEnc)
    %use same object for both STBC and SFBC - clever coding from 
    % "Understanding LTE with MATLAB" by Houman Zarinnkoub. 
    hTDEnc = comm.OSTBCEncoder('NumTransmitAntennas', 2);
end
new_imaginary = -1*imag(in(2));
new_real = -1*real(in(2));
neg_conj_in2 = complex(new_real, new_imaginary);

new_imaginary = -1*imag(in(1));
new_real = real(in(1));
conj_in1 = complex(new_real, new_imaginary);

tx1 = [in(1); in(2)];
tx2 = [neg_conj_in2; conj_in1];

end