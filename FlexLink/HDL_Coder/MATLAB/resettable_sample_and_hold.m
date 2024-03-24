function [output] = resettable_sample_and_hold(input, reset, trigger)
%#codegen

%Resettable Sample and Hold Circuit
% This was written because S&H within Resettable Subsystems in Simulink
% is not supported by HDL Coder. We needed this functionality, so this
% custom function was written. 

initial_value = 0; 
sample_held = 0; 


switch reset
    case 0 % not in reset
        output = initial_value; % may not need this
        switch trigger
            case 0
                output = initial_value;
            case 1
                if sample_held == 0
                    output = input;
                    sample_held = 1;
                end
        end
    case 1 % reset signal received
        output = initial_value;
        sample_held = 0;
        
end

