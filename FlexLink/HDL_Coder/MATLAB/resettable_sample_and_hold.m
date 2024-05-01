function output   = resettable_sample_and_hold(input, reset, trigger)
%#codegen

%Resettable Sample and Hold Circuit
% This was written because S&H within Resettable Subsystems in Simulink
% is not supported by HDL Coder. We needed this functionality, so this
% custom function was written. 



initial_value = false;

    persistent sample_held;
    if isempty(sample_held)
        sample_held = false;
    end

    persistent output_held_value;
    if isempty(output_held_value)
        output_held_value = false;
    end


switch reset
    case 0 % not in reset
        switch trigger
            case 0
                output = output_held_value;
            case 1
                if sample_held == false
                    output_held_value = input;
                    output = output_held_value;
                    sample_held = true;
                else
                    output = output_held_value;
                end
            otherwise
                output = initial_value;
        end
    case 1 % active high reset condition received
        output = initial_value;
        sample_held = false;
        output_held_value = false;
    otherwise
        output = initial_value;
        
end