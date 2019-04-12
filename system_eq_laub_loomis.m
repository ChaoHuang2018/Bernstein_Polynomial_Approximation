function final_val = system_eq_laub_loomis(x_initial,time, control_input)

global simulation_result;

function dxdt = laub(t,x)
 
    e = 0.1;
    dxdt =[ 1.4*x(3)-0.9*x(1);
            2.5*x(5)-1.5*x(2);
            %control_input; 
            0.6*x(7)-0.8*x(2)*x(3);
            2-1.3*x(3)*x(4);
            0.7*x(1)-x(4)*x(5);
            0.3*x(1)-3.1*x(6);
            1.8*x(6)-1.5*x(2)*x(7);];
end

[t ,y] = ode45(@laub, [0 time],x_initial);

simulation_result = [simulation_result y'];

s = size(y);
final_val = y(s(1),:);
final_val = final_val';

end