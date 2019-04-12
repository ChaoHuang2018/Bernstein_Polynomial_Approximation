Ts = 0.2;  % Sample Time
N = 3;    % Prediction horizon
Duration = 18; % Simulation horizon

radius = 0.2;

global simulation_result;
global a;

for m=1:100

    
x1 = 1.1 + radius*rand(1);
x2 = 0.95 + radius*rand(1);
x3 = 1.4 + radius*rand(1);
x4 = 2.3 + radius*rand(1);
x5 = 0.9 + radius*rand(1);
x6 = 0 + radius*rand(1);
x7 = 0.35 + radius*rand(1);




x = [x1;x2;x3;x4;x5;x6;x7];

simulation_result = x;
a = [0.6*x7-0.8*x2*x3];


% Apply the control input constraints

x_now = x;

for ct = 1:(Duration/Ts)
    
     %u = 0;
     u = NN_output(x_now,10,1,'nn_LLM');
     
      
%     x_next = system_eq_NN(x_now, Ts, u);
    x_next = system_eq_laub_loomis(x_now, Ts, u);

    x = x_next;
    x_now = x_next;
end

plot(simulation_result(1,:),simulation_result(2,:), 'color', 'r');
xlabel('x');
ylabel('y');
hold on;

    
    
end

disp('----x1-----')
min(simulation_result(1,:))
max(simulation_result(1,:))
disp('----x2-----')
min(simulation_result(2,:))
max(simulation_result(2,:))
disp('----x3-----')
min(simulation_result(3,:))
max(simulation_result(3,:))
disp('----x4-----')
min(simulation_result(4,:))
max(simulation_result(4,:))
disp('----x5-----')
min(simulation_result(5,:))
max(simulation_result(5,:))
disp('----x6-----')
min(simulation_result(6,:))
max(simulation_result(6,:))
disp('----x7-----')
min(simulation_result(7,:))
max(simulation_result(7,:))
disp('----u-----')
min(a)
max(a)