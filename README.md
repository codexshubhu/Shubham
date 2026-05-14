Sure. Below is ALL code in ONE SINGLE BLOCK (same code, no changes) so you can copy everything at once and paste into Word.


---

%% ================================
%  1) Tool Wear Prediction (ANN)
%% ================================
clear all;
clc;
close all;

% Input variables
feed = [0.10 0.15 0.20 0.25 0.30];
speed = [100 120 110 130 140];

ip = [feed; speed];

% Output variable
tw = [0.11 0.16 0.20 0.24 0.30];

op = tw;

% Create and train neural network
net = feedforwardnet(10);
net = train(net, ip, op);

% Perform Simulation
o1 = net(ip);

% Error calculation
Err = (op - o1);

disp('Predicted Tool Wear:')
disp(o1)

disp('Error:')
disp(Err)


%% ================================
%  2) Population Growth Simulation
%% ================================
clc;
clear;

P0 = 100;        % Initial population
r = 0.05;        % Growth rate (5%)

t = 0:1:50;      % Time period

P = P0 * exp(r * t);   % Exponential growth formula

% Plot
figure;
plot(t, P, 'b', 'LineWidth', 2);
xlabel('Time');
ylabel('Population');
title('Population Growth Simulation');
grid on;


%% ================================
%  3) Traffic Flow Simulation
%% ================================
clc;
clear;

n = 50;                      % Number of vehicles
t = 0:1:100;                 % Time

arrival = cumsum(rand(1,n)*2);  % Random arrival intervals

speed = 10 + rand(1,n)*5;    % Random speeds (10–15 m/s)

position = zeros(n, length(t));

for i = 1:n
    for j = 1:length(t)
        if t(j) >= arrival(i)
            position(i,j) = speed(i) * (t(j) - arrival(i));
        else
            position(i,j) = 0;
        end
    end
end

% Plot
figure;
plot(t, position);
xlabel('Time');
ylabel('Position');
title('Traffic Flow Simulation');


%% ================================
%  4) Signal Noise Filtering
%% ================================
clc;
clear;

t = 0:0.01:1;                 % Time vector
signal = sin(2*pi*5*t);       % Original sinusoidal signal

noise = 0.5 * randn(size(t)); % Generate noise
noisy_signal = signal + noise;

% Moving Average Filter
window_size = 5;
filtered_signal = filter(ones(1,window_size)/window_size, 1, noisy_signal);

% Plotting
figure;

subplot(3,1,1);
plot(t, signal);
title('Original Signal');

subplot(3,1,2);
plot(t, noisy_signal);
title('Noisy Signal');

subplot(3,1,3);
plot(t, filtered_signal);
title('Filtered Signal');


%% ================================
%  5) Robot Position Estimation
%% ================================
clear all;
clc;
close all;

% Training data
s1 = [10 15 20 25 30 35 40 45 50 55 60 65];
s2 = [12 18 22 28 32 38 42 48 52 58 62 68];

inputs = [s1; s2];

x = [5 7 10 12 15 18 20 22 25 28 30 32];
y = [6 9 11 14 17 19 21 24 26 29 31 34];

targets = [x; y];

% Train network
net = feedforwardnet(10);
net = train(net, inputs, targets);

% TEST INPUT (THIS IS CASE 3)
test_input = [25; 30];

predicted_position = net(test_input);

disp('Robot Position:')
disp(predicted_position)


%% ================================
%  6) Production Rate Prediction
%% ================================
clear all;
clc;
close all;

% -------------------------------
% Step 1: Training Data
% -------------------------------

% Input (Machine conditions)
% Speed (rpm) and Time (hours)
speed = [100 120 140 160 180 200 220 240 260 280 300 320];
time  = [1   1.5 2   2.5 3   3.5 4   4.5 5   5.5 6   6.5];

inputs = [speed; time];

% Output (Production Rate)
production = [50 60 72 85 95 110 125 140 155 170 185 200];

targets = production;

% -------------------------------
% Step 2: Create & Train Network
% -------------------------------

net = feedforwardnet(10);

[net, tr] = train(net, inputs, targets);

% -------------------------------
% Step 3: Test Prediction
% -------------------------------

% New input (you can change this)
test_input = [150; 2.2];

predicted_production = net(test_input);

disp('Predicted Production Rate:')
disp(predicted_production)

% -------------------------------
% Step 4: Graphs (for marks)
% -------------------------------

figure;
plotperform(tr);

figure;
plotregression(targets, net(inputs));


%% ================================
%  7) Monte Carlo Simulation
%% ================================
clc;
clear;

n = 10000;              % Number of trials
p = 0.1;                % Probability of failure

failures = 0;

for i = 1:n
    r = rand();         % Generate random number between 0 and 1
    
    if r < p
        failures = failures + 1;
    end
end

success = n - failures;

reliability = success / n;

disp(['Estimated Reliability: ', num2str(reliability)]);


%% ================================
%  8) Engineering Optimization Problem
%% ================================
clc;
clear;

% Objective function
fun = @(x) x^2 + 4*x + 4;

% Initial guess
x0 = 1;

% Constraint (x >= 0)
lb = 0;   % lower bound
ub = [];  % no upper bound

% Optimization using fmincon
x_opt = fmincon(fun, x0, [], [], [], [], lb, ub);

% Minimum value
f_opt = fun(x_opt);

disp(['Optimal x: ', num2str(x_opt)]);
disp(['Minimum value: ', num2str(f_opt)]);


%% ================================
%  9) Energy Consumption Prediction
%% ================================
clear all;
clc;
close all;

% ---------------------------------
% Step 1: Training Data
% ---------------------------------

% Input Data (Load in kW, Usage in hours)
load_data = [2 3 4 5 6 7 8 9 10 11 12 13];
usage     = [1 2 2 3 3 4 4 5 5 6 6 7];

inputs = [load_data; usage];

% Output Data (Energy Consumption)
energy = [2 6 8 15 18 28 32 45 50 66 72 91];

targets = energy;

% ---------------------------------
% Step 2: Create & Train Network
% ---------------------------------

net = feedforwardnet(10);

[net, tr] = train(net, inputs, targets);

% ---------------------------------
% Step 3: Predict Energy Consumption
% ---------------------------------

% New Input (you can change values)
test_input = [7; 3];   % load = 7 kW, usage = 3 hours

predicted_energy = net(test_input);

disp('Predicted Energy Consumption:')
disp(predicted_energy)

% ---------------------------------
% Step 4: Graphs (for marks)
% ---------------------------------

figure;
plotperform(tr);

figure;
plotregression(targets, net(inputs));


%% ================================
%  10) Demand Forecasting using ANN
%% ================================
clear all;
clc;
close all;

% ---------------------------------
% Step 1: Historical Demand Data
% ---------------------------------

time = [1 2 3 4 5 6 7 8 9 10 11 12];   % months
demand = [100 120 130 150 170 180 200 210 230 250 270 300];

inputs = time;
targets = demand;

% ---------------------------------
% Step 2: Train Neural Network
% ---------------------------------

net = feedforwardnet(10);
[net, tr] = train(net, inputs, targets);

% ---------------------------------
% Step 3: Predict Future Demand
% ---------------------------------

test_input = 13;   % next month

predicted_demand = net(test_input);

disp('Predicted Future Demand:')
disp(predicted_demand)

% ---------------------------------
% Step 4: Graphs
% ---------------------------------

figure;
plotperform(tr);

figure;
plotregression(targets, net(inputs));


%% ================================
%  11) Bouncing Ball Simulation
%% ================================
% Bouncing Ball Simulation - Practical 1
% Parameters and Constants
g = 9.81;               % Gravity (m/s^2)
dt = 1e-4;              % Time step (seconds)
num_steps = 1e5;        % Total iterations (100,000)
elasticity = 0.75;      % 75% elasticity (loses 25% energy)

% Initialization
y = zeros(num_steps, 1); % Pre-allocate height array
y(1) = 70;               % Initial height in meters
v = 0;                   % Initial velocity
time = (0:num_steps-1) * dt;

% Simulation Loop
for i = 2:num_steps
    % Update velocity: v = u + at
    v = v - g * dt;

    % Update height
    y(i) = y(i-1) + v * dt;

    % Check for bounce condition
    if y(i) <= 0
        y(i) = 0;              % Reset to ground level
        v = -v * elasticity;   % Reverse direction and apply energy loss
    end
end

% Plotting the Results
figure;
plot(time, y);
grid on;
box on;
xlabel('Time (seconds)');
ylabel('Height (meters)');
title('Practical 1: Bouncing Ball Simulation');

% Add an orange marker for the ball at the 30,000th step
hold on;
plot(time(30000), y(30000), 'o', 'MarkerFaceColor', [1 0.5 0], 'MarkerSize', 15);
hold off;


%% ================================
%  12) Arithmetic Operation
%% ================================
clc;
clear;

% Input two numbers
a = input('Enter first number: ');
b = input('Enter second number: ');

% Operations
add = a + b;
sub = a - b;
mul = a * b;
div = a / b;
power_val = a ^ b;

% Display results
fprintf('Addition = %f\n', add);
fprintf('Subtraction = %f\n', sub);
fprintf('Multiplication = %f\n', mul);
fprintf('Division = %f\n', div);
fprintf('Power (a^b) = %f\n', power_val);


%% ================================
%  13) Menu Driven Calculator
%% ================================
clc;
clear;

% Input numbers
a = input('Enter first number: ');
b = input('Enter second number: ');

% Display menu
fprintf('\nMENU:\n');
fprintf('1. Addition\n');
fprintf('2. Subtraction\n');
fprintf('3. Multiplication\n');
fprintf('4. Division\n');
fprintf('5. Power\n');

% User choice
choice = input('Enter your choice (1-5): ');

% Switch-case
switch choice
    case 1
        result = a + b;
        fprintf('Addition = %f\n', result);

    case 2
        result = a - b;
        fprintf('Subtraction = %f\n', result);

    case 3
        result = a * b;
        fprintf('Multiplication = %f\n', result);

    case 4
        result = a / b;
        fprintf('Division = %f\n', result);

    case 5
        result = a ^ b;
        fprintf('Power = %f\n', result);

    otherwise
        fprintf('Invalid choice!\n');
end


%% ================================
%  14) Trigonometric Functions
%% ================================
clc;
clear;
close all;

% x values
x = 0:0.1:2*pi;

% functions
y1 = sin(x);
y2 = cos(x);

% plot both
plot(x, y1, 'r', 'LineWidth', 2);
hold on;
plot(x, y2, 'b', 'LineWidth', 2);

% labels
xlabel('x');
ylabel('Value');
title('sin(x) and cos(x)');

legend('sin(x)', 'cos(x)');
grid on;


%% ================================
%  15) Simple Harmonic Motion
%% ================================
clc;
clear;
close all;

% Given values
A = 5;
w = 2;

% Time range
t = 0:0.1:10;

% SHM equation
x = A .* sin(w .* t);

% Plot
figure;
plot(t, x, 'LineWidth', 2);

xlabel('Time (t)');
ylabel('Displacement (x)');
title('Simple Harmonic Motion');
grid on;


%% ================================
%  16) Area and Volume Calculations
%% ================================
clc;
clear;
close all;

% ===== AREA OF A CIRCLE =====
radius = input('Enter radius of the circle: ');
area_circle = pi * radius^2;

% ===== VOLUME OF A CYLINDER =====
r = input('Enter radius of the cylinder: ');
h = input('Enter height of the cylinder: ');
volume_cylinder = pi * r^2 * h;

% ===== SURFACE AREA OF A SPHERE =====
sphere_radius = input('Enter radius of the sphere: ');
surface_area_sphere = 4 * pi * sphere_radius^2;

% ===== DISPLAY RESULTS =====
fprintf('\n----- RESULTS -----\n');

fprintf('Area of Circle = %.2f\n', area_circle);

fprintf('Volume of Cylinder = %.2f\n', volume_cylinder);

fprintf('Surface Area of Sphere = %.2f\n', surface_area_sphere);


%% ================================
%  17) Unit Conversion Program
%% ================================
clc;
clear;
close all;

% ===== TEMPERATURE CONVERSION =====
celsius = input('Enter temperature in Celsius: ');
fahrenheit = (celsius * 9/5) + 32;

% ===== METERS TO FEET =====
meters = input('Enter length in meters: ');
feet = meters * 3.28084;

% ===== KG TO POUNDS =====
kg = input('Enter weight in kilograms: ');
pounds = kg * 2.20462;

% ===== DISPLAY RESULTS =====
fprintf('\n----- CONVERSION RESULTS -----\n');

fprintf('Temperature in Fahrenheit = %.2f F\n', fahrenheit);

fprintf('Length in Feet = %.2f ft\n', feet);

fprintf('Weight in Pounds = %.2f lb\n', pounds);


%% ================================
%  18) Vectors and Matrices
%% ================================
clc;
clear;
close all;

% ===== CREATE TWO 3x3 MATRICES =====
A = [1 2 3;
     4 5 6;
     7 8 9];

B = [9 8 7;
     6 5 4;
     3 2 1];

% ===== MATRIX ADDITION =====
addition = A + B;

% ===== MATRIX SUBTRACTION =====
subtraction = A - B;

% ===== MATRIX MULTIPLICATION =====
multiplication = A * B;

% ===== DETERMINANT =====
det_A = det(A);
det_B = det(B);

% ===== INVERSE =====
% Inverse exists only if determinant is not zero

if det_A ~= 0
    inverse_A = inv(A);
else
    inverse_A = 'Inverse does not exist';
end

if det_B ~= 0
    inverse_B = inv(B);
else
    inverse_B = 'Inverse does not exist';
end

% ===== DISPLAY RESULTS =====
disp('Matrix A = ');
disp(A);

disp('Matrix B = ');
disp(B);

disp('Addition of A and B = ');
disp(addition);

disp('Subtraction of A and B = ');
disp(subtraction);

disp('Multiplication of A and B = ');
disp(multiplication);

fprintf('Determinant of Matrix A = %.2f\n', det_A);
fprintf('Determinant of Matrix B = %.2f\n', det_B);

disp('Inverse of Matrix A = ');
disp(inverse_A);

disp('Inverse of Matrix B = ');
disp(inverse_B);


%% ================================
%  19) Solve Linear Equations
%% ================================
clc;
clear;
close all;

% ===== COEFFICIENT MATRIX =====
A = [2 3 -1;
     4 -1 5;
     1 2 3];

% ===== CONSTANT MATRIX =====
B = [5;
     10;
     7];

% ===== SOLVE EQUATIONS =====
X = A \ B;

% ===== DISPLAY RESULTS =====
fprintf('Solution of the given equations:\n');

fprintf('x = %.2f\n', X(1));
fprintf('y = %.2f\n', X(2));
fprintf('z = %.2f\n', X(3));


%% ================================
%  20) Eigenvalues and Eigenvector
%% ================================
clc;
clear;
close all;

% ===== CREATE A 3x3 MATRIX =====
A = [4 2 1;
     1 3 2;
     2 1 5];

% ===== COMPUTE EIGENVALUES AND EIGENVECTORS =====
[V, D] = eig(A);

% ===== DISPLAY MATRIX =====
disp('Matrix A = ');
disp(A);

% ===== DISPLAY EIGENVALUES =====
disp('Eigenvalues of Matrix A = ');
disp(D);

% ===== DISPLAY EIGENVECTORS =====
disp('Eigenvectors of Matrix A = ');
disp(V);


%% ================================
%  21) Even or Odd Number
%% ================================
clc;
clear;
close all;

% ===== INPUT NUMBER =====
num = input('Enter a number: ');

% ===== CHECK EVEN OR ODD =====
if mod(num,2) == 0
    fprintf('%d is an Even number.\n', num);
else
    fprintf('%d is an Odd number.\n', num);
end


%% ================================
%  22) Grade Calculator
%% ================================
clc;
clear;
close all;

% ===== INPUT MARKS =====
marks = input('Enter marks: ');

% ===== GRADE CALCULATION =====
if marks >= 90
    grade = 'A';
    
elseif marks >= 75
    grade = 'B';
    
elseif marks >= 60
    grade = 'C';
    
elseif marks >= 40
    grade = 'D';
    
else
    grade = 'F';
end

% ===== DISPLAY RESULT =====
fprintf('Grade = %s\n', grade);


%% ================================
%  23) Factorial using Loop
%% ================================
clc;
clear;
close all;

% ===== INPUT NUMBER =====
n = input('Enter a number: ');

% ===== FACTORIAL USING FOR LOOP =====
fact = 1;

for i = 1:n
    fact = fact * i;
end

% ===== DISPLAY RESULT =====
fprintf('Factorial of %d = %d\n', n, fact);


%% ================================
%  24) Plot a Mathematical Function
%% ================================
clc;
clear;
close all;

% ===== DEFINE x VALUES =====
x = -10:0.1:10;

% ===== DEFINE FUNCTION =====
y = x.^2 + 3*x + 5;

% ===== PLOT GRAPH =====
plot(x, y, 'b', 'LineWidth', 2);

% ===== ADD TITLE AND LABELS =====
title('Plot of y = x^2 + 3x + 5');
xlabel('x-axis');
ylabel('y-axis');

% ===== ADD GRID =====
grid on;


%% ================================
%  25) Subplots Plot
%% ================================
clc;
clear;
close all;

% ===== DEFINE x VALUES =====
x = -10:0.1:10;

% ===== SUBPLOT 1 =====
subplot(2,2,1);
plot(x, x, 'r', 'LineWidth', 2);
title('y = x');
xlabel('x-axis');
ylabel('y-axis');
grid on;

% ===== SUBPLOT 2 =====
subplot(2,2,2);
plot(x, x.^2, 'b', 'LineWidth', 2);
title('y = x^2');
xlabel('x-axis');
ylabel('y-axis');
grid on;

% ===== SUBPLOT 3 =====
subplot(2,2,3);
plot(x, x.^3, 'g', 'LineWidth', 2);
title('y = x^3');
xlabel('x-axis');
ylabel('y-axis');
grid on;

% ===== SUBPLOT 4 =====
subplot(2,2,4);
plot(x, x.^4, 'm', 'LineWidth', 2);
title('y = x^4');
xlabel('x-axis');
ylabel('y-axis');
grid on;


%% ================================
%  26) Numerical Differentiation
%% ================================
clc;
clear;
close all;

% ===== DISPLACEMENT DATA =====
x = [0 5 15 30 50];     % displacement
t = [0 1 2 3 4];        % time

% ===== NUMERICAL DIFFERENTIATION =====
v = diff(x) ./ diff(t);

% ===== DISPLAY RESULTS =====
disp('Velocity values are:');
disp(v);

% ===== PLOT VELOCITY =====
plot(t(1:end-1), v, 'b-o', 'LineWidth', 2);

title('Velocity vs Time');
xlabel('Time');
ylabel('Velocity');
grid on;


%% ================================
%  27) Numerical Integration
%% ================================
clc;
clear;
close all;

% ===== DEFINE x VALUES =====
x = 0:1:5;

% ===== DEFINE FUNCTION =====
y = x.^2;

% ===== NUMERICAL INTEGRATION USING TRAPEZOIDAL RULE =====
area = trapz(x, y);

% ===== DISPLAY RESULT =====
fprintf('Area under the curve = %.2f\n', area);

% ===== PLOT GRAPH =====
plot(x, y, 'b-o', 'LineWidth', 2);

title('y = x^2');
xlabel('x-axis');
ylabel('y-axis');
grid on;


%% ================================
%  28) User Defined Function
%% ================================
clc;
clear;
close all;

% ===== INPUT VALUES =====
length1 = input('Enter length of rectangle: ');
width1 = input('Enter width of rectangle: ');

% ===== FUNCTION CALL =====
[area, perimeter] = rectangle_calc(length1, width1);

% ===== DISPLAY RESULTS =====
fprintf('Area of Rectangle = %.2f\n', area);
fprintf('Perimeter of Rectangle = %.2f\n', perimeter);

% ===== USER-DEFINED FUNCTION =====
function [area, perimeter] = rectangle_calc(length1, width1)

    % Area calculation
    area = length1 * width1;

    % Perimeter calculation
    perimeter = 2 * (length1 + width1);

end


%% ================================
%  29) Fibonacci Series
%% ================================
clc;
clear;
close all;

% ===== INPUT NUMBER OF TERMS =====
n = input('Enter number of terms: ');

% ===== FUNCTION CALL =====
fib_series(n);

% ===== USER-DEFINED FUNCTION =====
function fib_series(n)

% First two terms
a = 0;
b = 1;

fprintf('Fibonacci Series:\n');

for i = 1:n

    fprintf('%d ', a);

    c = a + b;
    a = b;
    b = c;
end

end


---
