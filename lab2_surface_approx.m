%% lab2
clc
close all

%Turimi duomenys
[x1, x2] = meshgrid(0.1:1/22:1);%create twoD grid to solve surface approx
[x1_test, x2_test] = meshgrid(0.1:1/200:1);
y = (1 + 0.6 * sin(2 * pi * x1 / 0.7)) + 0.3 * sin(2 * pi * x2);  % modifikuota funkcija, turinti du kintamuosius
y_test = (1 + 0.6 * sin(2 * pi * x1_test / 0.7)) + 0.3 * sin(2 * pi * x2_test);  % modifikuota funkcija, turinti du kintamuosius


%Svoriai
%Pirmas sluoksnis
w11_1=rand(1);
w21_1=rand(1);
w31_1=rand(1);
w41_1=rand(1);
w51_1=rand(1);
w61_1=rand(1);
%Antras iejimas
w12_1=rand(1);
w22_1=rand(1);
w32_1=rand(1);
w42_1=rand(1);
w52_1=rand(1);
w62_1=rand(1);


b1_1=rand(1);
b2_1=rand(1);
b3_1=rand(1);
b4_1=rand(1);
b5_1=rand(1);
b6_1=rand(1);

%Antras sluoksnis
w11_2=rand(1);
w12_2=rand(1);
w13_2=rand(1);
w14_2=rand(1);
w15_2=rand(1);
w16_2=rand(1);

b1_2=rand(1);

eta=0.1;

% Initialize output matrix Y to match the dimensions of x1 and x2
Y = zeros(size(x1));%we need matrix same size as x1 and x2 since we work with 2D
Y_test = zeros(size(x1_test));
% Number of epochs
ciklu_sk = 10000;

% Training loop
for j = 1:ciklu_sk
    for i = 1:size(x1, 1)%iterate through all x1 values
        for k = 1:size(x1, 2)%iterate through all x2 values
            % Forward pass for each element in x1, x2
            v1_1 = w11_1 * x1(i, k) + w12_1 * x2(i, k) + b1_1;
            v2_1 = w21_1 * x1(i, k) + w22_1 * x2(i, k) + b2_1;
            v3_1 = w31_1 * x1(i, k) + w32_1 * x2(i, k) + b3_1;
            v4_1 = w41_1 * x1(i, k) + w42_1 * x2(i, k) + b4_1;
            v5_1 = w51_1 * x1(i, k) + w52_1 * x2(i, k) + b5_1;
            v6_1 = w61_1 * x1(i, k) + w62_1 * x2(i, k) + b6_1;

            % Activation for hidden layer
            y1_1 = tanh(v1_1);
            y2_1 = tanh(v2_1);
            y3_1 = tanh(v3_1);
            y4_1 = tanh(v4_1);
            y5_1 = tanh(v5_1);
            y6_1 = tanh(v6_1);

            % Output neuron
            v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + ...
                   y4_1 * w14_2 + y5_1 * w15_2 + y6_1 * w16_2 + b1_2;

            % Output layer activation (linear)
            y1_2 = v1_2;

            % Store the output in the matrix Y
            Y(i, k) = y1_2;

            % Compute the error
            e = y(i, k) - y1_2;

            % Backpropagation
            delta1_2 = e;

            % Hidden layer error gradients
            delta1_1 = (1 - tanh(v1_1)^2) * delta1_2 * w11_2;
            delta2_1 = (1 - tanh(v2_1)^2) * delta1_2 * w12_2;
            delta3_1 = (1 - tanh(v3_1)^2) * delta1_2 * w13_2;
            delta4_1 = (1 - tanh(v4_1)^2) * delta1_2 * w14_2;
            delta5_1 = (1 - tanh(v5_1)^2) * delta1_2 * w15_2;
            delta6_1 = (1 - tanh(v6_1)^2) * delta1_2 * w16_2;

            % Update weights for output layer
            w11_2 = w11_2 + eta * delta1_2 * y1_1;
            w12_2 = w12_2 + eta * delta1_2 * y2_1;
            w13_2 = w13_2 + eta * delta1_2 * y3_1;
            w14_2 = w14_2 + eta * delta1_2 * y4_1;
            w15_2 = w15_2 + eta * delta1_2 * y5_1;
            w16_2 = w16_2 + eta * delta1_2 * y6_1;
            b1_2 = b1_2 + eta * delta1_2;

            % Update weights for hidden layer
            w11_1 = w11_1 + eta * delta1_1 * x1(i, k);
            w12_1 = w12_1 + eta * delta1_1 * x2(i, k);
            w21_1 = w21_1 + eta * delta2_1 * x1(i, k);
            w22_1 = w22_1 + eta * delta2_1 * x2(i, k);
            w31_1 = w31_1 + eta * delta3_1 * x1(i, k);
            w32_1 = w32_1 + eta * delta3_1 * x2(i, k);
            w41_1 = w41_1 + eta * delta4_1 * x1(i, k);
            w42_1 = w42_1 + eta * delta4_1 * x2(i, k);
            w51_1 = w51_1 + eta * delta5_1 * x1(i, k);
            w52_1 = w52_1 + eta * delta5_1 * x2(i, k);
            w61_1 = w61_1 + eta * delta6_1 * x1(i, k);
            w62_1 = w62_1 + eta * delta6_1 * x2(i, k);

            b1_1 = b1_1 + eta * delta1_1;
            b2_1 = b2_1 + eta * delta2_1;
            b3_1 = b3_1 + eta * delta3_1;
            b4_1 = b4_1 + eta * delta4_1;
            b5_1 = b5_1 + eta * delta5_1;
            b6_1 = b6_1 + eta * delta6_1;
        end
    end
end

for i = 1:size(x1_test, 1)%iterate through all x1 values
        for k = 1:size(x1_test, 2)%iterate through all x2 values
            % Forward pass for each element in x1, x2
            v1_1 = w11_1 * x1_test(i, k) + w12_1 * x2_test(i, k) + b1_1;
            v2_1 = w21_1 * x1_test(i, k) + w22_1 * x2_test(i, k) + b2_1;
            v3_1 = w31_1 * x1_test(i, k) + w32_1 * x2_test(i, k) + b3_1;
            v4_1 = w41_1 * x1_test(i, k) + w42_1 * x2_test(i, k) + b4_1;
            v5_1 = w51_1 * x1_test(i, k) + w52_1 * x2_test(i, k) + b5_1;
            v6_1 = w61_1 * x1_test(i, k) + w62_1 * x2_test(i, k) + b6_1;

            % Activation for hidden layer
            y1_1 = tanh(v1_1);
            y2_1 = tanh(v2_1);
            y3_1 = tanh(v3_1);
            y4_1 = tanh(v4_1);
            y5_1 = tanh(v5_1);
            y6_1 = tanh(v6_1);

            % Output neuron
            v1_2 = y1_1 * w11_2 + y2_1 * w12_2 + y3_1 * w13_2 + ...
                   y4_1 * w14_2 + y5_1 * w15_2 + y6_1 * w16_2 + b1_2;

            % Output layer activation (linear)
            y1_2 = v1_2;

            % Store the output in the matrix Y
            Y_test(i, k) = y1_2;
        end
 end

% Plot the approximated surface with a specific color (e.g., red)
figure;
surf(x1_test, x2_test, Y_test, 'FaceColor', 'red', 'EdgeColor', 'none');  % Approximated surface
hold on;

% Plot the target surface with a different color (e.g., blue)
surf(x1_test, x2_test, y_test, 'FaceColor', 'blue', 'EdgeColor', 'none');  % Target surface

% Add a legend and labels
legend('Predicted Surface', 'Target Surface');
xlabel('x1');
ylabel('x2');
zlabel('Output');
title('Surface Approximation Comparison');
hold off;
