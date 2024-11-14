%% lab2
clc
close all

%Turimi duomenys
 x = 0.1:1/22:1;
 x_test=0.1:1/200:1;
 y = ((1+0.6*sin(2*pi*x/0.7))+0.3*sin(2*pi*x))/2;
 y_test = ((1+0.6*sin(2*pi*x_test/0.7))+0.3*sin(2*pi*x_test))/2;

figure(1)
plot(x_test,y_test)
grid on

%Svoriai
%Pirmas sluoksnis
w11_1=rand(1);
w21_1=rand(1);
w31_1=rand(1);
w41_1=rand(1);
w51_1=rand(1);
w61_1=rand(1);

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

Y=zeros(1,length(x));

Y_test=zeros(1,length(x));

ciklu_sk=10000;

%Atsakas
for j=1:ciklu_sk
    for i=1:length(x)
        v1_1=w11_1*x(i)+b1_1;
        v2_1=w21_1*x(i)+b2_1;
        v3_1=w31_1*x(i)+b3_1;
        v4_1=w41_1*x(i)+b4_1;
        v5_1=w51_1*x(i)+b5_1;
        v6_1=w61_1*x(i)+b6_1;
        %aktyvacijos funkcija
        y1_1=tanh(v1_1);
        y2_1=tanh(v2_1);
        y3_1=tanh(v3_1);
        y4_1=tanh(v4_1);
        y5_1=tanh(v5_1);
        y6_1=tanh(v6_1);
        %Antras
        v1_2=y1_1*w11_2+y2_1*w12_2+y3_1*w13_2+y4_1*w14_2+y5_1*w15_2+y6_1*w16_2+b1_2;
        %Išėjimo sluoksnio aktyvacija
        y1_2=v1_2;
        Y(i)=y1_2;
        %Klaidos skaiciavimas
        e=y(i)-y1_2;
        %Svoriu atnaujinimas
        delta1_2=e;
        %Pasleptojo sl klaidos gradientas
        delta1_1=(1-tanh(v1_1)^2)*delta1_2*w11_2;
        delta2_1=(1-tanh(v2_1)^2)*delta1_2*w12_2;
        delta3_1=(1-tanh(v3_1)^2)*delta1_2*w13_2;
        delta4_1=(1-tanh(v4_1)^2)*delta1_2*w14_2;
        delta5_1=(1-tanh(v5_1)^2)*delta1_2*w15_2;
        delta6_1=(1-tanh(v6_1)^2)*delta1_2*w16_2;
        %Atnaujinam svorius
        w11_2=w11_2+eta*delta1_2*y1_1;
        w12_2=w12_2+eta*delta1_2*y2_1;
        w13_2=w13_2+eta*delta1_2*y3_1;
        w14_2=w14_2+eta*delta1_2*y4_1;
        w15_2=w15_2+eta*delta1_2*y5_1;
        w16_2=w16_2+eta*delta1_2*y6_1;
        b1_2=b1_2+eta*delta1_2;
        %pasleptasis
        w11_1=w11_1+eta*delta1_1*x(i);
        w21_1=w21_1+eta*delta2_1*x(i);
        w31_1=w31_1+eta*delta3_1*x(i);
        w41_1=w41_1+eta*delta4_1*x(i);
        w51_1=w51_1+eta*delta5_1*x(i);
        w61_1=w61_1+eta*delta6_1*x(i);
        b1_1=b1_1+eta*delta1_1;
        b2_1=b2_1+eta*delta2_1;
        b3_1=b3_1+eta*delta3_1;
        b4_1=b4_1+eta*delta4_1;
        b5_1=b5_1+eta*delta5_1;
        b6_1=b6_1+eta*delta6_1;
    end
end

%Testing function
for i=1:length(x_test)
        v1_1=w11_1*x_test(i)+b1_1;
        v2_1=w21_1*x_test(i)+b2_1;
        v3_1=w31_1*x_test(i)+b3_1;
        v4_1=w41_1*x_test(i)+b4_1;
        v5_1=w51_1*x_test(i)+b5_1;
        v6_1=w61_1*x_test(i)+b6_1;
        %aktyvacijos funkcija
        y1_1=tanh(v1_1);
        y2_1=tanh(v2_1);
        y3_1=tanh(v3_1);
        y4_1=tanh(v4_1);
        y5_1=tanh(v5_1);
        y6_1=tanh(v6_1);
        %Antras
        v1_2=y1_1*w11_2+y2_1*w12_2+y3_1*w13_2+y4_1*w14_2+y5_1*w15_2+y6_1*w16_2+b1_2;
        %Išėjimo sluoksnio aktyvacija
        y1_2=v1_2;
        Y_test(i)=y1_2;
 end

%plot testing results
hold on
plot(x_test,Y_test)
hold off
legend('Target', 'Predicted')

%hold on
%plot(x_test,Y_test)
%hold off
%legend('Target', 'Predicted')
