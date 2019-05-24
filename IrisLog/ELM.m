%Autor: Varner Damasceno Junior
%Graduando em Engenharia da Computação

clear;
clc
close all

load 'iris_log.dat';
q = 3; %qtd de neurônios 
p = 4; %qtd atributos
n = 0.001; %taxa de aprendizagem
X = iris_log(:,1:4);
D = iris_log(:,5:7);
x_ones = -ones(150,1);
X = [x_ones X];
W = zeros(q,p+1);
X = X';
D = D';
%----------------método hold-olt-------------------------------------------
XT = [X(:,36:50) X(:,86:100) X(:,136:150)];
for k = 1 : 3000
    for i = [1 : 35, 100:135, 50:85]
        Y = W*X(:,i);
        E = D(:,i) - Y;
        A = (X(:,i))';
        W = W + n*E*A;
    end
end

YT = W*XT;
fprintf('método hold-out: \n');
disp(YT');

%----------------método 10-fold--------------------------------------------
W = zeros(q,p+1);
fprintf('método 10-fold: ');
for x = 1 : 10
    if (x==1)
        lista = [x*15+1 : 150];
    else
        lista = [1:(x-1)*15+1, x*15+1 : 150];
    end
    for i = lista
        Y = W*X(:,i);
        E = D(:,i) - Y;
        A = (X(:,i))';
        W = W + n*E*A;
       
    end
    XT = X(:,(x-1)*15+1:x*15);
    YT(:,(x-1)*15+1:x*15) = W*XT;
end
fprintf('\n');
disp(YT');


%----------------leav-one-out--------------------------------------------
W = zeros(q,p+1);
fprintf('método leave-one-out: ');
for x = 1 : 150
    if (x==1)
        lista = [x+1 : 150];
    else
        lista = [1:x-1, x+1 : 150];
    end
    for i = lista
        Y = W*X(:,i);
        E = D(:,i) - Y;
        A = (X(:,i))';
        W = W + n*E*A;
       
    end
    XT = X(:,x);
    YT(:,x) = W*XT;
end
fprintf('\n');
disp(YT');