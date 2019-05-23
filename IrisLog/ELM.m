%Autor: Varner Damasceno Junior
%Graduando em Engenharia da Computação
%Inteligência Computacional - Dr. Jarbas Joaci

clc
clf
clear
load aerogerador.dat; %Carrega a base de dados
X1 = aerogerador(:,1); %X é a velocidade do vento
Y = aerogerador(:,2); %Y é a potencia gerada
plot(X1,Y); %plota os dados
hold on; %segura o grafico

%construir vetor de entrada com Baes = -1
X = [-ones(length(X1),1),X1]'; % vetor de entradas X(i) = [Baes,X(i)]
[atributos,~] = size(X); %retorna o numero de atributos com o Baes
%iniciar W1 aleatoriamente
n = input('Digite o numero de neuronios da camada oculta: '); %recebe o numero de neuronios da camada oculta
for i=1:n
    for j=1:atributos
        W1(i,j) = rand(1); %preenche W1 com numeros aleatorios
    end
end

%Treinando
% XTreino = X(:,1:round(0.70*length(X1)));
% YTreino = Y(1:round(0.70*length(X1)));
W2 = (pinv(sigmf(W1*X,[1 1]))'*Y)';

%Obtendo saida
Yout = (W2*sigmf(W1*X,[1 1]))';

%Verificando qualidade do modelo por R²
R(Y,Yout)