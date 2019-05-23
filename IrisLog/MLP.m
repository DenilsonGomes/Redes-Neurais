clear;
clc

%Carregar os dados
load 'iris_log.dat';

%Retirar os valores de entrada
X = iris_log(:,1:4)';

%¨Retirar os valores de saida
Y = iris_log(:,5:7)';

%Criar a rede MLP
%Entrada -> X
%Saída -> Y
%[10 3] -> rede neural com duas camadas ocultas, a primeira com 10
%neurônios e a segunda com 3 neurônios
%logsig -> função de ativação logistica
%traingdx -> função de treinamento, chamada de Backpropagation de gradiente
%decrescente com momentum e taxa adaptative
net = newff(X,Y,[10 3], {'logsig' 'logsig'}, 'traingdx');


net.trainParam.epochs = 150;%número de épocas
net.trainParam.goal = 0.0001;%erro desejado
net.trainParam.lr = 0.01;%taxa de aprendizagem

%Permuta as amostras, para misturar as classes
[X,Y] = permuta(X,Y);

%----hold-out----
Xtreino = X(:,1:round(0.7*length(X)));
Ytreino = Y(:,1:round(0.7*length(X)));

%treinando a rede
net = train(net,Xtreino,Ytreino);%função para treinar a rede com as amostras treino

%Testando a rede
Xteste = X(:,round(0.7*length(X))+1:end);
Yteste = Y(:,round(0.7*length(X))+1:end);
Yout = round(net(Xteste));%testar a rede com as amostras teste

%verifica acuracia do metodo
acertos=0;
for i=1:length(Xteste)
    if Yteste(:,i) == Yout(:,i);
        acertos=acertos+1;
    end
end
str = ['Acuracia obtida com hold-out: ' num2str(100*acertos/length(Xteste)) ' %'];
disp(str); %exibe a Acuracia obtida

%----10-fold----
for i=1:10
    XTreinamento = X; 
    YTreinamento = Y;
    for j=15:-1:1
        XTeste(:,j) = X(:,(i-1)*15 + j); %amostras para testar o modelo
        XTreinamento(:,(i-1)*15 + j) = []; %ficam so as amostras que vão treinar o modelo
        YTeste(:,j) = Y(:,(i-1)*15 + j); %classes para verificar acuracia do modelo
        YTreinamento(:,(i-1)*15 + j) = []; %ficam so as classes que vão treinar o modelo
    end
    %treinando a rede
    net = train(net,Xtreino,Ytreino);%função para treinar a rede com as amostras treino

    %Testando a rede
    Yout = round(net(Xteste));%testar a rede com as amostras teste

    %verifica acuracia do metodo
    acertos=0;
    for i=1:length(Xteste)
        if Yteste(:,i) == Yout(:,i);
            acertos=acertos+1;
        end
    end
    str = ['Acuracia obtida com 10-fold: ' num2str(100*acertos/length(Xteste)) ' %'];
    disp(str); %exibe a Acuracia obtida
end

