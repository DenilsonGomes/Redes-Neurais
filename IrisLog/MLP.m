clear;
clc

%Carregar os dados
load 'iris_log.dat';

%Retirar os valores de entrada
X = iris_log(:,1:4)';

%¨Retirar os valores de saida
Y = iris_log(:,5:7)';

%Criar a rede MLP
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
z = length(X)/10;
for i=1:10
    Xtreinamento = X; 
    Ytreinamento = Y;
    for j=z:-1:1
        Xteste(:,j) = X(:,(i-1)*15 + j); %amostras para testar o modelo
        Xtreinamento(:,(i-1)*15 + j) = []; %ficam so as amostras que vão treinar o modelo
        Yteste(:,j) = Y(:,(i-1)*15 + j); %classes para verificar acuracia do modelo
        Ytreinamento(:,(i-1)*15 + j) = []; %ficam so as classes que vão treinar o modelo
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
end
str = ['Acuracia media obtida pelo 10-fold: ' num2str(100*acertos/length(Xteste)) ' %'];
disp(str); %exibe a Acuracia obtida

%----leave-one-out----
acertos=0;
for i=1:150
    Xtreino = X; 
    Ytreino = Y;
    Xteste = X(:,i); %amostra para testar o modelo
    Xtreino(:,i) = []; %ficam so as amostras que vão treinar o modelo
    Yteste = Y(:,i); %classe para verificar acuracia do modelo
    Ytreino(:,i) = []; %ficam so as classes que vão treinar o modelo
  
    %treinando a rede
    net = train(net,Xtreino,Ytreino);%função para treinar a rede com as amostras treino
    
    %Testando a rede
    Yout = round(net(Xteste));%testar a rede com as amostras teste
    
    %verifica acuracia do metodo
    if Yteste == Yout;
        acertos=acertos+1;
    end
end
str = ['Acuracia obtida pelo leave-one-out: ' num2str(100*acertos/length(X)) ' %'];
disp(str); %exibe a Acuracia obtida