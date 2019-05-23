%Autor: Denilson Gomes Vaz da Silva
%Graduando em Engenharia da Computação
%Inteligência Computacional - Dr. Jarbas Joaci
%Função que permuta os dados

%função recebe os dados e os retorna permutados
function [X,Y] = permuta(dados, classes) 
[~,tam] = size(dados); %recebe a dimensão dos dados
indices = randperm(tam,tam); %escolhe tam indices aleatoriamente de 1 a tam (embaralha)
for i=1:tam %para todos os vetores
    X(:,i) = dados(:,indices(i)); %permuta os dados
    Y(:,i) = classes(:,indices(i)); %permuta as classes
end