clear all;
close all;

% Task 1
% Loading data for the exercise
data_training = csvread('reg-2d-train.csv');
data_test = csvread('reg-2d-test.csv');

%Extracting X and y for training set
X = data_training(:,1:2);
N = length(X);
append = ones(N,1);
X = [append X];
y = data_training(:,end);

% 
% %Task 1.1: 
% %Implementation of OLS using the closed-form solution presented
w = pinv(((X)'*X))*(X')*y;
% 
% %Task 1.2:
% %Weights obtained during the training:
% w
% %Model Error Emse for the training set
Emse1 = ((w')*(X')*X*w - 2*((X*w)')*y + (y')*y)/N
h1 = (w')*X';
 
%Extracting X and y for test set
X2 = data_test(:,1:2);
N2 = length(X2);
append2 = ones(N2,1);
X2 = [append2 X2];

y2 = data_test(:,end);
 
%Model Error Emse for the test set
Emse2 = ((w')*(X2')*X2*w - 2*((X2*w)')*y2 + (y2')*y2)/N2
h2 = (w')*X2';
% 
% %Comparison for the training set: Function obtained vs real function
subplot(121)
plot(y)
grid
hold on
plot(h1,'g')
 
subplot(122)
plot(y2)
grid
hold on
plot(h2,'g')
% 
% %Loading data for task 1.3
data_training2 = csvread('reg-1d-train.csv');
data_test2 = csvread('reg-1d-test.csv');
 
%Extracting X and y for training set
X3 = data_training2(:,1);
N3 = length(X3);
append3 = ones(N3,1);
X3 = [append3 X3];
y3 = data_training2(:,end);

% %Extracting X and y for test set
X4 = data_test2(:,1);
N4 = length(X4);
append4 = ones(N4,1);
X4 = [append4 X4];
y4 = data_test2(:,end);
% 
% %Implementation of OLS using the closed-form solution presented
w2 = pinv(((X3)'*X3))*(X3')*y3;
h3 = (w2')*X3';
h4 = (w2')*X4';
% %Model Error Emse for the training and set sets
Emse3 = ((w2')*(X3')*X3*w2 - 2*((X3*w2)')*y3 + (y3')*y3)/N3
Emse4 = ((w2')*(X4')*X4*w2 - 2*((X4*w2)')*y4 + (y4')*y4)/N4
% 
figure(2)
% 
subplot(121)
plot(X3(:,end), y3,'og')
grid 
hold on
plot(X3(:,end),h3,'rx')
refline(w2(2),w2(1))

subplot(122)
plot(X4(:,end),y4,'og')
grid
hold on
plot(X4(:,end),h4,'rx')
refline(w2(2),w2(1))




