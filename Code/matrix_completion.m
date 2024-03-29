close all;
clear all;
clc;

rng('default') % For reproducibility
% Dimensioni della matrice
n1 = 10000;
n2 = 10000;
n = max(n1,n2);
% Numero di campioni
m = round((n1*n2)/100 * 5);
% Massimo numero intero nella natrice
imax = 10;
% Rango della matrice
r = 5;
% Probabilità di Bernoulli
p = m / (n1 * n2);

% Genero la matrice M randomicamente
M = randi([-imax, imax], n1, r) * randi([-imax, imax], r, n2);

% randomly select n indices
m = 200; % set the number of indices to select
idx = randperm(length(row), m);

% create a new sparse matrix with the same size as the original matrix
P_M_mat = sparse(n1,n2);

% insert the selected values into the new sparse matrix
P_M_mat(row(idx), col(idx)) = M(row(idx), col(idx));

% Creating observed matrix
observed = false(n1, n2);
observed(row(idx), col(idx)) = true;

% Calcolo della SVD
[U, S, V] = svdsketch(M);

ei = eye(n1); 
mu_u = n1/r * max(sum((U'*ei).^2, 1)); 
ei = eye(n2); 
mu_v = n2/r * max(sum((V'*ei).^2, 1)); 
mu = max(mu_u, mu_v);

% Calcolo della sample complexity
sam_co = (mu * r * (log(n))^2) /n;

tic;

% Nuclear norm minimization
cvx_begin
    variable Phi(n1,n2)
    minimize ( norm_nuc(Phi) )
    subject to
        Phi(~observed) == P_M_mat(~observed)
cvx_end

time1 = toc;

figure(1)
plot(cvx_slvitrerr(1:cvx_slvitr))
grid()
title('Errore di CVX in funzione delle iterazioni')
xlabel('n° iterazioni')
ylabel("Valore dell'errore di CVX")



% MSE
E    = Phi-M;
SQE  = E.^2;
MSE  = mean(SQE(:));
RMSE_1 = sqrt(MSE);

% Calculate sample complexity
sam_co_2 = (mu^3 * r^3 * (log(n))^3) /n;

X = normrnd(0,1,[n1,r]);
Y = normrnd(0,1,[n2,r]);

sigma_1 = max(diag(S));
sigma_r = min(diag(S));
k = sigma_1/sigma_r;
eta = 2/(25*k*sigma_1);
max_iter = 10000;
tolerance = 1e-5;

% Definisco la cost function e i suoi gradienti
% (calcolati con la chain rule)
F = @(x,y,m) 1/(4*p) * norm(m - P_M_mat, 'fro')^2 + (1/16) * norm(x'*x - y'*y, 'fro')^2;
df_dx = @(x,y,m) 1/(2*p) * ((m - P_M_mat) * y)   + (1/8) * x * (x'*x - y'*y);
df_dy = @(x,y,m) 1/(2*p) * (x' * (m - P_M_mat))' + (1/8) * y * (x'*x - y'*y);

tic;

% Algoritmo Gradient Descent con Inizializzazione Spettrale
iter = 0;
converged = false;
f_vals = zeros(max_iter,1);
error = zeros(max_iter,1);
xy = X*Y';
m = zeros(n1,n2);
m(~observed) = xy(~observed);
while ~converged && iter < max_iter
    X = X - eta * df_dx(X,Y,m);
    Y = Y - eta * df_dy(X,Y,m);
    xy = X*Y';
    m = zeros(n1,n2);
    m(~observed) = xy(~observed);
    f_vals(iter+1) = F(X,Y,m);
    error(iter+1) = norm(xy - M, 'fro') / norm(M, 'fro');
    if iter > 0
        converged = abs(f_vals(iter+1)-f_vals(iter)) < tolerance*f_vals(iter);
    end
    iter = iter + 1;
end

% Matrice ricostruita
M_rec = X*Y';
M_rec(~observed) = M(~observed);

time2 = toc;


% MSE
E    = M_rec-M;
SQE  = E.^2;
MSE  = mean(SQE(:));
RMSE_2 = sqrt(MSE);

figure(2)
plot(error(1:iter))
grid()
title('Errore normalizzato in funzione delle iterazioni')
xlabel('n° iterazioni')
ylabel("Valore dell'errore normalizzato")
