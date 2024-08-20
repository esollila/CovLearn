function [Ilocs,gam0,sigc] = CLOMP(A,Y,K)
% function [Ilocs,gam0,sigc] = CLOMP(A,Y,K)
% 
% Estimating the (sparse) signal support using the greedy covariance 
% learning method (CL-OMP) described in [1,2].
% 
% INPUT: 
%   A       - Dictionary of size N x M
%   Y       - Matrix of N x L (L is the number of measurement vectors)
%   K       - number of non-zero sources
% 
% OUTPUT:
%   Ilocs  -  support of non-zeros signal powers (K-vector)
%   gam0   -  Kx1 vector containing non-zero source signal powers
%   sigc   -  noise variance estimate (positive scalar)
% 
% REFERENCE:
% 
%   [1] Esa Ollila, "Sparse signal recovery and source localization via 
%   covariance learning," arXiv:2401.13975 [stat.ME], Jan 2024. 
%   https://arxiv.org/abs/2401.13975
% 
%   [2] Esa Ollila, "Matching pursuit covariance learning", EUSIPCO-2024, 
%   Lyon, August 26-30, 2024. 
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%% Initialize variables
N = size(A,1);% number of sensors
M = size(A,2);% number of dictionary entries
L = size(Y,2);% number of snapshots in the data covariance

%-- assert: 
assert(isequal(round(K),K) && K <= M && isreal(K) && K >0, ... 
    ['''K'' must be a positive integer smaller than ', num2str(M)]);
RY = (1/L)*Y*(Y'); 

%% Initialize
sigc = real(trace(RY)/N); % noise estimate when gamma = 0 
SigmaYinv = (1/sigc)*eye(N); 
SigmaY= sigc*eye(N);
Ilocs = zeros(1,K);

for k = 1:K
    
    %% 1. go through basis vectors
    B =  SigmaYinv*A; % \Sigma^-1 a_m , m=1,..,M
    gamma_num = subplus(real(sum(conj(B).*((RY-SigmaY)*B))));
    gamma_denum = subplus(real(sum(conj(A).*B)));
    gamma_denum(gamma_denum<=10^-12) = 10^-12; % make sure not zero
    gamma = gamma_num./gamma_denum.^2;   
    tmp  = gamma.*gamma_denum;
    fk = log(1+tmp) - tmp;
    
    %% 2. Update the index set 
    [~,indx] = min(fk);
    if k > 1 && (length(setdiff(Ilocs(1:(k-1)),indx)) ~= k-1)
        %fprintf("error: same index is found!");
        tmp = setdiff(1:M,Ilocs(1:k-1));
        [~,indx] = min(fk(tmp));
        indx = tmp(indx);
    end
    Ilocs(k) = indx;

    %% 3. Update gamma0 and noise power sigma^2 
    Am = A(:,Ilocs(1:k)); 
    Am_plus = pinv(Am);
    P_N = eye(N)-Am*Am_plus;
    al = real(trace(P_N*RY));
    sigc = al/(N - k); 
    if k==1
        gam0 = gamma(Ilocs(1))*(N/(N-1));
    else
        gam0 = real(diag(Am_plus*(RY  - sigc*eye(N))*Am_plus'));
        if ~all(gam0>0)
            % If not all powers are positive then find the signal powers
            % using the method described in Bresler (1988)
            [~,gam0,sigc] = SML_MLE(Am,RY,al);
        end
    end
    %% 4. Update the covariance matrix
    SigmaY = sigc*eye(N) + Am*diag(gam0)*Am';        
    SigmaYinv   = SigmaY \ eye(N); 
end

[~,idx] =  sort(Ilocs);
Ilocs = Ilocs(idx);
gam0 = gam0(idx).';

end

function [Gamhat,gamhat,sigmahat] = SML_MLE(A,RY,al)
%
% If not all powers are positive then find the signal powers
% using the method described in Bresler (1988)
% 
% Parameters:
% A -- (N, K) matrix of selected dictionary atoms
% RY -- (N, N) data covariance matrix
% al -- scaled noise estimate ( = trace( (I-A*pinv(A))*RY) )
% 
% Returns:
% Gamhat -- estimated covariance matrix of the signal
% gamhat -- signal power estimates
% sigmahat -- estimated noise power
% 
% Author: Esa Ollila, Aalto University 
%
[N,K] = size(A);
KtoZero= K:-1:0;

%1. Compute the Cholesky factor:
L = chol(A'*A,'lower');
Linv = L \ eye(K);

%2. Compute X and its EV
X =  Linv*(A'*RY*A)*Linv';
[G,Phi] = eig(X);
[phi,ind] = sort(real(diag(Phi)),'ascend');
G = G(:,ind);

%3. Compute sigma^2-s 
sigmasq = (al+cumsum([0;phi]))./(N-[K:-1:0].');
ind1 = sigmasq >= [0;phi];
ind2 = sigmasq <= [phi;Inf];
indvec = ind1 & ind2 ;
ii = find(indvec,1,"first");

% sigmasq lopp done, set sigmasq_hat = sigmasq(ii)
k = KtoZero(ii);
sigmahat = sigmasq(ii);
phi = phi(K:-1:1);
G = G(:,K:-1:1);
psi = phi - sigmahat;
psi(k+1:K) = 0;
Gamhat = Linv'*(G*diag(psi)*G')*Linv;
gamhat = real(diag(Gamhat));

end