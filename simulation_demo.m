clearvars;
%%
% % simulation parameters
M = 2^8; % = 256 (nr of atoms) 
K = 2^2; % = 4 sparsity 
L = 2^4; % = 16 (nr of snapshots) 
dims = [2^3,2^4,2^5,2^6]; % N= 8, 16, 32, 64

SNR = 1:10;
nSNR = length(SNR);
sigmas = zeros(nSNR,K);

sigmas(:,1) = 10.^(SNR.'/10);
sigmas(:,2) = 10.^((SNR.'-1)/10);
sigmas(:,3) = 10.^((SNR.'-2)/10);
sigmas(:,4) = 10.^((SNR.'-4)/10);

GaussianS = true; % gaussianS = true if non-zero sources are Gaussian distributed
nmethods = 2; % We compare CLOMP and SOMP
cputime = zeros(length(dims),nSNR,nmethods);  % Running time in seconds
recmat  = zeros(length(dims),nSNR,nmethods);  % exact recovery rate 
mdmat   = zeros(length(dims),nSNR,nmethods);  % miss detection rate 

NRSIM = 300; % increase this, was 2000 in the EUSIPCO paper

for ii  = 1:length(dims)

    % iterations through dimensions
    N = dims(ii); 
    fprintf('N = %d, M = %d, L=%d, K=%d',N,M,L,K);

    for  jj = 1:length(SNR)
  
        sig = sigmas(jj,:);        
        Lam = diag(sqrt(sig));
   
        fprintf('Starting iterations for sig = %.2f SNR = %.2f\n',sig(1), SNR(jj));    
        rng('default');

        for iter = 1:NRSIM
    
            %% generate sparse source S and error E           
            S = Lam*complex(randn(K,L),randn(K,L))/sqrt(2); %  complex Gaussian data           
            E = (1/sqrt(2))*complex(randn(N,L),randn(N,L)); 
            A = GaussianMM(N,M);                

            %% generate Y  
            loc = sort(randsample(M,K));  % random locations of nonzeros
            Y = A(:,loc)*S + E;  

            %% method 1: CLOMP (proposed)
            tStart = tic;
            Scl = CLOMP(A,Y,K);
            tEnd = toc(tStart);
            cputime(ii,jj,1) = cputime(ii,jj,1) + tEnd;
            mdmat(ii,jj,1) = mdmat(ii,jj,1) + numel(setdiff(Scl,loc));
            if  isempty(setdiff(Scl,loc)) 
                recmat(ii,jj,1) =recmat(ii,jj,1) + 1;
            end 
                      
            %% Method 2: SOMP 
            tStart = tic;
            [~, Somp] = SOMP(Y,A,K); 
            tEnd = toc(tStart);
            cputime(ii,jj,2) = cputime(ii,jj,2) + tEnd;
            mdmat(ii,jj,2) = mdmat(ii,jj,2) + numel(setdiff(Somp,loc));
            if  isempty(setdiff(Somp,loc)) 
                recmat(ii,jj,2) =recmat(ii,jj,2) + 1;
            end                    
        end

    end
end
recmat = recmat/NRSIM;
mdmat = mdmat/(NRSIM*K);
cputime = cputime/NRSIM
%%
x = SNR;
fignro=1

figure(fignro); clf
for k = 1:4
    subplot(1,4,k);
    hold on;
    plot(x, recmat(k,:,1),'ro-','DisplayName', 'CL-OMP','LineWidth',0.8,'MarkerSize',12);    
    plot(x, recmat(k,:,2),'bx-.','DisplayName','SOMP','LineWidth',0.8,'MarkerSize',12);
    legend('FontSize',18);
    ylabel('Exact recovery'); xlabel('SNR')
    grid on; ylim([0,1])
end

%%
figure(fignro+1); clf
for k = 1:4
    subplot(1,4,k);
    hold on;
    plot(x, mdmat(k,:,1),'ro-','DisplayName', 'CL-OMP','LineWidth',0.8,'MarkerSize',12);    
    plot(x, mdmat(k,:,2),'bx-.','DisplayName','SOMP','LineWidth',0.8,'MarkerSize',12);
    legend('FontSize',18);
    ylabel('Probability of mis-detection'); xlabel('SNR')
    grid on; ylim([0,1])
end
%%
function A = GaussianMM(m,n)
% Generates Gaussian measurement matrix of size m x n 
A = (1/2)*(randn(m,n)+1i*randn(m,n));
len = sqrt(sum(A.*conj(A)));
A = A*diag(1./len);      %  normalized to  unit-norm columns 
end

function [X,S] = SOMP(Y,A,K)
% Simultaneous orthogonal matching pursuit algorithm (SOMP) 
% Y: N X L , A: N X M, K = sparsity
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

[~,L]=size(Y);
X = zeros(K,L);
R = Y;
S = zeros(1,K);

for k=1:K
    E = sum(abs(A'*R),2);
    [~,pos] = max(E); 
    S(k) = pos;
    V = A(:,S(1:k));
    X(1:k,:) = pinv(V)*Y;
    R = Y- V*X(1:k,:);
 
end
end