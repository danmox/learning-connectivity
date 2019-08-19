function [A,B] = nodemarginconsts(qos,R)
% NODEMARGINCONSTS form node margin constraint matrices of the form:
% A(i,:)*y = sum_i (a_ij_k)^2 R.var(i,j) - sum j (a_ji_k)^2 R.var(i,j)
% B(i,:)*y = sum_i a_ij_k R.avg(i,j) - sum j a_ji_k R.avg(i,j) - s
%
% inputs:
%   qos - a struct with the following fields:
%         flow       - a struct with struct with fields:
%                      src  - a list of nodes where the flow originates
%                      dest - a list of nodes where the flow terminates
%         margin     - the required data rate margin at the source node
%         confidence - the probabilistic confidence with which network
%                      constraints associated with this flow should be
%                      satisfied
%   R   - a struct with the following fields:
%         avg - a pairwise matrix of expected channel rates between nodes
%         var - a pairwise matrix of variances in the expected channel
%               rates between nodes
%
% outputs:
%   A - an NKxNNK matrix as described above
%   B - an NKxNNK matrix as described above

N = size(R.avg,1);
K = length(qos);

if ~isnumeric(R.avg) % allows this function to work with symbolic inputs
  A = sym(zeros(N*K, N*N*K+1));
  B = sym(zeros(N*K, N*N*K+1));
  Aseed = sym(zeros(N,N));
  Bseed = sym(zeros(N,N));
else
  A = zeros(N*K, N*N*K+1);
  B = zeros(N*K, N*N*K+1);
  Aseed = zeros(N,N);
  Bseed = zeros(N,N);
end

for k = 1:K
  for i = 1:N
    
    Aki = Aseed;
    Bki = Bseed;
    
    for j = 1:N
      
      if i == j
        continue
      end
      
      Aki(i,j) = R.var(i,j);
      Bki(i,j) = R.avg(i,j);
            
      if ~any(qos(k).flow.dest == j)
        Bki(j,i) = -R.avg(j,i);
        Aki(j,i) = R.var(j,i);
      end
      
    end
    
    A((k-1)*N+i, (k-1)*N*N+1:k*N*N) = sqrt(Aki(:)');
    B((k-1)*N+i, (k-1)*N*N+1:k*N*N) = Bki(:)';
    B((k-1)*N+i,end) = -1; % slack
    
  end
end