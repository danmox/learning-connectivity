function [A, B] = makesocp(qos, R)
% MAKESOCP form SOCP constraints of the form: ||A(i,:)*y|| <= B(i,:)*y for
% i = 1:NK
% 
% inputs:
%   qos   - a Kx1 array of structs containing:
%           flow.src   - the list of source nodes
%           flow.dest  - the list of destination nodes
%           margin     - the margin with which the constraint must be met
%           confidence - probabilistic confidence that the constraint holds
%   R.avg - NxN matrix with R.avg(i,j) representing the expected channel
%           rate between agents i,j with R.avg == 0 when i == j
%   R.var - NxN matrix with R.var(i,j) representing the variance in the
%           expected channel rate between agents i,j with R.avg == 0 when
%           i == j
% outputs:
%   A - an NKxNNK+1 matrix as described above
%   B - an NKxNNK+1 matrix as described above

K = length(qos); % number of data streams
N = size(R.avg,1); % number of robots

aik_coeffs = cell(K,1);
for k = 1:K
    
  aik_coeffs{k} = zeros(N, N^2);
  for i = 1:N
    
    out_idcs = false(N,N); % outgoing traffic from agent i for flow k
    in_idcs = false(N,N);  % incoming traffic to agent i for flow k
    for j = 1:N
      
      if j == i
        continue
      end
      
      out_idcs(i,j) = 1;
      
      if nnz(j == qos(k).flow.dest) == 0
        in_idcs(j,i) = 1; % incoming traffic for s
      end
      
    end
    
    aik_coeffs{k}(i, out_idcs(:)) = 1;
    aik_coeffs{k}(i, in_idcs(:)) = -1;
    
  end
end

L_mat = diag([repmat(sqrt(R.var(:)), [K 1]); 0]);
R_mat = diag([repmat(R.avg(:), [K 1]); 1]);
Sl = [blkdiag(aik_coeffs{:}) zeros(N*K,1)];
Sr = [blkdiag(aik_coeffs{:}) -ones(N*K,1)];

A = Sl*L_mat;
B = Sr*R_mat;