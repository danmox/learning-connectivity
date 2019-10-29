function [A,B,zero_vars] = nodemarginconsts(qos,R)
% NODEMARGINCONSTS form node margin constraint matrices of the form:
% norm(diag(A(i,:))*y) = sqrt(sum_i (a_ij_k)^2*R.var(i,j) - sum j(a_ji_k)^2*R.var(i,j))
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
%   A         - an NKxNNK matrix as described above
%   B         - an NKxNNK matrix as described above
%   zero_vars - an NxNxK binary matrix where 1s correspond to routing
%               variables set to zero in the corresponding optimization
%               problem

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

% build mask to keep track of zero routing variables
zero_vars = logical(repmat(eye(N), [1 1 K]));
for k = 1:K

  % destination nodes should not re-broadcast packets
  for j = 1:length(qos(k).flow.dest)
    zero_vars(qos(k).flow.dest(j),:,k) = 1;
  end

end

idx = 1;
for k = 1:K

  for i = 1:N

    Aki = Aseed;
    Bki = Bseed;

    Aki(:,i) = sqrt(R.var(:,i));
    Aki(i,:) = sqrt(R.var(i,:));
    Aki(zero_vars(:,:,k)) = 0;

    if ~any(i == qos(k).flow.dest)
      Bki(:,i) = -R.avg(:,i);
      Bki(i,:) = R.avg(i,:);
      Bki(zero_vars(:,:,k)) = 0;
    else
      Bki(:,i) = R.avg(:,i);  % sense of the constraint is flipped for dests
      Bki(i,:) = -R.avg(i,:); % sense of the constraint is flipped for dests
      Bki(zero_vars(:,:,k)) = 0;
    end
    
    A(idx, (k-1)*N*N+1:k*N*N) = Aki(:)';
    B(idx, (k-1)*N*N+1:k*N*N) = Bki(:)';
    B(idx, end) = -1; % slack
    
    idx = idx + 1;

  end

end
