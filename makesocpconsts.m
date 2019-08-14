function [A,B] = makesocpconsts(qos,R)

N = size(R.avg,1);
K = length(qos);

if ~isnumeric(R.avg)
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