function plotroutes(fig, x,routes,qos)
% PLOTROUTES visualize probabilistic routing solutions and print useful
% information
%
% inputs:
%   ax     - a set of axes to plot on
%   x      - 2Nx1 vector of [x;y] agent positions stacked
%   routes - NxNxK matrix of routes to display

N = size(routes,1);
K = size(routes,3);

x = reshape(x, [2 N])';

if issparse(routes)
  routes = full(routes);
end

lim = [min(x(:,1)) max(x(:,1)) min(x(:,2)) max(x(:,2))];
window_scale = max(lim([2 4]) - lim([1 3]));
lim = lim + [-1 1 -1 1]*0.1*window_scale;

figure(fig);clf;
for k = 1:K
  
  subplot(K,1,k)
  plot(x(:,1), x(:,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k')
  hold on
  axis equal
  axis(lim)
  cm = colormap('parula');
  colorbar
  
  for i = 1:N
    for j = i+1:N
      
      Pi = x(i,:);
      Pj = x(j,:);
      Aij = routes(i,j,k);
      Aji = routes(j,i,k);
      
      if Pj(1) < Pi(1) || Pj(2) < Pi(2)
        [Pi,Pj] = deal(Pj,Pi);
        [Aij, Aji] = deal(Aji, Aij);
      end
      
      theta = atan2(Pj(2)-Pi(2), Pj(1)-Pi(1));
      beta  = atan2(Pi(2)-Pj(2), Pi(1)-Pj(1));
      
      ds = pi/16;
      offset_scale = 0.03*window_scale;
      l1p1 = Pi + offset_scale*[cos(theta+ds) sin(theta+ds)];
      l2p1 = Pi + offset_scale*[cos(theta-ds) sin(theta-ds)];
      l1p2 = Pj + offset_scale*[cos(beta-ds) sin(beta-ds)];
      l2p2 = Pj + offset_scale*[cos(beta+ds) sin(beta+ds)];
      
      arrowhead_scale = 0.04*window_scale;
      ds = pi/8;
      l1head = l1p1 + arrowhead_scale*[cos(theta+ds) sin(theta+ds)];
      l2head = l2p2 + arrowhead_scale*[cos(beta+ds) sin(beta+ds)];
      
      line_width = 2;
      if Aji > 0.01
        cm_idx = ceil(Aji*size(cm,1));
        plot([l1p1(1) l1p2(1)], [l1p1(2) l1p2(2)],...
          'Color', cm(cm_idx,:), 'LineWidth', line_width)
        plot([l1p1(1) l1head(1)], [l1p1(2) l1head(2)],...
          'Color', cm(cm_idx,:), 'LineWidth', line_width)
      end
      if Aij > 0.01
        cm_idx = ceil(Aij*size(cm,1));
        plot([l2p1(1) l2p2(1)], [l2p1(2) l2p2(2)],...
          'Color', cm(cm_idx,:), 'LineWidth', line_width)
        plot([l2p2(1) l2head(1)], [l2p2(2) l2head(2)],...
          'Color', cm(cm_idx,:), 'LineWidth', line_width)
      end
      
    end
  end
  
  label_offset = 0.04*window_scale;
  text(x(:,1), x(:,2)+label_offset, vec2cellstr(1:N), 'FontSize', 12, 'FontWeight', 'bold')
  title(sprintf('flow %d, src = {%s}, dest = {%s}, margin = %.2f, conf = %.3f',...
    k , num2str(qos(k).flow.src), num2str(qos(k).flow.dest),...
    qos(k).margin, qos(k).confidence));
    
  
  hold off

end

end

function char_vec = vec2cellstr(vec)

char_vec = cell(1,length(vec));
for i = 1:length(vec)
  char_vec{i} = num2str(vec(i));
end

end