function plotroutes(ax, x,routes)

N = size(routes,1);
K = size(routes,3);

x = reshape(x, [2 N])';

lim = [min(x(:,1)) max(x(:,1)) min(x(:,2)) max(x(:,2))];
window_scale = max(lim([2 4]) - lim([1 3]));
lim = lim + [-1 1 -1 1]*0.1*window_scale;

axes(ax)
plot(x(:,1), x(:,2), 'ko', 'MarkerSize', 10, 'MarkerFaceColor', 'k')
hold on
axis equal
axis(lim)

cm = colormap('cool');
colorbar

for i = 1:N
  for j = i+1:N
    
    Pi = x(i,:);
    Pj = x(j,:);
    Aij = routes(i,j);
    Aji = routes(j,i);
    
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
    
    line_width = 3;
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

label_offset = 0.03*window_scale;
text(x(:,1)+label_offset, x(:,2)+label_offset, {'1' '2' '3'}, 'FontSize', 12, 'FontWeight', 'bold')

hold off

end