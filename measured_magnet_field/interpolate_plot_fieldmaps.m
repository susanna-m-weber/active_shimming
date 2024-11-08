clear all;

load('map3_field');
load('map3_pos');

data_use = field_matrix(:,1);


%%interpolating
F = scatteredInterpolant(pos,data_use);
res = 5;
xyi = -50:res:50;
N = numel(xyi);
[X,Y,Z] = meshgrid(xyi);
Binterp = F(X,Y,Z);   %% interpolated fieldmap

%% plot slices
% xslice = [];   
% yslice = [0];
% zslice = [];
xslice = [];   
yslice = [];
zslice = [-50,0,50];
slice(X,Y,Z,Binterp,xslice,yslice,zslice,'nearest');
xlabel('x'); ylabel('y'); zlabel('z');
shading flat;

set(gca, 'yDir','reverse');
set(gca, 'zDir','reverse');
colormap jet;

%set fixed color map
Bmag = field_matrix(:,1);
meanB = mean(Bmag(:));
C1 = (meanB - 0.1);
C2 = (meanB + 0.1);
caxis([C1,C2]);
