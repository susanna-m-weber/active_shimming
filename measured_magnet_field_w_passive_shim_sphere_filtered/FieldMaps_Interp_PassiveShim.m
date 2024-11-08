clear all;

load('map4_field');
load('map4_pos');
idx = sqrt(pos(:,1).^2+pos(:,2).^2+pos(:,3).^2)<=60;
col_sensors_pos = pos(idx,:);
field_matrix_B0 = field_matrix(idx,:);

load('shimmap1_field');
load('shimmap1_pos');
pos = pos*1e3;
idx = sqrt(pos(:,1).^2+pos(:,2).^2+pos(:,3).^2)<=60;
pos = pos(idx,:);
field_matrix = field_matrix(idx,:)*1e3;
for i = 1:3
    F = scatteredInterpolant(pos,field_matrix(:,i));
    shimfield_matrix(:,i) = F(col_sensors_pos);   %% interpolated fieldmap
end
B = field_matrix_B0(:,1:3)+shimfield_matrix;

writematrix(B, 'B.csv');
writematrix(col_sensors_pos, 'col_sensors_pos.csv');
