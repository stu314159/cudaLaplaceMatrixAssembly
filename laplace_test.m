% laplace_test.m

clear
clc
close('all')

mesh_file_name = 'ex_5_9_2.msh';
fprintf('reading mesh file...\n');
msh = load_gmsh4(mesh_file_name,[1 2 15]);
gcoord_p = msh.POS(:,1:2);
nodes = msh.TRIANGLES(:,1:3);

[nel,~]=size(nodes);
[nnodes,~]=size(gcoord_p);
fprintf('Number of degrees of freedom = %d.\n',nnodes);
fprintf('Number of elements = %d.\n',nel);


fprintf('Constructing matrix with m-language routine...\n');
tic

L_tst = makeLinLaplace2Dtri(gcoord_p,nodes);

built_in_time =toc;

fprintf('Constructing matrix with cuda routine...\n');

tic
L = makeLinLaplace2Dtri_cusp(gcoord_p,nodes);
cusp_time = toc;

fprintf('Absolute error in sparse matrices = %g. \n',norm(L_tst - L,Inf));

fprintf('Time for m-language routine = %g. \n',built_in_time);
fprintf('Time for cuda routine = %g. \n',cusp_time);

fprintf('speedup = %g.\n',built_in_time/cusp_time);