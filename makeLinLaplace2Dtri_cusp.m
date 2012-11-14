function L = makeLinLaplace2Dtri_cusp(gcoord,nodes)

[nnodes,~]=size(gcoord);
[nel,~]=size(nodes);

% write gcoord and nodes data to a file
tic
% params.laplace2d - give number of nodes and number of elements
params = fopen('params.laplace2d','w');
fprintf(params,'%d \n',nnodes);
fprintf(params,'%d \n',nel);
fclose(params);

%gcoord.laplace2d - give nodal coordinates
save('gcoord.laplace2d','gcoord','-ascii');

node_file = fopen('nodes.laplace2d','w');
for i = 1:nel
   for nd = 1:3
       fprintf(node_file,'%d  ',nodes(i,nd));
   end
   fprintf(node_file,'\n'); % after each element, carriage return line feed.
    
end
write_time = toc; 
fprintf('time to write the data to disk = %g. \n',write_time);

% invoke a function to generate the sparse entries
%tic
system('./make_LaplaceT3r4');
%program_time = toc;
%fprintf('Actual time executing the program = %g.\n',program_time);
% read that data

%tic
[L,rows,cols,entries,~,~,~]=mmread('L_cuda.mtx');
%read_time = toc;

%fprintf('Time to read the matrix-market file = %g. \n',read_time);

%fprintf('Returned matrix has %d rows, %d columns and %d entries.\n',rows,cols,entries);

system('rm *.laplace2d *.mtx');