%Date: 10/3/11
% Purpose: generate isoperimentric advection matrices for a 2D linear
% triangular elements

function [L] = makeLinLaplace2Dtri(gcoord,nodes)


[nnode,~] = size(gcoord);
[nel,nnel]=size(nodes);
ndof = 1;
edof = ndof*nnel;
xi_coord = [1/6 2/3 1/6];
eta_coord = [1/6 1/6 2/3];
weight2 = [1/6 1/6 1/6];

nzmax = nnode*20; % just a guess
% Dx_vec = zeros(nzmax*3,3);
% Dy_vec = zeros(nzmax*3,3);
% M_vec = zeros(nzmax*3,3);
L_vec = zeros(nzmax*4,3);

num_entries = 0;


for iel = 1:nel
    nd = nodes(iel,:); % get nodes for iel element
    xcoord = zeros(nnel,1); ycoord = zeros(nnel,1);
    index = feeldof(nd,nnel,ndof); % get global DOFs for nodes
    
    
    for i = 1:nnel
        xcoord(i) = gcoord(nd(i),1); ycoord(i)=gcoord(nd(i),2);
    end % get nodal coordinates
    
    
    
    
    for int_p = 1:3
        xi = xi_coord(int_p);
        eta = eta_coord(int_p);
        wt = weight2(int_p);
        [~,dhdr,dhds]=feisot3(xi,eta);
        jacob2 = fejacob2(nnel,dhdr,dhds,xcoord,ycoord);
        detjacob = det(jacob2);
        invjacob = inv(jacob2);
        [dhdx,dhdy]=federiv2(nnel,dhdr,dhds,invjacob);
        for i = 1:edof
            ii = index(i);
            for  j = 1:edof
                jj = index(j);
                num_entries = num_entries+1;
                %                 M_vec(num_entries,1:3) = [ii,jj,shape(i)*shape(j)*wt*detjacob];
                %                 Dx_vec(num_entries,1:3)=[ii,jj,shape(i)*dhdx(j)*wt*detjacob];
                %                 Dy_vec(num_entries,1:3)=[ii,jj,shape(i)*dhdy(j)*wt*detjacob];
                L_vec(num_entries,1:3) = [ii,jj,(dhdx(i)*dhdx(j)+dhdy(i)*dhdy(j))*wt*detjacob];
            end %j
        end % i
    end % int_p
    
end % iel

L_vec = L_vec(1:num_entries,:);
L = sparse(L_vec(:,1),L_vec(:,2),L_vec(:,3),nnode,nnode);

% Dx_vec = Dx_vec(1:num_entries,:);
% Dy_vec = Dy_vec(1:num_entries,:);
% M_vec = M_vec(1:num_entries,:);
% 
% Dx = sparse(Dx_vec(:,1),Dx_vec(:,2),Dx_vec(:,3),nnode,nnode);
% Dy = sparse(Dy_vec(:,1),Dy_vec(:,2),Dy_vec(:,3),nnode,nnode);
% M = sparse(M_vec(:,1),M_vec(:,2),M_vec(:,3),nnode,nnode);
% 
% alpha = sum(diag(M));
% tot_mass = sum(sum(M));
% M = diag(M).*(tot_mass/alpha);  % now diagonal


