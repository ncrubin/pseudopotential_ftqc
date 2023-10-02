function [laml, lamnl, lamT, lamV]=alllam(n,type,nonu,lat,eta)
% This is for giving the overall lambda value.
% n - the vector for the size of the lattice [nx,ny,nz]
% "type" - the list of nucleus types, e.g. ["Al","Ti"]
% nonu - a vector of the number of nuclei of each type
% lat - the lattice type
% eta - number of electrons


[g1,g2,g3,d1,d2,d3]=lattice(lat);

laml=0;
lamnl=0;
noty=length(type);
for no=1:noty
    % Get the pseudopotentiual parameters.
    [Z,rl,C,r,E]=parameters(type(no));
    % Compute parts of lambda for given rl.
    [lam1,lam2,lam3]=lambdaloc(rl,g1,g2,g3,n);
    % Add together based on Z and C values. We need to multiply by the
    % number of this type of nucleus, nonu(no).
    laml=laml+(Z*lam1+abs(C(1))*lam2+abs(C(2))*lam3)*nonu(no);

    % Now for nonlocal lambda.
    [lam0,  lamm]=lamnonloc(r,E,g1,g2,g3,d1,d2,d3,n);
    
    % The second output is for the case where we are accounting for maxima
    % over boxes.
    lamnl=lamnl+sum(sum(sum(lamm)))*nonu(no);
end
% Multiply by number of electrons too.
laml=laml*eta;
lamnl=lamnl*eta;

% Next compute lambda_T.
v1=(2^n(1)-2)*g1;
v2=(2^n(2)-2)*g2;
v3=(2^n(3)-2)*g3;
lamT=(eta/8)*max([norm(v1+v2+v3), norm(v1+v2-v3), norm(v1-v2+v3), norm(v1-v2-v3)]);

N1=2^n(1)-1;
N2=2^n(2)-1;
N3=2^n(3)-1;
vsum=0;
for nx=-N1:N1
    for ny=-N2:N2
        for nz=-N3:N3
            tmp=norm(nx*g1+ny*g2+nz*g3);
            if tmp>0
                vsum=vsum+1/norm(nx*g1 + ny*g2 + nz*g3);
            end
        end
    end
end
Omega=(2*pi)^3/det([g1; g2; g3]);
lamV=(2*pi/Omega)*eta*(eta-1)*vsum;