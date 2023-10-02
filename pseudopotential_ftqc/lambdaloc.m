function [lam1,lam2,lam3]=lambdaloc(rloc,g1,g2,g3,n)
% This is for computing the value of lambda for local pseudopotentials.
% It gives the lambda in three parts, with the first needing to be multiplied by Z, the next by C_1, and the next by C_2.
% The lattice vectors are given in g1,g2,g3, and n must be a vector [nx,ny,nz].

N1=2^n(1)-1;
N2=2^n(2)-1;
N3=2^n(3)-1;
lam1=0;
lam2=0;
lam3=0;
Omega=(2*pi)^3/det([g1; g2; g3]);
for nx=-N1:N1
    for ny=-N2:N2
        for nz=-N3:N3
            mu=max([abs(nx),abs(ny),abs(nz)]);
            if mu>0
                vec=nx*g1+ny*g2+nz*g3;
                nrm=(vec*vec');
                tmp=exp(-nrm*rloc^2/2);
                lam1=lam1+tmp/nrm;
                lam2=lam2+tmp;
                lam3=lam3+tmp*abs(3-nrm*rloc^2);
            end
        end
    end
end
tmp=sqrt(8*pi^3)*rloc^3/Omega;
lam1=4*pi*lam1/Omega;
lam2=tmp*lam2;
lam3=tmp*lam3;