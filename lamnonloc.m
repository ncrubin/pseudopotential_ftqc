function [lambda,lamb]=lamnonloc(rl,E,g1,g2,g3,d1,d2,d3,n)
% This is for computing the value of lambda for nonlocal pseudopotentials.
% The lambda is the value in the tight case, but lamb is if we are using maxima based on the box for nu.

N1=2^n(1)-1;
N2=2^n(2)-1;
N3=2^n(3)-1;
sz=size(E);
if length(sz)<3
    sz=[sz,sz(2)];
end

% Determine scaled E.
Esc=zeros(3,3,3);
% Input GTH nonlocal pseudopotential projector parameters.
Cli=[4*sqrt(2)  8*sqrt(2/15)   (16/3)*sqrt(2/105)
8/sqrt(3)  16/sqrt(105)   (32/3)/sqrt(1155)
8*sqrt(2/15)  (16/3)*sqrt(2/105)  (32/3)*sqrt(2/15015)]*pi^(5/4);
% Now multiply E by Cli.
for el=1:sz(1)
    for ii=1:sz(2)
        for jj=1:sz(3)
            Esc(el,ii,jj)=E(el,ii,jj)*Cli(el,ii)*Cli(el,jj)*rl(el)^(1+2*el);
        end
    end
end

Omega=(2*pi)^3/det([g1; g2; g3]);

% First compute a full matrix of Fli values.
Fli=zeros(sz(1),sz(2),2*N1+1,2*N2+1,2*N3+1);
for nx=-N1:N1
    for ny=-N2:N2
        for nz=-N3:N3
            vec=nx*g1+ny*g2+nz*g3;
            nrm=(vec*vec');
            % l=0
            tmp1=nrm*rl(1)^2;
            tmp2=exp(-tmp1/2);
            tmp=[1   (3-tmp1)    (15-10*tmp1+tmp1^2)]*tmp2;
            for jj=1:sz(2)
                Fli(1,jj,nx+N1+1,ny+N2+1,nz+N3+1)=tmp(jj);
            end
            % l=1
            if sz(1)>1
                tmp1=nrm*rl(2)^2;
                tmp2=exp(-tmp1/2);
                tmp=[1   (5-tmp1)    (35-14*tmp1+tmp1^2)]*tmp2;
                for jj=1:sz(2)
                    Fli(2,jj,nx+N1+1,ny+N2+1,nz+N3+1)=tmp(jj);
                end
            end
            % l=2
            if sz(1)>2
                tmp1=nrm*rl(3)^2;
                tmp2=exp(-tmp1/2);
                tmp=[1   (7-tmp1)    (63-18*tmp1+tmp1^2)]*tmp2;
                for jj=1:sz(2)
                    Fli(3,jj,nx+N1+1,ny+N2+1,nz+N3+1)=tmp(jj);
                end
            end
        end
    end
end

maxs=zeros(4*N1+1,sz(1),sz(2),sz(3),2*N2+1,4*N3+1);

% We loop over all nu values, but only use non-negative nu_y because it must be symmetric under reflection about all three.
for nut=1:4*N1+1
    nux=nut-2*N1-1;
    maxt=zeros(sz(1),sz(2),sz(3),2*N2+1,4*N3+1);
    for nuy=0:2*N2
        for nuz=-2*N3:2*N3
            % Now find appropriate range of p vector given nu.
            for mx=max(-N1,-nux-N1):min(N1,-nux+N1)
                for my=max(-N2,-nuy-N2):min(N2,-nuy+N2)
                    for mz=max(-N3,-nuz-N3):min(N3,-nuz+N3)
                        nx=nux+mx;
                        ny=nuy+my;
                        nz=nuz+mz;
                        vecp=nx*g1+ny*g2+nz*g3;
                        vecq=mx*g1+my*g2+mz*g3;
                        dot=vecp*vecq';

                        Pl=[1  dot  (3*dot^2-(vecp*vecp')*(vecq*vecq'))/2];
                        for el=1:sz(1)
                            for ii=1:sz(2)
                                for jj=1:sz(3)
                                    if Esc(el,ii,jj)~=0
                                        tmp=abs((2*el-1)/(4*pi*Omega)*Pl(el)*Esc(el,ii,jj)*Fli(el,ii,nx+N1+1,ny+N2+1,nz+N3+1)*Fli(el,jj,mx+N1+1,my+N2+1,mz+N3+1));
                                        if tmp>maxt(el,ii,jj,nuy+1,nuz+2*N3+1)
                                            maxt(el,ii,jj,nuy+1,nuz+2*N3+1)=tmp;
                                        end
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    maxs(nut,:,:,:,:,:)=maxt(:,:,:,:,:);
end
lambda=zeros(sz(1),sz(2),sz(3));
mumax=max([n(1)+d1,n(2)+d2,n(3)+d3])+2;
maxy=zeros(sz(1),sz(2),sz(3),mumax);
nums=zeros(mumax,1);
D1=2^d1;
D2=2^d2;
D3=2^d3;
for nux=-2*N1:2*N1
    for nuy=-2*N2:2*N2
        for nuz=-2*N3:2*N3
            mut=max([abs(nux*D1),abs(nuy*D2),abs(nuz*D3)]);
            if mut==0
                mu=1;
            else
                mu=floor(log2(mut))+2;
            end
            nums(mu)=nums(mu)+1;
            for el=1:sz(1)
                for ii=1:sz(2)
                    for jj=1:sz(3)
                        if Esc(el,ii,jj)~=0
                            tmp=maxs(nux+2*N1+1,el,ii,jj,abs(nuy)+1,nuz+2*N3+1);
                            lambda(el,ii,jj)=lambda(el,ii,jj)+tmp;
                            if tmp>maxy(el,ii,jj,mu)
                                maxy(el,ii,jj,mu)=tmp;
                            end
                        end
                    end
                end
            end
        end
    end
end
for el=1:sz(1)
    for ii=1:sz(2)
        for jj=ii+1:sz(3)
            lambda(el,ii,jj)=lambda(el,jj,ii);
            maxy(el,ii,jj,:)=maxy(el,jj,ii,:);
        end
    end
end
for el=1:sz(1)
    for ii=1:sz(2)
        for jj=1:sz(3)
            lamb(el,jj,ii)=sum(nums.*squeeze(maxy(el,ii,jj,:)));
        end
    end
end