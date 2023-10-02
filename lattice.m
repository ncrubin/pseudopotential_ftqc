function [g1,g2,g3,d1,d2,d3]=lattice(num)
% This gives the lattice vectors for a range of materials, and d1,d2,d3 which are the delta values that make the state preparation efficient.
% The input is a number corresponding to the strucute as:
% 1 - LiNo2 (C2/m)
% 2 - LiNo2 (P21/c)
% 3 - LiNo2 (P2/c)
% 4 - LiNo2 (R3m)
% 5 - Pd (3x3)
% 6 - Pt (2x2)
% 7 - Pt (3x3)
% 8 - Pt (4x4) 
% 9 - Rh (3x3)
% 10 - LiMnO3
% 11 - Li[LiNiMn]O2
% 12 - LiMnO2F
% 13 - C
% 14 - AlN
% 15 - CaTiO3


if num==1 % LiNo2 (C2/m)
    g1=[1.10349089 ,   0.77586412 ,   0.35561708]/2;
    g2=[0.02753438 ,   1.34866569 ,  -0.35561708]/2;
    g3=[-0.08728916 ,   0.04646969 ,   0.70472651];
    d1=0;
    d2=0;
    d3=0;
elseif num==2 % LiNo2 (P21/c)
    g1=[0.64911019 ,   0.00000000 ,   0.29605690];
    g2=[0.00000000 ,   1.14642763 ,   0.00000000]/2;
    g3=[-0.08516076 ,   0.00000000 ,   0.71179228];
    d1=0;
    d2=0;
    d3=0;
elseif num==3 % LiNo2 (P2/c)
    g1=[0.70228065 ,   0.00000000 ,   0.12733190];
    g2=[0.00000000 ,   0.57745311 ,   0.00000000];
    g3=[0.11747103 ,   0.00000000 ,   0.70959936];
    d1=0;
    d2=0;
    d3=0;
elseif num==4 % LiNo2 (R3m)
    g1=[0.67116441 ,  -1.01900663 ,  -0.62361656];
    g2=[-0.00023507 ,   1.22017873 ,  -0.62361656];
    g3=[-0.00023507 ,  -0.00007048 ,   1.37030425];
    d1=0;
    d2=0;
    d3=0;  % There is no apparent choice of d that gives a good state preparation probability.
elseif num==5 % Pd (3x3)
    g1=[0.39609712 ,  -0.22868678 ,   0.00000000];
    g2=[0.00000000 ,   0.45737356 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.22821558];
    d1=1;
    d2=1;
    d3=0;
elseif num==6 % Pt (2x2)    
    g1=[0.59340589 ,  -0.34260305 ,   0.00000000];
    g2=[0.00000000 ,   0.68520610 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.25236764];
    d1=1;
    d2=1;
    d3=0;
elseif num==7 % Pt (3x3)     
    g1=[0.39560369 ,  -0.22840190 ,   0.00000000];
    g2=[0.00000000 ,   0.45680379 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.25236764];
    d1=1;
    d2=1;
    d3=0;
elseif num==8 % Pt (4x4)    
    g1=[0.29653147 ,   0.17120253 ,   0.00000000];
    g2=[0.00000000 ,   0.34240505 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.25576296];
    d1=0;
    d2=0;
    d3=0;
elseif num==9 % Rh (3x3)
    g1=[0.40775832 ,  -0.23541938 ,   0.00000000];
    g2=[0.00000000 ,   0.47083876 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.23027984];
    d1=1;
    d2=1;
    d3=0;
elseif num==10 % LiMnO3
    g1=[0.33182819 ,   0.00000000 ,   0.11718212];
    g2=[0.00000000 ,   0.19196989 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.69276351];
    d1=1;
    d2=0;
    d3=2;
elseif num==11 % Li[LiNiMn]O2
    g1=[0.58249128 ,   0.33630071 ,   0.00000000];
    g2=[0.00000000 ,   0.44839833 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.16936478];
    d1=2;
    d2=1;
    d3=0;
elseif num==12 % LiMnO2F
    g1=[0.26641975 ,   0.00000000 ,   0.00000000];
    g2=[0.00000000 ,   0.39962962 ,   0.00000000];
    g3=[0.00000000 ,   0.00000000 ,   0.39962962];
    d1=0;
    d2=1;
    d3=1;
elseif num==13 % C
    g1=[-0.93213302 ,   0.93213302 ,   0.93213302]/3;
    g2=[0.93213302 ,  -0.93213302 ,   0.93213302]/3;
    g3=[0.93213302 ,   0.93213302 ,  -0.93213302]/3;
    d1=0;
    d2=0;
    d3=0;
elseif num==14 % AlN
    g1=[1.06910562 ,   0.61724842 ,   0.00000000]/3;
    g2=[0.00000000 ,   1.23449683 ,   0.00000000]/3;
    g3=[0.00000000 ,   0.00000000 ,   0.66765431]/3;
    d1=1;
    d2=1;
    d3=0;
elseif num==15 % CaTiO3
    g1=[0.61806058 ,  -0.00000000 ,  -0.00000000]/3;
    g2=[0.00000000 ,   0.61093999 ,  -0.00000000]/3;
    g3=[0.00000000 ,   0.00000000 ,   0.43519309]/3;
    d1=1;
    d2=1;
    d3=0;
end