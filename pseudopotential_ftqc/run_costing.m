% cst=allcosts([1, 2, 3],20,["Li", "Mn", "Ni", "O"],11,1,256, 468)
% [laml, lamnl, lamT, lamV]=alllam([1, 2, 3],["Pd"],[27],5,270);
% fprintf('%.13f\n', laml);
% fprintf('%.13f\n', lamT);
% fprintf('%.13f\n', lamV);
% fprintf('%.13f\n', lamnl);
% 
% 
% [laml, lamnl, lamT, lamV]=alllam([4, 3, 2],["Pd"],[27],5,270);
% fprintf('%.13f\n', laml);
% fprintf('%.13f\n', lamT);
% fprintf('%.13f\n', lamV);
% fprintf('%.13f\n', lamnl);


[laml, lamnl, lamT, lamV]=alllam([4, 4, 4],["Pt"],[27],7,270);
fprintf('%.13f\n', laml);
fprintf('%.13f\n', lamT);
fprintf('%.13f\n', lamV);
fprintf('%.13f\n', lamnl);

% [laml, lamnl, lamT, lamV] = alllam([3, 4, 3],["Li", "Mn", "Ni", "O"],[22, 14, 6, 47], 11, 468);
% fprintf('%.13f\n', laml);
% fprintf('%.13f\n', lamT);
% fprintf('%.13f\n', lamV);
% fprintf('%.13f\n', lamnl);


% [laml, lamnl, lamT, lamV] = alllam([2, 4, 3],["Li", "Mn", "O"],[8, 16, 48], 10, 408);
% fprintf('%.13f\n', laml);
% fprintf('%.13f\n', lamT);
% fprintf('%.13f\n', lamV);
% fprintf('%.13f\n', lamnl);

% [laml, lamnl, lamT, lamV] = alllam([2, 1, 3],["Li", "Mn", "F", "O"],[12, 16, 16, 32], 10, 428);
% fprintf('%.13f\n', laml);
% fprintf('%.13f\n', lamT);
% fprintf('%.13f\n', lamV);
% fprintf('%.13f\n', lamnl);

% [laml, lamnl, lamT, lamV]=alllam([2, 4, 3],["Rh"],[27],9,243);
% fprintf('%.13f\n', laml);
% fprintf('%.13f\n', lamT);
% fprintf('%.13f\n', lamV);
% fprintf('%.13f\n', lamnl);