% Auto-generated ProbLog rules from BranchNet latent branches

threshold(t0_0,5.521128368).
threshold(t0_1,2.69445031).
threshold(t0_2,3.235752501).
threshold(t0_6,5.533355168).
threshold(t1_0,1.55049555).
threshold(t1_1,2.592328529).
threshold(t1_2,1.631602043).
threshold(t1_4,2.421156362).
threshold(t1_5,2.963456616).
threshold(t2_0,1.546526634).
threshold(t2_1,4.607131704).
threshold(t2_3,0.5012225273).
threshold(t2_4,5.085041249).
threshold(t2_6,3.02946625).

branch_struct(b0, X) :- le(b0,f0,t0_0,X), le(b0,f1,t0_1,X).
branch_struct(b1, X) :- le(b1,f0,t0_0,X), gt(b1,f1,t0_1,X).
branch_struct(b2, X) :- gt(b2,f0,t0_0,X).
branch_struct(b3, X) :- gt(b3,f0,t0_0,X), gt(b3,f2,t0_2,X).
branch_struct(b4, X) :- gt(b4,f0,t0_0,X), gt(b4,f2,t0_2,X), le(b4,f2,t0_6,X).
branch_struct(b5, X) :- le(b5,f3,t1_0,X).
branch_struct(b6, X) :- le(b6,f3,t1_0,X), gt(b6,f2,t1_1,X).
branch_struct(b7, X) :- le(b7,f3,t1_0,X), gt(b7,f2,t1_1,X), gt(b7,f1,t1_4,X).
branch_struct(b8, X) :- gt(b8,f3,t1_0,X).
branch_struct(b9, X) :- gt(b9,f3,t1_0,X), le(b9,f3,t1_2,X).
branch_struct(b10, X) :- gt(b10,f3,t1_0,X), le(b10,f3,t1_2,X), gt(b10,f1,t1_5,X).
branch_struct(b11, X).
branch_struct(b12, X) :- le(b12,f3,t2_0,X), le(b12,f2,t2_1,X).
branch_struct(b13, X) :- le(b13,f3,t2_0,X), le(b13,f2,t2_1,X), gt(b13,f3,t2_3,X).
branch_struct(b14, X) :- le(b14,f3,t2_0,X), le(b14,f2,t2_1,X), gt(b14,f3,t2_3,X), le(b14,f2,t2_6,X).
branch_struct(b15, X) :- le(b15,f3,t2_0,X), gt(b15,f2,t2_1,X).
branch_struct(b16, X) :- le(b16,f3,t2_0,X), gt(b16,f2,t2_1,X), le(b16,f2,t2_4,X).

0.89414304::z(b0,0).
0.89414304::z(b1,0).
0.86787426::z(b2,0).
0.86787426::z(b3,0).
0.86059356::z(b4,0).
0.68850505::z(b5,0).
0.78863090::z(b6,0).
0.78863090::z(b7,0).
0.64939439::z(b8,0).
0.76244634::z(b9,0).
0.76244634::z(b10,0).
0.62956524::z(b11,0).
0.73913014::z(b12,0).
0.73913014::z(b13,0).
0.73913014::z(b14,0).
0.73913014::z(b15,0).
0.73913014::z(b16,0).
0.70706677::z(b0,1).
0.70706677::z(b1,1).
0.59285980::z(b2,1).
0.59285980::z(b3,1).
0.53384334::z(b4,1).
0.36666337::z(b5,1).
0.48552975::z(b6,1).
0.48552975::z(b7,1).
0.38188437::z(b8,1).
0.51364368::z(b9,1).
0.51364368::z(b10,1).
0.39787003::z(b11,1).
0.35286492::z(b12,1).
0.35286492::z(b13,1).
0.35286492::z(b14,1).
0.35286492::z(b15,1).
0.35286492::z(b16,1).

not_z(b0,X) :- \+ z(b0,X).
not_z(b1,X) :- \+ z(b1,X).
not_z(b2,X) :- \+ z(b2,X).
not_z(b3,X) :- \+ z(b3,X).
not_z(b4,X) :- \+ z(b4,X).
not_z(b5,X) :- \+ z(b5,X).
not_z(b6,X) :- \+ z(b6,X).
not_z(b7,X) :- \+ z(b7,X).
not_z(b8,X) :- \+ z(b8,X).
not_z(b9,X) :- \+ z(b9,X).
not_z(b10,X) :- \+ z(b10,X).
not_z(b11,X) :- \+ z(b11,X).
not_z(b12,X) :- \+ z(b12,X).
not_z(b13,X) :- \+ z(b13,X).
not_z(b14,X) :- \+ z(b14,X).
not_z(b15,X) :- \+ z(b15,X).
not_z(b16,X) :- \+ z(b16,X).

0.95000000::le(b0,f0,t0_0,X) :- z(b0,X).
0.05000000::le(b0,f0,t0_0,X) :- not_z(b0,X).
0.95000000::le(b0,f1,t0_1,X) :- z(b0,X).
0.05000000::le(b0,f1,t0_1,X) :- not_z(b0,X).
0.95000000::le(b1,f0,t0_0,X) :- z(b1,X).
0.05000000::le(b1,f0,t0_0,X) :- not_z(b1,X).
0.95000000::gt(b1,f1,t0_1,X) :- z(b1,X).
0.05000000::gt(b1,f1,t0_1,X) :- not_z(b1,X).
0.95000000::gt(b2,f0,t0_0,X) :- z(b2,X).
0.05000000::gt(b2,f0,t0_0,X) :- not_z(b2,X).
0.95000000::gt(b3,f0,t0_0,X) :- z(b3,X).
0.05000000::gt(b3,f0,t0_0,X) :- not_z(b3,X).
0.95000000::gt(b3,f2,t0_2,X) :- z(b3,X).
0.05000000::gt(b3,f2,t0_2,X) :- not_z(b3,X).
0.95000000::gt(b4,f0,t0_0,X) :- z(b4,X).
0.05000000::gt(b4,f0,t0_0,X) :- not_z(b4,X).
0.95000000::gt(b4,f2,t0_2,X) :- z(b4,X).
0.05000000::gt(b4,f2,t0_2,X) :- not_z(b4,X).
0.95000000::le(b4,f2,t0_6,X) :- z(b4,X).
0.05000000::le(b4,f2,t0_6,X) :- not_z(b4,X).
0.95000000::le(b5,f3,t1_0,X) :- z(b5,X).
0.05000000::le(b5,f3,t1_0,X) :- not_z(b5,X).
0.95000000::le(b6,f3,t1_0,X) :- z(b6,X).
0.05000000::le(b6,f3,t1_0,X) :- not_z(b6,X).
0.95000000::gt(b6,f2,t1_1,X) :- z(b6,X).
0.05000000::gt(b6,f2,t1_1,X) :- not_z(b6,X).
0.95000000::le(b7,f3,t1_0,X) :- z(b7,X).
0.05000000::le(b7,f3,t1_0,X) :- not_z(b7,X).
0.95000000::gt(b7,f2,t1_1,X) :- z(b7,X).
0.05000000::gt(b7,f2,t1_1,X) :- not_z(b7,X).
0.95000000::gt(b7,f1,t1_4,X) :- z(b7,X).
0.05000000::gt(b7,f1,t1_4,X) :- not_z(b7,X).
0.95000000::gt(b8,f3,t1_0,X) :- z(b8,X).
0.05000000::gt(b8,f3,t1_0,X) :- not_z(b8,X).
0.95000000::gt(b9,f3,t1_0,X) :- z(b9,X).
0.05000000::gt(b9,f3,t1_0,X) :- not_z(b9,X).
0.95000000::le(b9,f3,t1_2,X) :- z(b9,X).
0.05000000::le(b9,f3,t1_2,X) :- not_z(b9,X).
0.95000000::gt(b10,f3,t1_0,X) :- z(b10,X).
0.05000000::gt(b10,f3,t1_0,X) :- not_z(b10,X).
0.95000000::le(b10,f3,t1_2,X) :- z(b10,X).
0.05000000::le(b10,f3,t1_2,X) :- not_z(b10,X).
0.95000000::gt(b10,f1,t1_5,X) :- z(b10,X).
0.05000000::gt(b10,f1,t1_5,X) :- not_z(b10,X).
0.95000000::le(b12,f3,t2_0,X) :- z(b12,X).
0.05000000::le(b12,f3,t2_0,X) :- not_z(b12,X).
0.95000000::le(b12,f2,t2_1,X) :- z(b12,X).
0.05000000::le(b12,f2,t2_1,X) :- not_z(b12,X).
0.95000000::le(b13,f3,t2_0,X) :- z(b13,X).
0.05000000::le(b13,f3,t2_0,X) :- not_z(b13,X).
0.95000000::le(b13,f2,t2_1,X) :- z(b13,X).
0.05000000::le(b13,f2,t2_1,X) :- not_z(b13,X).
0.95000000::gt(b13,f3,t2_3,X) :- z(b13,X).
0.05000000::gt(b13,f3,t2_3,X) :- not_z(b13,X).
0.95000000::le(b14,f3,t2_0,X) :- z(b14,X).
0.05000000::le(b14,f3,t2_0,X) :- not_z(b14,X).
0.95000000::le(b14,f2,t2_1,X) :- z(b14,X).
0.05000000::le(b14,f2,t2_1,X) :- not_z(b14,X).
0.95000000::gt(b14,f3,t2_3,X) :- z(b14,X).
0.05000000::gt(b14,f3,t2_3,X) :- not_z(b14,X).
0.95000000::le(b14,f2,t2_6,X) :- z(b14,X).
0.05000000::le(b14,f2,t2_6,X) :- not_z(b14,X).
0.95000000::le(b15,f3,t2_0,X) :- z(b15,X).
0.05000000::le(b15,f3,t2_0,X) :- not_z(b15,X).
0.95000000::gt(b15,f2,t2_1,X) :- z(b15,X).
0.05000000::gt(b15,f2,t2_1,X) :- not_z(b15,X).
0.95000000::le(b16,f3,t2_0,X) :- z(b16,X).
0.05000000::le(b16,f3,t2_0,X) :- not_z(b16,X).
0.95000000::gt(b16,f2,t2_1,X) :- z(b16,X).
0.05000000::gt(b16,f2,t2_1,X) :- not_z(b16,X).
0.95000000::le(b16,f2,t2_4,X) :- z(b16,X).
0.05000000::le(b16,f2,t2_4,X) :- not_z(b16,X).

% Branch-to-class support rules initialized from BranchNet class proportions
0.00000093::supports(b0,c0,X) :- z(b0,X).
0.05754095::supports(b0,c1,X) :- z(b0,X).
0.00961036::supports(b0,c2,X) :- z(b0,X).
0.31458867::supports(b1,c0,X) :- z(b1,X).
0.00954832::supports(b1,c1,X) :- z(b1,X).
0.00000094::supports(b1,c2,X) :- z(b1,X).
0.01916818::supports(b2,c0,X) :- z(b2,X).
0.22402018::supports(b2,c1,X) :- z(b2,X).
0.30605072::supports(b2,c2,X) :- z(b2,X).
0.00000093::supports(b3,c0,X) :- z(b3,X).
0.22402018::supports(b3,c1,X) :- z(b3,X).
0.30605072::supports(b3,c2,X) :- z(b3,X).
0.00000093::supports(b4,c0,X) :- z(b4,X).
0.22403142::supports(b4,c1,X) :- z(b4,X).
0.12645316::supports(b4,c2,X) :- z(b4,X).
0.33450586::supports(b5,c0,X) :- z(b5,X).
0.27398250::supports(b5,c1,X) :- z(b5,X).
0.02895978::supports(b5,c2,X) :- z(b5,X).
0.00000093::supports(b6,c0,X) :- z(b6,X).
0.27384576::supports(b6,c1,X) :- z(b6,X).
0.02893160::supports(b6,c2,X) :- z(b6,X).
0.00000093::supports(b7,c0,X) :- z(b7,X).
0.21401797::supports(b7,c1,X) :- z(b7,X).
0.01927137::supports(b7,c2,X) :- z(b7,X).
0.00000093::supports(b8,c0,X) :- z(b8,X).
0.01912759::supports(b8,c1,X) :- z(b8,X).
0.29026952::supports(b8,c2,X) :- z(b8,X).
0.00000093::supports(b9,c0,X) :- z(b9,X).
0.01911208::supports(b9,c1,X) :- z(b9,X).
0.00961756::supports(b9,c2,X) :- z(b9,X).
0.00000093::supports(b10,c0,X) :- z(b10,X).
0.00954770::supports(b10,c1,X) :- z(b10,X).
0.00961756::supports(b10,c2,X) :- z(b10,X).
0.33449781::supports(b11,c0,X) :- z(b11,X).
0.29406026::supports(b11,c1,X) :- z(b11,X).
0.32092342::supports(b11,c2,X) :- z(b11,X).
0.33451867::supports(b12,c0,X) :- z(b12,X).
0.24399444::supports(b12,c1,X) :- z(b12,X).
0.00000094::supports(b12,c2,X) :- z(b12,X).
0.00957247::supports(b13,c0,X) :- z(b13,X).
0.24399444::supports(b13,c1,X) :- z(b13,X).
0.00000094::supports(b13,c2,X) :- z(b13,X).
0.00957247::supports(b14,c0,X) :- z(b14,X).
0.00955686::supports(b14,c1,X) :- z(b14,X).
0.00000094::supports(b14,c2,X) :- z(b14,X).
0.00000093::supports(b15,c0,X) :- z(b15,X).
0.02872035::supports(b15,c1,X) :- z(b15,X).
0.02896057::supports(b15,c2,X) :- z(b15,X).
0.00000093::supports(b16,c0,X) :- z(b16,X).
0.02872035::supports(b16,c1,X) :- z(b16,X).
0.00963733::supports(b16,c2,X) :- z(b16,X).

% Class predicates aggregate support from all active branches
class(X,c0) :- supports(B,c0,X).
class(X,c1) :- supports(B,c1,X).
class(X,c2) :- supports(B,c2,X).

% Class queries for exported objects
query(class(0,c0)).
query(class(0,c1)).
query(class(0,c2)).
query(class(1,c0)).
query(class(1,c1)).
query(class(1,c2)).