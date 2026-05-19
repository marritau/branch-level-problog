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

0.89407885::z(b0,0).
0.89407885::z(b1,0).
0.86747527::z(b2,0).
0.86747527::z(b3,0).
0.86019284::z(b4,0).
0.68832350::z(b5,0).
0.78875053::z(b6,0).
0.78875053::z(b7,0).
0.64953256::z(b8,0).
0.76282960::z(b9,0).
0.76282960::z(b10,0).
0.62968701::z(b11,0).
0.73844665::z(b12,0).
0.73844665::z(b13,0).
0.73844665::z(b14,0).
0.73844665::z(b15,0).
0.73844665::z(b16,0).
0.70685607::z(b0,1).
0.70685607::z(b1,1).
0.59304029::z(b2,1).
0.59304029::z(b3,1).
0.53401142::z(b4,1).
0.36686826::z(b5,1).
0.48534685::z(b6,1).
0.48534685::z(b7,1).
0.38197318::z(b8,1).
0.51340127::z(b9,1).
0.51340127::z(b10,1).
0.39794761::z(b11,1).
0.35324875::z(b12,1).
0.35324875::z(b13,1).
0.35324875::z(b14,1).
0.35324875::z(b15,1).
0.35324875::z(b16,1).

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
0.05747503::supports(b0,c1,X) :- z(b0,X).
0.00954370::supports(b0,c2,X) :- z(b0,X).
0.31409946::supports(b1,c0,X) :- z(b1,X).
0.00953681::supports(b1,c1,X) :- z(b1,X).
0.00000093::supports(b1,c2,X) :- z(b1,X).
0.01911259::supports(b2,c0,X) :- z(b2,X).
0.22374074::supports(b2,c1,X) :- z(b2,X).
0.30400190::supports(b2,c2,X) :- z(b2,X).
0.00000093::supports(b3,c0,X) :- z(b3,X).
0.22374074::supports(b3,c1,X) :- z(b3,X).
0.30400190::supports(b3,c2,X) :- z(b3,X).
0.00000093::supports(b4,c0,X) :- z(b4,X).
0.22374696::supports(b4,c1,X) :- z(b4,X).
0.12543444::supports(b4,c2,X) :- z(b4,X).
0.33407333::supports(b5,c0,X) :- z(b5,X).
0.27364916::supports(b5,c1,X) :- z(b5,X).
0.02869426::supports(b5,c2,X) :- z(b5,X).
0.00000093::supports(b6,c0,X) :- z(b6,X).
0.27359319::supports(b6,c1,X) :- z(b6,X).
0.02868767::supports(b6,c2,X) :- z(b6,X).
0.00000093::supports(b7,c0,X) :- z(b7,X).
0.21380314::supports(b7,c1,X) :- z(b7,X).
0.01910804::supports(b7,c2,X) :- z(b7,X).
0.00000092::supports(b8,c0,X) :- z(b8,X).
0.01909749::supports(b8,c1,X) :- z(b8,X).
0.28397959::supports(b8,c2,X) :- z(b8,X).
0.00000093::supports(b9,c0,X) :- z(b9,X).
0.01909101::supports(b9,c1,X) :- z(b9,X).
0.00954380::supports(b9,c2,X) :- z(b9,X).
0.00000093::supports(b10,c0,X) :- z(b10,X).
0.00953708::supports(b10,c1,X) :- z(b10,X).
0.00954380::supports(b10,c2,X) :- z(b10,X).
0.33406556::supports(b11,c0,X) :- z(b11,X).
0.29372230::supports(b11,c1,X) :- z(b11,X).
0.31422001::supports(b11,c2,X) :- z(b11,X).
0.33408484::supports(b12,c0,X) :- z(b12,X).
0.24367397::supports(b12,c1,X) :- z(b12,X).
0.00000093::supports(b12,c2,X) :- z(b12,X).
0.00954311::supports(b13,c0,X) :- z(b13,X).
0.24367397::supports(b13,c1,X) :- z(b13,X).
0.00000093::supports(b13,c2,X) :- z(b13,X).
0.00954311::supports(b14,c0,X) :- z(b14,X).
0.00954078::supports(b14,c1,X) :- z(b14,X).
0.00000093::supports(b14,c2,X) :- z(b14,X).
0.00000092::supports(b15,c0,X) :- z(b15,X).
0.02867283::supports(b15,c1,X) :- z(b15,X).
0.02869339::supports(b15,c2,X) :- z(b15,X).
0.00000092::supports(b16,c0,X) :- z(b16,X).
0.02867283::supports(b16,c1,X) :- z(b16,X).
0.00954742::supports(b16,c2,X) :- z(b16,X).

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