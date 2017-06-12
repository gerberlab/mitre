""" Not an actual test suite, just useful objects for debugging.

Usage: import rules, %run -i test.py

"""

testdata_323 = rules.linear_test_data(30,2,3,100)
r1 = rules.PrimitiveRule(0,(0,1./3),'average','above',0.)
rl = rules.RuleList([[r1]])
true_classes = rules.generate_y_from_brl(testdata_323, rl, [0, 1])
testmodel = rules.DiscreteBRLModel(testdata_323, 0.2, 0.2, 0.2, N_intervals=3, alpha=1, beta=1)

testsampler = rules.DiscreteBRLSampler(testmodel, rl)

rl2 = rules.RuleList([[testmodel.rule_population.flat_rules[0]]])

r2 = rules.PrimitiveRule(1,(0,1./3),'average','above',0.)
r3 = rules.PrimitiveRule(1,(1./3,2./3),'slope','above',0.)

rl3 = rules.RuleList([[r1,r2]])
#rl3 = rules.RuleList([[r1,r2],[r3]])
#rl3 = rules.RuleList([[r1]])
testdata_complex = rules.linear_test_data(60,2,3,100)
true_classes_complex = rules.generate_y_from_brl(testdata_complex, rl3, [1, 0, 0])
testmodel_complex = rules.DiscreteBRLModel(testdata_complex, 0.2, 0.2, 0.2, N_intervals=3, alpha=1, beta=1)
complexsampler = rules.DiscreteBRLSampler(testmodel_complex, rl2)
