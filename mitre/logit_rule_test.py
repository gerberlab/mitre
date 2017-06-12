""" Not an actual test suite, just useful objects for debugging.

"""
import mitre.data_processing.synthetic_data as synthetic_data
import mitre.logit_rules as logit_rules
import mitre.rules as rules
import numpy as np

testdata = synthetic_data.linear_test_data(100,3,3,100)
r1 = rules.PrimitiveRule(0,(0,1./2),'average','above',0.)
r2 = rules.PrimitiveRule(1,(0,1./2),'average','above',0.)
rl = rules.RuleList([[r1]])
rl2 = rules.RuleList([[r1,r2]])
#true_classes = rules.generate_y_from_brl(testdata, rl, [1.0,0.0])
probabilities = rules.generate_y_logistic(testdata, rl2, [8.0, -4.0])


testmodel = logit_rules.LogisticRuleModel(testdata,tmin=0.01,tmax=5,max_thresholds=10)

testmodel.hyperparameter_alpha_m = 0.1
testmodel.hyperparameter_alpha_primitives = 1.0
testmodel.hyperparameter_beta_m = 2.0
testmodel.hyperparameter_beta_primitives = 1.0
testmodel.hyperparameter_a_empty = 0.001
testmodel.hyperparameter_b_empty = 0.999


priormodel = logit_rules.LogisticRuleModelPriorOnly(testdata,tmin=0.01,tmax=5,max_thresholds=10)
priormodel.hyperparameter_a_empty = 0.001
priormodel.hyperparameter_b_empty = 0.999

other_r1 = rules.PrimitiveRule(*testmodel.rule_population.flat_rules[1])
other_r2 = rules.PrimitiveRule(*testmodel.rule_population.flat_rules[2])
other_rl = rules.RuleList([[other_r1]])
pair = rules.RuleList([[other_r1, other_r2]])

testsampler = logit_rules.LogisticRuleSampler(testmodel, pair)
priorsampler = logit_rules.LogisticRuleSampler(priormodel, pair)
logit_rules.logger.setLevel(rules.logging.INFO)
#mod_testsampler.sample(1500)
#testsampler.sample(1500)

from mitre import posterior 
#s1 = posterior.PosteriorSummary(testsampler)
#s2 = posterior.PosteriorSummary(mod_testsampler)
#s2.quick_report()

# testmodel = logit_rules.LogisticRuleModelPriorOnly(testdata,tmin=0.01,tmax=5,max_thresholds=10, max_primitives=1)
# testmodel.hyperparameter_a_empty = 0.001
# testmodel.hyperparameter_b_empty = 0.999
# other_r1 = rules.PrimitiveRule(*testmodel.rule_population.flat_rules[1])
# other_r2 = rules.PrimitiveRule(*testmodel.rule_population.flat_rules[2])
# other_rl = rules.RuleList([[other_r1],[other_r2]])
# testsampler = logit_rules.LogisticRuleSampler(testmodel, other_rl)
# logit_rules.logger.setLevel(rules.logging.INFO)
# #testsampler.sample(1000)

# testmodel2 = logit_rules.LogisticRuleModelPriorOnly(testdata,tmin=0.01,tmax=5,max_thresholds=10, max_primitives=1, hyperparameter_alpha_m=0.25, hyperparameter_beta_m=1.0)
# testmodel2.hyperparameter_a_empty = 0.001
# testmodel2.hyperparameter_b_empty = 0.999
# testsampler2 = logit_rules.LogisticRuleSampler(testmodel, other_rl)
# testsampler2.current_state.lambda_m = 1e-3
# testsampler2.sample(1000)
