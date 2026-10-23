import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl

qualidade = ctrl.Antecedent(np.arange(0, 11, 1), 'qualidade')
servico = ctrl.Antecedent(np.arange(0, 11, 1), 'servico')
gorjeta = ctrl.Consequent(np.arange(0, 26, 1), 'gorjeta')

qualidade['ruim'] = fuzz.trimf(qualidade.universe, [0, 0, 5])
qualidade['boa'] = fuzz.trimf(qualidade.universe, [0, 5, 10])
qualidade['saborosa'] = fuzz.trimf(qualidade.universe, [5, 10, 10])

servico['ruim'] = fuzz.trimf(servico.universe, [0, 0, 5])
servico['medio'] = fuzz.trimf(servico.universe, [0, 5, 10])
servico['otimo'] = fuzz.trimf(servico.universe, [5, 10, 10])

gorjeta['baixa'] = fuzz.trimf(gorjeta.universe, [0, 0, 13])
gorjeta['media'] = fuzz.trimf(gorjeta.universe, [0, 13, 25])
gorjeta['alta'] = fuzz.trimf(gorjeta.universe, [13, 25, 25])

regra1 = ctrl.Rule(qualidade['ruim'] | servico['ruim'], gorjeta['baixa'])
regra2 = ctrl.Rule(servico['medio'], gorjeta['media'])
regra3 = ctrl.Rule(servico['otimo'] & qualidade['saborosa'], gorjeta['alta'])

sistema = ctrl.ControlSystem([regra1, regra2, regra3])
sim = ctrl.ControlSystemSimulation(sistema)
sim.input['qualidade'] = 6.5
sim.input['servico'] = 9.8
sim.compute()
print(sim.output['gorjeta'])  # ~19.8
