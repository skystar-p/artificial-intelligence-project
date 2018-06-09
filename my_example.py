from pomegranate import *


s1 = State(DiscreteDistribution({ 'a': 0.1, 'b': 0.9 }), name='s1')
s2 = State(DiscreteDistribution({ 'a': 0.9, 'b': 0.1 }), name='s2')

model = HiddenMarkovModel()

model.add_states(s1, s2)
model.add_transition(model.start, s1, 0.5)
model.add_transition(model.start, s2, 0.5)
model.add_transition(s1, s1, 0.5)
model.add_transition(s1, s2, 0.5)
model.add_transition(s2, s1, 0.5)
model.add_transition(s2, s2, 0.5)
model.bake()

p = model.predict(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
print(p)
print('hmm state 0: {}'.format(model.states[0].name))
print('hmm state 1: {}'.format(model.states[1].name))

# print(model.to_json())
