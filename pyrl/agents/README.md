pyrl.agents
=========

Reinforcement Learning agents that have been implemented in python using the RLGlue framework.

The following sections describe the algorithms that are implemented in the library and provide some useful references. Different basis can be used with linear function approximators (check the specific options of the agents).

---
### skeleton\_agnet.py
Base class for the agents. Do not use directly (picks random actions all the time).

---
### sarsa\_lambda\_ann.py
Implementation of SARSA (with eligibility traces) and neural networks for approximate Q estimation. The main reference for this algorithm is:

Rummery G. and Niranjan, M. (1994). _On-line q-learning using connectionist systems_. Technical Report CUED/F-INFENG/TR 166, Cambridge University, Engineering Department.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This method uses a single network to estimate Q. The number of inputs in the network is the size of the feature space; the number of outputs is the number of possible discrete actions. In contrast, the original paper proposed to use a *single* network per action.

The original paper also suggests to decrease the exploration rate as the agent learns more from the environment. It looks like this implementation rather has a constant exploration rate (epsilon).

---
### sarsa\_lambda.py
Implementation of SARSA (with eligibility traces) and a linear approximator for Q. 

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This agent is similar to sarsa\_lambda\_ann.py but with a simpler approximator for Q.

---
### qlearning.py
Implementation of Q-Learning with a linear function approximator. Different to SARSA, Q learning uses the best action found so far to update Q's estimate and acts greedily with respect to its estimates.

The main reference for Q-Learning is:

C. J. Watkins. [_Learning from Delayed Rewards_](https://www.cs.rhul.ac.uk/home/chrisw/new_thesis.pdf). Phd thesis, Cambridge University, 1989.

A description of Q-Learning with linear function approximation can be found in:

Francisco S. Melo and M. Isabel Ribeiro, [_Q-learning with linear functionapproximation_](http://gaips.inesc-id.pt/~fmelo/pub/melo07tr-b.pdf). Technical Report, RT-602-07, Instituto de Sistemas e Robótica, Pólo de Lisboa.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This agent inherits sarsa\_lambda.py and re-implements the agent_step and update functions.

---
### delayed_qlearning.py
Implementation of [_PAC Model-Free Reinforcement Learning_](http://www.autonlab.org/icml_documents/camera-ready/111_PAC_Model_free_Reinf.pdf) by Alexander Strehl, Lihong Li, Eric Wiewiora, John Langford, and Michael Littman (2006). 

The standard Q-Learning agent changes its Q-value estimates on every time step. Rather, Delayed Q-Learning waits for _m_ sample updates to make any changes (_m_ is a parameter of the algorithm). According to the above paper, _"this variation has an averaging effect that mitigates some of the effects of randomness"_ and makes it optimistic. _"Since the action-selection strategy is greedy, the Delayed Q-Learning agent will tend to choose overly optimistic actions, therefore achieving direct exploration when necessary"_.


##### REQUIREMENTS
This agent is meant to work with discrete-state/discrete-action domains.

##### NOTES
There might be a bug in this implementation. The code indicates that: _"Unfortunately, I have no yet been able to get this to work consistently on the marble maze domain. It seems likely that it would work on something simpler like chain domain. Maybe there's a bug?"_.

---
### lstd.py
Implements Least Squares Temporal Difference Learning (LSTD). The main reference for this agent is:

Michail Lagoudakis and Ronald Parr, _Least-Squares Policy Iteration_. Journal of Machine Learning Research, v. 4, 2003.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
The code says: _"This is actually very nearly an implementation of LSTD-Q. The only difference with the paper, is that the code does not store the samples themselves, and instead stores A and b. This means that it can't reuse samples as effectively when the policy changes"_.

The implementation inherits sarsa\_lambda.py.

---
### modelbased.py
Implements an agent that learns from the environment (e.g., using linear regression, a super vector machine, or a random forest) and plans using Fitted Q Iteration. The main reference for the planner is:

Damien Ernst, Pierre Geurts and Louis Wehenkel, [_Tree-Based Batch Mode Reinforcement Learning_](http://www.jmlr.org/papers/volume6/ernst05a/ernst05a.pdf). Journal of Machine Learning Research, v.6, 2005.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This implementation supports using a variety of basis functions to represent the agent observations in a different space before passing them to the model learners.

The planner takes care of passing data to the model learner.

---
### mirror\_descent.py
Implements [_Sparse Q-Learning with Mirror Descent_](http://www.auai.org/uai2012/papers/261.pdf) by Sridhar Mahadevan and Bo Liu, 2012. This is a _proximal-gradient_ based temporal-difference (TD) algorithm that uses a p-norm distance generating function.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This agent inherits qlearning.py.

---
### policy\_gradient.py (REINFORCE)
Implements the [REINFORCE](http://www-anw.cs.umass.edu/~barto/courses/cs687/williams92simple.pdf) algorihtm by Ronald Williams.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This agent inherits the policy_gradient class in policy\_gradient.py, which in turn inherits sarsa\_lambda.py.

Breaks with the Tetris environment.

---
### policy\_gradient.py (twotime\_ac)
Implements Regular-Gradient Actor-Critic. This is Algorithm 1 from [Natural Actor-Critic Algorithms](https://webdocs.cs.ualberta.ca/~sutton/papers/BSGL-TR.pdf) by Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, and Mark Lee (2009).

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This agent inherits the policy_gradient class in policy\_gradient.py, which in turn inherits sarsa\_lambda.py.

---
### policy\_gradient.py (twotime\_nac)
Implements Natural-Gradient Actor-Critic with Advantage Parameters. This is Algorithm 3 from [Natural Actor-Critic Algorithms](https://webdocs.cs.ualberta.ca/~sutton/papers/BSGL-TR.pdf) by Shalabh Bhatnagar, Richard S. Sutton, Mohammad Ghavamzadeh, and Mark Lee (2009).

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
This agent inherits the policy_gradient class in policy\_gradient.py, which in turn inherits sarsa\_lambda.py.

---
### policy\_gradient.py (nac_lstd)
Implements the [Natural Actor-Critic](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/NaturalActorCritic.pdf) agent by Jan Peters and Stefan Schaal (2007). The actor updates are based on stochastic policy gradients (using Amari's natural gradient), while the critic obtains the natural gradient and additional parameters of the value function by linear regression.

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
As the code indicates, this implementation _"deviates from the pseudo-code given in the paper because it uses the Sheman-Morrison formula to do incremental updates to the matrix inverse"_.

This agent inherits the policy_gradient class in policy\_gradient.py, which in turn inherits sarsa\_lambda.py.

---
### policy\_gradient.py (nac_sarsa)
Implements the Natural Actor-Critic with SARSA(lambda) by Philip S. Thomas. This is algorithm 2 in his 2012 [Bias in Natural Actor-Critic Algorithms](http://psthomas.com/papers/Thomas2012b.pdf) paper. 

##### REQUIREMENTS
This agent is meant to work with continuous-state/discrete-action domains.

##### NOTES
The code says: _"While fundamentally the same as twotime\_nac (Algorithm 3 of BSGL's paper), this implements NACS which uses a different form of the same update equations. The main difference is in this algorithm's avoidance of the average reward accumulator"_.
