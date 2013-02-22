import numpy

class ModelLearner:

	def __init__(self, numDiscStates, contFeatureRanges, numActions, rewardRange, params={}):
		self.numDiscStates = numDiscStates
		self.numContStates = len(contFeatureRanges)
		self.numActions = numActions
		self.reward_range = rewardRange
		self.params = params
		self.feature_ranges = numpy.array([[0, self.numDiscStates-1]] + list(contFeatureRanges))
		self.feature_span = numpy.ones((len(self.feature_ranges),))
		non_constants = self.feature_ranges[:,0]!=self.feature_ranges[:,1]
		self.feature_span[non_constants] = self.feature_ranges[non_constants,1] - self.feature_ranges[non_constants,0]

        def updateExperience(self, lastState, action, newState, reward):
		return False
	
        def getStateSpace(self):
		return self.feature_ranges, self.numActions

	# This method does not gaurantee that num_requested is filled, but will not 
	# provide more than num_requested.
	def sampleStateActions(self, num_requested):
		pass

	def predict(self, state, action):
		pass

	def predictSet(self, states):
		pass


	def isKnown(self, state, action):
		return False


