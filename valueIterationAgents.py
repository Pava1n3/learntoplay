# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0

        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        # fill every state with some action.
        self.actions = dict()
        for state in mdp.getStates():
            stateActions = mdp.getPossibleActions(state)
            if len(stateActions) > 0:
                action = stateActions[0]
                self.actions[state] = action

        for i in xrange(iterations):
            # make a copy of all the values.
            # this copy will get modified in the for-loop,
            # and at the end of the loop,
            # the new values will become then real values.
            nextValues = self.values.copy()

            # for every state, and if it isn't a terminal state
            # (you can't do any action on a terminal state):
            for state in mdp.getStates():
                if not mdp.isTerminal(state):
                    # get the best action.
                    action = self.computeActionFromValues(state)
                    self.actions[state] = action
                        
                    # get the value for doing the currently stored action.
                    nextValues[state] = self.computeQValueFromValues(state, action)

            # copy the new values over the old values.
            self.values.update(nextValues)
            # end of for-loop.

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"
        # get the transition states with their probabilites.
        transitionStatesAndProbs = self.mdp.getTransitionStatesAndProbs(state, action)
        
        # calculate the value of the current state
        # by multiplying the reward of every possible transition state
        # with the probability of going to that next state.
        reward = 0
        for (nextState, prob) in transitionStatesAndProbs:
            nextStateReward = self.mdp.getReward(state, action, nextState) + self.values[nextState]
            reward += prob * nextStateReward

        # return the reward, and if there is more than one action,
        # multiply the reward with the discount.
        # (only a state that goes to the terminal state has one action, namely 'exit')
        if len(transitionStatesAndProbs) != 1:
            return reward * self.discount
        else:
            return reward

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"
        # if this is a terminal state, return 'None'.
        if self.mdp.isTerminal(state):
            return 'None'

        # get every action that can be done on this state.
        possibleActions = self.mdp.getPossibleActions(state)

        # for every action, check if the reward of doing that action
        # is higher than the current best action.
        # if so, then the new best action is the current action.
        bestAction = 'None', -99999
        for action in possibleActions:
            actionReward = self.discount * self.computeQValueFromValues(state, action)
            if actionReward > bestAction[1]:
                bestAction = action, actionReward
        return bestAction[0]

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
