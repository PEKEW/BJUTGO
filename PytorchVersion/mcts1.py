#!/usr/bin/env python
# -*- coding: utf-8 -*-

import math
import random
import sys

AVAILABLE_CHOICES = [1, -1, 2, -2]
AVAILABLE_CHOICE_NUMBER = len(AVAILABLE_CHOICES)
MAX_ROUND_NUMBER = 10
MAX_PASSES = 2


class State(object):
	"""
	The monte Carlo tree searches for the state of the game,
	recording the state data in a Node, including the current game score,
	the current number of games round, from the start to the current execution record.
	To determine whether the current state has reached the end of the game state,
	support random Action from the Action set.
  """

	def __init__(self, env):
		self.current_value = 0.0
		# For the first root node, the index is 0 and the game should start from 1
		self.current_round_index = 0
		self.cumulative_choices = []
		self.state = env
		self.passes = 0

	def get_current_value(self):
		return self.current_value

	def set_current_value(self, value):
		self.current_value = value

	def get_current_round_index(self):
		return self.current_round_index

	def set_current_round_index(self, turn):
		self.current_round_index = turn

	def get_cumulative_choices(self):
		return self.cumulative_choices

	def set_cumulative_choices(self, choices):
		self.cumulative_choices = choices

	def is_terminal(self):
		# The round index starts from 1 to max round number
		return self.current_round_index == MAX_ROUND_NUMBER or self.passes > MAX_PASSES

	def compute_reward(self):
		return -abs(1 - self.current_value)

	def get_next_state_with_random_choice(self):
		random_choice = random.choice([choice for choice in AVAILABLE_CHOICES])

		next_state = State()
		next_state.set_current_value(self.current_value + random_choice)
		next_state.set_current_round_index(self.current_round_index + 1)
		next_state.set_cumulative_choices(self.cumulative_choices +
		                                  [random_choice])

		return next_state

	def __repr__(self):
		return "State: {}, value: {}, round: {}, choices: {}".format(
			hash(self), self.current_value, self.current_round_index,
			self.cumulative_choices)


class Node(object):
	"""
	The node of the tree structure searched by the Monte Carlo tree contains information
	such as parent nodes and direct points, as well as the number of traversals and quality
	values used to calculate UCB, and the State of the node that the game selects
	"""

	def __init__(self):
		self.parent = None
		self.children = []

		self.visit_times = 0
		self.quality_value = 0.0

		self.state = None
		self.is_expand = False

	def set_state(self, state):
		self.state = state

	def get_state(self):
		return self.state

	def get_parent(self):
		return self.parent

	def set_parent(self, parent):
		self.parent = parent

	def get_children(self):
		return self.children

	def get_visit_times(self):
		return self.visit_times

	def set_visit_times(self, times):
		self.visit_times = times

	def visit_times_add_one(self):
		self.visit_times += 1

	def get_quality_value(self):
		return self.quality_value

	def set_quality_value(self, value):
		self.quality_value = value

	def quality_value_add_n(self, n):
		self.quality_value += n

	def is_all_expand(self):
		return self.is_expand

	def add_child(self, sub_node):
		sub_node.set_parent(self)
		self.children.append(sub_node)

	def __repr__(self):
		return "Node: {}, Q/N: {}/{}, state: {}".format(
			hash(self), self.quality_value, self.visit_times, self.state)


def tree_policy(node):
	"""
	  In the Selection and Expansion stages of Monte Carlo tree search,
	  the node that needs to be searched (such as the root node) is passed in,
	  and the best node that needs to be expanded is returned according to the
	  exploration/exploitation algorithm. Note that if the node is a leaf node, it is returned directly.
	  The basic strategy is to find the currently unselected child node first,
	  if there are more than one, select randomly. If you have all chosen, find
	  the largest UCB value that has been weighed against exploration/exploitation.
	  If the UCB values are equal, choose randomly.
	"""

	# Check if the current node is the leaf node
	while not node.get_state().is_terminal():

		if node.is_all_expand():
			node = best_child(node, True)
		else:
			# Return the new sub node
			sub_node = expand(node)
			return sub_node

	# Return the leaf node
	return node


def default_policy(node):
	"""
	In the Simulation stage of monte Carlo tree search, input a node to expand,
	create a new node after random operation, and return the reward of the new node.
	Note that the input node should not be a child node, and there should be an unexecuted Action to expend.
	The basic strategy is to randomly select the Action.
  """

	# Get the state of the game
	current_state = node.get_state()

	# Run until the game over
	while not current_state.is_terminal():
		# Pick one random action to play and get next state
		current_state = current_state.get_next_state_with_random_choice()

	final_state_reward = current_state.compute_reward()
	return final_state_reward


def expand(node):
	"""
	Enter a node, extend a new node on top of it, and perform an Action using the random method to return the new node.
	Note that you need to ensure that the new node has a different Action than the other nodes.
	"""

	tried_sub_node_states = [
		sub_node.get_state() for sub_node in node.get_children()
	]

	new_state = node.get_state().get_next_state_with_random_choice()

	# Check until get the new state which has the different action from others
	while new_state in tried_sub_node_states:
		new_state = node.get_state().get_next_state_with_random_choice()

	sub_node = Node()
	sub_node.set_state(new_state)
	node.add_child(sub_node)

	return sub_node


def best_child(node, is_exploration):
	"""
		Using the UCB algorithm, select the child node with the highest score
		after weighing the exploration and exploitation.
		Note that if it is the prediction stage, directly select the current Q-score.
	"""

	best_score = -sys.maxsize
	best_sub_node = None

	# Travel all sub nodes to find the best one
	for sub_node in node.get_children():

		# Ignore exploration for inference
		if is_exploration:
			C = 1 / math.sqrt(2.0)
		else:
			C = 0.0

		# UCB = quality / times + C * sqrt(2 * ln(total_times) / times)
		left = sub_node.get_quality_value() / sub_node.get_visit_times()
		right = 2.0 * math.log(node.get_visit_times()) / sub_node.get_visit_times()
		score = left + C * math.sqrt(right)

		if score > best_score:
			best_sub_node = sub_node
			best_score = score

	return best_sub_node


def backup(node, reward):
	"""
	In the Backpropagation phase of the Monte Carlo tree search,
	input the node that needs to be expend and the reward of the newly executed Action,
	feed back to the expend node and all upstream nodes and update the corresponding data.
	"""

	# Update util the root node
	while node is not None:
		# Update the visit times
		node.visit_times_add_one()

		# Update the quality value
		node.quality_value_add_n(reward)

		# Change the node to the parent node
		node = node.parent


def monte_carlo_tree_search(node):
	"""
	Implement the Monte Carlo tree search algorithm, pass in a root node, expand new nodes and update data according to the previously explored tree structure in a limited time, and then return as long as the highest-rated child node.
	Monte Carlo tree search consists of four steps: Selection, Expansion, Simulation, and Backpropagation.
	The first two steps use Tree Policy to find nodes worth exploring.
	The third step is to use the default policy, that is, to randomly select a child node on the selected node and calculate the reward.
	The last step is to use backup to update the reward to all nodes that pass through the selected node.
	When making prediction, only the node with the largest exploitation need to be selected according to the Q value to find the next optimal node.
  """

	computation_budget = 2

	for i in range(computation_budget):
		# 1. Find the best node to expand
		expand_node = tree_policy(node)

		# 2. Random run to add node and get reward
		reward = default_policy(expand_node)

		# 3. Update all passing nodes with reward
		backup(expand_node, reward)

	# N. Get the best next node
	best_next_node = best_child(node, False)

	return best_next_node


def main(env):
	init_state = State(env)
	init_node = Node()
	init_node.set_state(init_state)
	current_node = init_node

	# Set the rounds to play
	for i in range(10):
		print("Play round: {}".format(i + 1))
		current_node = monte_carlo_tree_search(current_node)
		print("Choose node: {}".format(current_node))


if __name__ == "__main__":
	main()
