import sys
from collections import Counter
from copy import deepcopy
from math import log, sqrt
from multiprocessing import Process, Queue, Pipe, Lock
from random import choice

import torch
from numpy import setdiff1d, where, argwhere, concatenate, ones, zeros
from torch import tensor

from Arg import Args
from law import check

MAX_COMPUTE = 9
MAX_DEEP = 20


class Queue_:
    def __init__(self):
        self.info = [0, 0, 0, 0, 0, 0, 0, 0, ]

    def push(self, x):
        """
        left -> right <==> old -> new
        """
        self.info.append(x)
        self.info = self.info[1::]


class State:
    """
    env => current board
    passes => point current players' pass opt, game will over when passes >= 3
    rob_list => jie
    player => 0 if current player is black else 1
    """

    def __init__(self, env, rob_list, player, init_player, available=None):
        self.env = env
        self.passes = [0, 0]
        self.rob_list = rob_list
        self.player = player
        self.init_player = init_player
        self.arg = Args()
        self.available = available

    def change_player(self):
        # self.player = 0 if self.player == 1 else 1
        return 0 if self.player == 1 else 1

    def is_terminal(self, count=0):
        """
        when passes >=3 , game over
        """
        return self.passes[0] > 2 or self.passes[1] > 2 or count > MAX_DEEP

    def init_terminal(self):
        self.passes = [0, 0]

    def get_next_state_with_random_choice(self):
        """
        using random to choice the option then generate the next state
        @return:
        """
        succ_flag = False
        env = deepcopy(self.env)

        if self.available is not None:
            available_choices = self.available
        else:
            available_choices = setdiff1d(where(((env[1] == 0) == (env[0] == 0)).reshape(361))[0], self.rob_list)
        while not succ_flag:
            random_choice = choice([choiced for choiced in available_choices])
            succ_flag, env, rob_list = check(env, random_choice, self.rob_list, self.arg)
            available_choices = setdiff1d(available_choices, [random_choice])
            if len(available_choices) == 0:
                random_choice = 361
                break
        env[0], env[1] = env[1], env[0]
        next_state = State(env, rob_list, self.change_player(), init_player=self.init_player)

        if random_choice == 361:
            self.passes[self.player] += 1
            next_state.passes = self.passes
        return next_state


class Node:
    """
    state => State Obj /  board
    children => list of current node's child nodes
    parent => current node's parent node
    visit_times => pointer of current node's visited times
    quality_value => win times
    is_expand => indicator current node if be expanded or not
    """

    def __init__(self):
        self.state = None
        self.children = []
        self.parent = None
        self.visit_times = 0
        self.quality_value = 0.0
        self.score = 0.0

    def __setstate__(self, state):
        self.state = state

    def __getstate__(self):
        return self.state

    def set_score(self, score):
        self.score = score

    def get_children(self):
        return self.children

    def get_quality_value(self):
        return self.quality_value

    def get_visit_times(self):
        return self.visit_times

    def add_child(self, sub_node):
        sub_node.set_parent(self)
        self.children.append(sub_node)

    def visit_times_add_one(self):
        self.visit_times += 1

    def quality_value_add_n(self, n):
        self.quality_value += n

    def is_all_expand(self):
        return len(self.children) >= 3

    def set_parent(self, parent):
        self.parent = parent


def best_child(node):
    best_score = -sys.maxsize
    best_sub_node = None
    C = 1.4
    for sub_node in node.get_children():
        left = sub_node.get_quality_value() / (sub_node.get_visit_times() + 1)
        right = log(node.get_visit_times() + 1) / (sub_node.get_visit_times() + 1)
        score = left + C * sqrt(right)
        if score > best_score:
            best_sub_node = sub_node
            best_score = score
    best_sub_node.set_score(best_score)
    return best_sub_node


def expand(node):
    tried_sub_node_states = [
        sub_node.__getstate__() for sub_node in node.get_children()
    ]
    new_state = node.__getstate__().get_next_state_with_random_choice()

    while new_state in tried_sub_node_states:
        new_state = new_state.get_state().get_next_state_with_random_choice()

    sub_node = Node()
    sub_node.__setstate__(new_state)
    node.add_child(sub_node)
    return sub_node


def tree_policy(node):
    while not node.__getstate__().is_terminal():
        if node.is_all_expand():
            node = best_child(node)
        else:
            sub_node = expand(node)
            return sub_node
    return node


def default_policy(node, net_env, winner, pin, pout, lock):
    # l.acquire()
    current_state = node.__getstate__()
    # l.release()
    count = 0
    # 分别保存黑白历史
    history = [Queue_(), Queue_()]
    # 防止历史信息过少
    other = 1 if current_state.player == 0 else 0
    env = deepcopy(net_env)
    for i in range(8):
        history[other].push(env[0][i])
        history[current_state.player].push(env[0][i + 8])
    history[current_state.player].push(current_state.env[0])

    while not current_state.is_terminal(count):
        count += 1
        current_state = current_state.get_next_state_with_random_choice()
        history[current_state.player].push(current_state.env[0])
    if count > MAX_DEEP:
        if other == 1:  # current is b
            cat_final = concatenate((zeros((1, 19, 19)), ones((1, 19, 19))), axis=0)
        else:
            cat_final = concatenate((ones((1, 19, 19)), zeros((1, 19, 19))), axis=0)
        lock.acquire()
        pout.send(
            concatenate((history[current_state.player].info, history[other].info, cat_final), axis=0).reshape(1, 18, 19,
                                                                                                              19))
        score = pin.recv()
        lock.release()
        current_win = score >= 0
        winner = current_state.player if current_win else other
    else:
        winner = 0 if current_state.passes[0] < current_state.passes[1] else 1
    return [winner, score]


def backup(node, winner, score):
    while node is not None:
        node.__getstate__().init_terminal()
        node.visit_times_add_one()
        node.quality_value_add_n(score if node.state.player == winner else 0)
        node = node.parent
    return node


def Server(net, pin, pout):
    while True:
        input = pin.recv()
        score = net(tensor(input, dtype=torch.float).cuda())[1].item()
        pout.send(score)


def __mcts(init_state, net_env, winner, q, pin, pout, lock):
    node = Node()
    node.__setstate__(init_state)
    init_env = deepcopy(node.__getstate__().env)
    for i in range(MAX_COMPUTE):
        expand_node = tree_policy(node)
        winner, score = default_policy(expand_node, net_env, winner, pin, pout, lock)
        node.__getstate__().init_terminal()
        backup(expand_node, winner, score)
    best_next_node = best_child(node)
    if len(argwhere((init_env[0] == best_next_node.state.env[1]) == False)) == 0:
        q.put(361)
    else:
        q.put(argwhere((init_env[0] == best_next_node.state.env[1]) == False)[0][0] * 19 +
              argwhere((init_env[0] == best_next_node.state.env[1]) == False)[0][1])


def mcts(env, rob_list, available, player, net, net_env, winner, queue):
    # 多线程（两种），一管道
    """
    mcts flow
    @param queue: multiprocessing queue
    @param winner: global value simulation winner
    @param net_env: init env
    @param net: network
    @param env: init env => [board[0], board[1]], board size => [19x19]
    @param rob_list: size => [,361]
    @param player: black if player == 0
    @param available: available move to be judge
    """
    lock = Lock()
    receve1, send1 = Pipe(False)
    receve2, send2 = Pipe(False)
    server = Process(target=Server, args=(deepcopy(net), receve1, send2))
    server.start()
    t_list = []
    for i in range(16):
        init_state = State(env, rob_list, player, available=available, init_player=player)
        t = Process(target=__mcts,
                    args=(deepcopy(init_state), deepcopy(net_env), deepcopy(winner), queue, receve2, send1, lock))
        t_list.append(t)
        t.start()
    res = []
    for t in t_list:
        x = queue.get()
        res.append(x)
        t.join()
    result = max(Counter(res))
    server.terminate()
    return result


def MCT(net, player, net_env, rob_list, args):
    torch.multiprocessing.set_start_method('spawn', force=True)
    policy, _ = net(tensor(net_env, dtype=torch.float).cuda())
    _available = policy.argsort().cpu().numpy()[0][::-1]
    available = []
    count = 0
    for each in _available:
        succ_flag, _, _ = check([net_env[0][0].copy(), net_env[0][8].copy()], each, rob_list, args)
        if succ_flag:
            available.append(each)
            count += 1
            if count == 3:
                break
    q = Queue()
    winner = -1
    best = mcts([net_env[0][0], net_env[0][8]], [], available, player=player, net=net, net_env=net_env, winner=winner,
                queue=q)
    return best


def mcts2(env, rob_list, available, player, net, net_env, winner, queue, send1, receve2):
    # 多线程（两种），一管道
    """
    mcts flow
    @param queue: multiprocessing queue
    @param winner: global value simulation winner
    @param net_env: init env
    @param net: network
    @param env: init env => [board[0], board[1]], board size => [19x19]
    @param rob_list: size => [,361]
    @param player: black if player == 0
    @param available: available move to be judge
    """
    lock = Lock()
    t_list = []
    for i in range(16):
        init_state = State(env, rob_list, player, available=available, init_player=player)
        t = Process(target=__mcts,
                    args=(deepcopy(init_state), deepcopy(net_env), deepcopy(winner), queue, receve2, send1, lock))
        t_list.append(t)
        t.start()
    res = []
    for t in t_list:
        x = queue.get()
        res.append(x)
        t.join()
    result = max(Counter(res))
    return result


def MCT2(net, player, net_env, rob_list, args, send1, receve2):
    torch.multiprocessing.set_start_method('spawn', force=True)
    policy, _ = net(tensor(net_env, dtype=torch.float).cuda())
    _available = policy.argsort().cpu().numpy()[0][::-1]
    available = []
    count = 0
    for each in _available:
        succ_flag, _, _ = check([net_env[0][0].copy(), net_env[0][8].copy()], each, rob_list, args)
        if succ_flag:
            available.append(each)
            count += 1
            if count == 3:
                break
    q = Queue()
    winner = -1
    best = mcts2([net_env[0][0], net_env[0][8]], [], available, player=player, net=net, net_env=net_env, winner=winner,
                 queue=q, send1=send1, receve2=receve2)
    return best
