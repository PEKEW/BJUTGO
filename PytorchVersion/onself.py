import os
from collections import deque
from copy import deepcopy
from multiprocessing import Process, Pipe

import numpy as np
import torch
import torch.cuda
from torch.autograd import Variable

import law
from Arg import Args
from mcts2 import MCT2

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

args = Args()


class PLAY:
    def __init__(self, model1path, model2path):
        self.model1 = torch.load(model1path, map_location='cuda:0')
        self.model2 = torch.load(model2path, map_location='cuda:0')
        self.model1.eval()
        self.model2.eval()
        self.rob_list = []
        self.black_histry = deque(maxlen=8)
        self.white_histry = deque(maxlen=8)
        for i in range(8):
            self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
            self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
        self.recevea1, self.senda1 = Pipe(False)
        self.recevea2, self.senda2 = Pipe(False)
        self.receveb1, self.sendb1 = Pipe(False)
        self.receveb2, self.sendb2 = Pipe(False)
        self.servera = Process(target=Server, args=(deepcopy(self.model1), self.recevea1, self.senda2))
        self.servera.start()
        self.serverb = Process(target=Server, args=(deepcopy(self.model2), self.receveb1, self.sendb2))
        self.serverb.start()

    def auto(self):
        """
        :return: the number of model1 win, the number of model2 win
        """
        count1 = 0
        count2 = 0
        for i in range(3):
            if self.__BvsW__(self.model1, self.model2):
                count1 += 1
            else:
                count2 += 1
            for i in range(8):
                self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
                self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
            if self.__BvsW__(self.model2, self.model1):
                count2 += 1
            else:
                count1 += 1
            for i in range(8):
                self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
                self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
        return count1, count2

    def auto2(self):
        """
        :return: the number of model1 win, the number of model2 win
        """
        count1 = 0
        count2 = 0
        for i in range(5):
            if self.__BvsW2__(self.model1, self.senda1, self.recevea2, self.model2, self.sendb1, self.receveb2):
                count1 += 1
            else:
                count2 += 1
            for i in range(8):
                self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
                self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
            if self.__BvsW2__(self.model2, self.sendb1, self.receveb2, self.model1, self.senda1, self.recevea2):
                count2 += 1
            else:
                count1 += 1
            for i in range(8):
                self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
                self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
        self.servera.terminate()
        self.serverb.terminate()
        return count1, count2

    def __BvsW__(self, black, white):
        """
        :param black:
        :param white:
        :return: True if black win
        """
        pass1, pass2 = 0, 0
        while True:
            # black down
            input = []
            for i in range(8):
                input.append(deepcopy(self.black_histry[i]).reshape((1, args.ways, args.ways)))
            for i in range(8):
                input.append(deepcopy(self.white_histry[i]).reshape((1, args.ways, args.ways)))
            input.append(np.ones((1, args.ways, args.ways)))
            input.append(np.zeros((1, args.ways, args.ways)))
            input = np.vstack(input)
            with torch.no_grad():
                policy, _ = black(Variable(torch.Tensor(input.copy().reshape((1, 18, 19, 19)))).cuda())
                policy = policy.cpu().numpy().reshape((-1,))
            policy = np.argsort(policy)[::-1]
            for each_policy in policy:
                succ_flag, [me, en], rob_list = law.check([input[0].copy(), input[8].copy()], each_policy,
                                                          self.rob_list, args)
                if succ_flag:
                    self.rob_list = rob_list
                    self.black_histry.appendleft(me)
                    self.white_histry.appendleft(en)
                    if each_policy == 361:
                        pass1 += 1
                    else:
                        pass1 = 0
                    break
            if pass1 >= 3:
                return False
            # white down
            input = []
            for i in range(8):
                input.append(deepcopy(self.white_histry[i]).reshape((1, args.ways, args.ways)))
            for i in range(8):
                input.append(deepcopy(self.black_histry[i]).reshape((1, args.ways, args.ways)))
            input.append(np.zeros((1, args.ways, args.ways)))
            input.append(np.ones((1, args.ways, args.ways)))
            input = np.vstack(input)
            with torch.no_grad():
                policy, _ = white(Variable(torch.Tensor(input.copy().reshape((1, 18, 19, 19)))).cuda())
                policy = policy.cpu().numpy().reshape((-1,))
            policy = np.argsort(policy)[::-1]
            for each_policy in policy:
                succ_flag, [me, en], rob_list = law.check([input[0].copy(), input[8].copy()], each_policy,
                                                          self.rob_list, args)
                if succ_flag:
                    self.rob_list = rob_list
                    self.black_histry.appendleft(en)
                    self.white_histry.appendleft(me)
                    if each_policy == 361:
                        pass2 += 1
                    else:
                        pass2 = 0
                    break
            if pass2 >= 3:
                return True

    def __BvsW2__(self, black, sendb1, receveb2, white, sendw1, recevew2):
        """
        :param black:
        :param white:
        :return: True if black win
        """
        pass1, pass2 = 0, 0
        while True:
            # black down
            input = []
            for i in range(8):
                input.append(deepcopy(self.black_histry[i]).reshape((1, args.ways, args.ways)))
            for i in range(8):
                input.append(deepcopy(self.white_histry[i]).reshape((1, args.ways, args.ways)))
            input.append(np.ones((1, args.ways, args.ways)))
            input.append(np.zeros((1, args.ways, args.ways)))
            input = np.vstack(input)
            best = MCT2(black, player=0, net_env=input.copy().reshape((1, 18, 19, 19)),
                        rob_list=self.rob_list, args=args, send1=sendb1, receve2=receveb2)
            succ_flag, [me, en], rob_list = law.check([input[0].copy(), input[8].copy()], best,
                                                      self.rob_list, args)
            self.rob_list = rob_list
            self.black_histry.appendleft(me)
            self.white_histry.appendleft(en)
            if best == 361:
                pass1 += 1
            else:
                pass1 = 0
            if pass1 >= 3:
                return False

            # white down
            input = []
            for i in range(8):
                input.append(deepcopy(self.white_histry[i]).reshape((1, args.ways, args.ways)))
            for i in range(8):
                input.append(deepcopy(self.black_histry[i]).reshape((1, args.ways, args.ways)))
            input.append(np.zeros((1, args.ways, args.ways)))
            input.append(np.ones((1, args.ways, args.ways)))
            input = np.vstack(input)
            best = MCT2(white, player=1, net_env=input.copy().reshape((1, 18, 19, 19)),
                        rob_list=self.rob_list, args=args, send1=sendw1, receve2=recevew2)
            succ_flag, [me, en], rob_list = law.check([input[0].copy(), input[8].copy()], best,
                                                      self.rob_list, args)
            self.rob_list = rob_list
            self.black_histry.appendleft(en)
            self.white_histry.appendleft(me)
            if best == 361:
                pass2 += 1
            else:
                pass2 = 0
            if pass2 >= 3:
                return True


def Server(net, pin, pout):
    while True:
        input = pin.recv()
        score = net(torch.tensor(input, dtype=torch.float).cuda())[1].item()
        pout.send(score)


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn', force=True)
    filepaths = []
    for root, dirs, files in os.walk('./BEST/'):
        for file in files:
            filepaths.append(root + "/" + file)
    play = PLAY(filepaths[0], filepaths[1])
    count1, count2 = play.auto2()
    print(filepaths[0] + '\t' + filepaths[1])
    print(str(count1) + "\t" + str(count2))
    if count1 > count2:
        win = filepaths[0]
    else:
        win = filepaths[1]
    print('winner:  ' + win)
    for i in range(2, len(filepaths)):
        play = PLAY(win, filepaths[i])
        count1, count2 = play.auto2()
        print(win + '\t' + filepaths[i])
        print(str(count1) + "\t" + str(count2))
        if count2 > count1:
            win = filepaths[i]
        print('winner:  ' + win)
    print('best model is:' + win)
