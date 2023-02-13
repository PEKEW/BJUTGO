#!/usr/bin/python3
# 使用Python内置GUI模块tkinter
from tkinter import *
# ttk覆盖tkinter部分对象，ttk对tkinter进行了优化
from tkinter.ttk import *
import tkinter.messagebox
# 深拷贝时需要用到copy模块
import copy 
import numpy as np
import law
from collections import deque
import logging
import torch
import torch.cuda
from Arg import Args
from torch.autograd import Variable
from mcts2 import MCT2
from multiprocessing import Process, Queue, Pipe, Lock
import time


args = Args()
# 围棋应用对象定义
class Application(Tk):
    # 初始化棋盘,默认九路棋盘
    def __init__(self, my_mode_num=args.ways):
        Tk.__init__(self)
        self.model = torch.load('./MODELS/EXP_0_9000', map_location='cuda:0')
        self.model.share_memory()
        self.model.eval()
        self.receve1, self.send1 = Pipe(False)
        self.receve2, self.send2 = Pipe(False)
        self.server = Process(target=Server, args=(copy.deepcopy(self.model), self.receve1, self.send2))
        self.server.start()
        # 模式，九路棋：9，十三路棋：13，十九路棋：19
        self.mode_num = my_mode_num
        # 窗口尺寸设置，默认：1.8
        self.size = 1.8
        # 棋盘每格的边长
        self.dd = 360 * self.size / (self.mode_num - 1)
        # 相对九路棋盘的矫正比例
        self.p = 1 if self.mode_num == 9 else (2 / 3 if self.mode_num == 13 else 4 / 9)
        # 定义棋盘阵列,超过边界：-1，无子：0，黑棋：1，白棋：2
        self.positions = [[0 for i in range(self.mode_num + 2)] for i in range(self.mode_num + 2)]
        # 初始化棋盘，所有超过边界的值置-1
        self.ourBlack = False
        for m in range(self.mode_num + 2):
            for n in range(self.mode_num + 2):
                if m * n == 0 or m == self.mode_num + 1 or n == self.mode_num + 1: # 边界
                    self.positions[m][n] = -1
        self.last_1_positions = copy.deepcopy(self.positions)
        self.last_2_positions = copy.deepcopy(self.positions)
        # 记录鼠标经过的地方，用于显示shadow时
        self.cross_last = None
        # 当前轮到的玩家，黑：0，白：1，执黑先行
        self.present = 0
        # 初始停止运行，点击“开始游戏”运行游戏
        self.stop = True
        # 悔棋次数，次数大于2才可悔棋，初始置0（初始不能悔棋），悔棋后置0，下棋或弃手时恢复为1，以禁止连续悔棋
        self.regretchance = 0
        # 图片资源，存放在当前目录下的/Pictures/中
        self.photoW = PhotoImage(file="./Pictures/W.png")
        self.photoB = PhotoImage(file="./Pictures/B.png")
        self.photoBD = PhotoImage(file="./Pictures/" + "BD" + "-" + str(self.mode_num) + ".png")
        self.photoWD = PhotoImage(file="./Pictures/" + "WD" + "-" + str(self.mode_num) + ".png")
        self.photoBU = PhotoImage(file="./Pictures/" + "BU" + "-" + str(self.mode_num) + ".png")
        self.photoWU = PhotoImage(file="./Pictures/" + "WU" + "-" + str(self.mode_num) + ".png")
        # 用于黑白棋子图片切换的列表
        self.photoWBU_list = [self.photoBU, self.photoWU]
        self.photoWBD_list = [self.photoBD, self.photoWD]
        # 窗口大小
        self.geometry(str(int(600 * self.size)) + 'x' + str(int(400 * self.size)))
        self.black_histry = deque(maxlen=8)
        self.white_histry = deque(maxlen=8)
        for i in range(8):
            self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
            self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
        self.black_1_trash = np.zeros((args.ways, args.ways))
        self.white_1_trash = np.zeros((args.ways, args.ways))
        self.black_2_trash = np.zeros((args.ways, args.ways))
        self.white_2_trash = np.zeros((args.ways, args.ways))
        self.rob_list = []
        self.rob_1_trash = []
        self.rob_2_trash = []
        self.gohistory = []

        # 画布控件，作为容器
        self.canvas_bottom = Canvas(self, bg='#369', bd=0, width=600 * self.size, height=400 * self.size)
        self.canvas_bottom.place(x=0, y=0)
        # 几个功能按钮
        self.startButton = Button(self, text='Start Game', command=self.start)
        self.startButton.place(x=480 * self.size, y=200 * self.size)
        self.passmeButton = Button(self, text='Pass', command=self.passme)
        self.passmeButton.place(x=480 * self.size, y=225 * self.size)
        self.regretButton = Button(self, text='Regret', command=self.regret)
        self.regretButton.place(x=480 * self.size, y=250 * self.size)
        # 初始悔棋按钮禁用
        self.regretButton['state'] = DISABLED
        self.replayButton = Button(self, text='Restart', command=self.reload)
        self.replayButton.place(x=480 * self.size, y=275 * self.size)
        self.quitButton = Button(self, text='End Game', command=self.quit)
        self.quitButton.place(x=480 * self.size, y=300 * self.size)

        # 画棋盘，填充颜色
        self.canvas_bottom.create_rectangle(0 * self.size, 0 * self.size, 400 * self.size, 400 * self.size, fill='#c51')
        # 刻画棋盘线及九个点
        # 先画外框粗线
        self.canvas_bottom.create_rectangle(20 * self.size, 20 * self.size, 380 * self.size, 380 * self.size, width=3)
        # 棋盘上的九个定位点，以中点为模型，移动位置，以作出其余八个点
        for m in [-1, 0, 1]:
            for n in [-1, 0, 1]:
                self.oringinal = self.canvas_bottom.create_oval(200 * self.size - self.size * 2,
                                                                200 * self.size - self.size * 2,
                                                                200 * self.size + self.size * 2,
                                                                200 * self.size + self.size * 2, fill='#000')
                self.canvas_bottom.move(self.oringinal,
                                        m * self.dd * (2 if self.mode_num == 9 else (3 if self.mode_num == 13 else 6)),
                                        n * self.dd * (2 if self.mode_num == 9 else (3 if self.mode_num == 13 else 6)))
        # 画中间的线条
        for i in range(1, self.mode_num - 1):
            self.canvas_bottom.create_line(20 * self.size, 20 * self.size + i * self.dd, 380 * self.size,
                                           20 * self.size + i * self.dd, width=2)
            self.canvas_bottom.create_line(20 * self.size + i * self.dd, 20 * self.size, 20 * self.size + i * self.dd,
                                           380 * self.size, width=2)
        # 放置右侧初始图片
        self.pW = self.canvas_bottom.create_image(500 * self.size + 11, 65 * self.size, image=self.photoW)
        self.pB = self.canvas_bottom.create_image(500 * self.size - 11, 65 * self.size, image=self.photoB)
        # 每张图片都添加image标签，方便reload函数删除图片
        self.canvas_bottom.addtag_withtag('image', self.pW)
        self.canvas_bottom.addtag_withtag('image', self.pB)

        # 鼠标移动时，调用shadow函数，显示随鼠标移动的棋子
        self.canvas_bottom.bind('<Motion>', self.shadow)
        # 鼠标左键单击时，调用getdown函数，放下棋子
        self.canvas_bottom.bind('<Button-1>', self.getDown)


    # 开始游戏函数，点击“开始游戏”时调用
    def start(self):
        # 删除右侧太极图
        self.canvas_bottom.delete(self.pW)
        self.canvas_bottom.delete(self.pB)
        # 利用右侧图案提示开始时谁先落子
        if self.present == 0:
            self.create_pB()
            self.del_pW()
        else:
            self.create_pW()
            self.del_pB()
        # 开始标志，解除stop
        self.stop = False
        self.ourBlack = tkinter.messagebox.askokcancel(title='', message='If our machine is black?')
        if self.ourBlack:
            self.autoDown()

    # 放弃一手函数，跳过落子环节
    def passme(self):
        # 悔棋恢复
        if self.regretchance < 2:
            self.regretchance += 1
        if self.regretchance == 2:
            self.regretButton['state'] = NORMAL
        # 拷贝棋盘状态，记录前二次棋局
        self.last_2_positions = copy.deepcopy(self.last_1_positions)
        self.last_1_positions = copy.deepcopy(self.positions)
        self.canvas_bottom.delete('image_added_sign')
        self.black_2_trash = copy.deepcopy(self.black_1_trash)
        self.black_1_trash = copy.deepcopy(self.black_histry[-1])
        self.white_2_trash = copy.deepcopy(self.white_1_trash)
        self.white_1_trash = copy.deepcopy(self.white_histry[-1])
        self.black_histry.appendleft(copy.deepcopy(self.black_histry[0]))
        self.white_histry.appendleft(copy.deepcopy(self.white_histry[0]))
        self.rob_2_trash = copy.deepcopy(self.rob_1_trash)
        self.rob_1_trash = copy.deepcopy(self.rob_list)
        self.rob_list = []

        # 轮到下一玩家
        if self.present == 0:
            self.create_pW()
            self.del_pB()
            log = 'blackpass'
            self.present = 1
        else:
            self.create_pB()
            self.del_pW()
            log = 'whitepass'
            self.present = 0
        print(log)
        # self.gohistory.append(log)
        self.autoDown()

    # 悔棋函数，可悔棋一回合，下两回合不可悔棋
    def regret(self):
        # 判定是否可以悔棋，以前第二盘棋局复原棋盘
        if self.regretchance >= 2:
            self.regretchance = 0
            list_of_b = []
            list_of_w = []
            self.canvas_bottom.delete('image')
            if self.present == 0:
                self.create_pB()
                self.del_pW()
                self.present = 0
            else:
                self.create_pW()
                self.del_pB()
                self.present = 1
            for m in range(1, self.mode_num + 1):
                for n in range(1, self.mode_num + 1):
                    self.positions[m][n] = 0
            for m in range(1, self.mode_num + 1):
                for n in range(1, self.mode_num + 1):
                    if self.last_2_positions[m][n] == 1:
                        list_of_b += [[m, n]]
                    elif self.last_2_positions[m][n] == 2:
                        list_of_w += [[m, n]]
            self.recover(list_of_b, 0)
            self.recover(list_of_w, 1)
            self.black_histry.append(copy.deepcopy(self.black_1_trash))
            self.black_histry.append(copy.deepcopy(self.black_2_trash))
            self.white_histry.append(copy.deepcopy(self.white_1_trash))
            self.white_histry.append(copy.deepcopy(self.white_2_trash))
            self.rob_list = copy.deepcopy(self.rob_2_trash)
            del self.gohistory[-2:]
            print('regret')


    # 重新加载函数,删除图片，序列归零，设置一些初始参数，点击“重新开始”时调用
    def reload(self):
        with open(str(time.time())+'.txt', 'w') as f:
            for line in self.gohistory:
                f.write(line+'\n')
        if self.stop == True:
            self.stop = False
        self.canvas_bottom.delete('image')
        self.regretchance = 0
        self.present = 0
        self.create_pB()
        for m in range(1, self.mode_num + 1):
            for n in range(1, self.mode_num + 1):
                self.positions[m][n] = 0
                self.last_1_positions[m][n] = 0
                self.last_2_positions[m][n] = 0
        self.ourBlack = tkinter.messagebox.askokcancel(title='', message='If our machine is black?')
        for i in range(8):
            self.black_histry.appendleft(np.zeros((args.ways, args.ways)))
            self.white_histry.appendleft(np.zeros((args.ways, args.ways)))
        self.black_1_trash = np.zeros((args.ways, args.ways))
        self.white_1_trash = np.zeros((args.ways, args.ways))
        self.black_2_trash = np.zeros((args.ways, args.ways))
        self.white_2_trash = np.zeros((args.ways, args.ways))
        self.rob_list = []
        self.rob_1_trash = []
        self.rob_2_trash = []
        self.gohistory = []
        if self.ourBlack:
            self.autoDown()
    
    def quit(self):
        self.destroy()
        self.server.terminate()
        with open(str(time.time())+'.txt', 'w') as f:
            for line in self.gohistory:
                f.write(line+'\n')

    # 以下四个函数实现了右侧太极图的动态创建与删除
    def create_pW(self):
        self.pW = self.canvas_bottom.create_image(500 * self.size + 11, 65 * self.size, image=self.photoW)
        self.canvas_bottom.addtag_withtag('image', self.pW)

    def create_pB(self):
        self.pB = self.canvas_bottom.create_image(500 * self.size - 11, 65 * self.size, image=self.photoB)
        self.canvas_bottom.addtag_withtag('image', self.pB)

    def del_pW(self):
        self.canvas_bottom.delete(self.pW)

    def del_pB(self):
        self.canvas_bottom.delete(self.pB)

    # 显示鼠标移动下棋子的移动
    def shadow(self, event):
        if not self.stop:
            # 找到最近格点，在当前位置靠近的格点出显示棋子图片，并删除上一位置的棋子图片
            if (20 * self.size < event.x < 380 * self.size) and (20 * self.size < event.y < 380 * self.size): #在棋盘内
                dx = (event.x - 20 * self.size) % self.dd
                dy = (event.y - 20 * self.size) % self.dd
                self.cross = self.canvas_bottom.create_image(event.x - dx + round(dx / self.dd) * self.dd + 22 * self.p,
                                                             event.y - dy + round(dy / self.dd) * self.dd - 27 * self.p,
                                                             image=self.photoWBU_list[self.present])
                self.canvas_bottom.addtag_withtag('image', self.cross)
                if self.cross_last != None:
                    self.canvas_bottom.delete(self.cross_last)
                self.cross_last = self.cross

    # 落子，并驱动玩家的轮流下棋行为
    def getDown(self, event):
        if not self.stop:
            self.last_2_positions = copy.deepcopy(self.last_1_positions)
            self.last_1_positions = copy.deepcopy(self.positions)  # 保存上一步
            self.black_2_trash = copy.deepcopy(self.black_1_trash)
            self.black_1_trash = copy.deepcopy(self.black_histry[-1])
            self.white_2_trash = copy.deepcopy(self.white_1_trash)
            self.white_1_trash = copy.deepcopy(self.white_histry[-1])
            self.rob_2_trash = copy.deepcopy(self.rob_1_trash)
            self.rob_1_trash = copy.deepcopy(self.rob_list)

            # 先找到最近格点
            if (20 * self.size - self.dd * 0.4 < event.x < self.dd * 0.4 + 380 * self.size) and (
                    20 * self.size - self.dd * 0.4 < event.y < self.dd * 0.4 + 380 * self.size):
                dx = (event.x - 20 * self.size) % self.dd
                dy = (event.y - 20 * self.size) % self.dd
                x = int((event.x - 20 * self.size - dx) / self.dd + round(dx / self.dd) + 1)
                y = int((event.y - 20 * self.size - dy) / self.dd + round(dy / self.dd) + 1)
                self.positions[y][x] = self.present + 1
                self.image_added = self.canvas_bottom.create_image(
                    event.x - dx + round(dx / self.dd) * self.dd + 4 * self.p,
                    event.y - dy + round(dy / self.dd) * self.dd - 5 * self.p,
                    image=self.photoWBD_list[self.present])
                self.canvas_bottom.addtag_withtag('image', self.image_added)
                # 棋子与位置标签绑定，方便“杀死”
                self.canvas_bottom.addtag_withtag('position_' + str(y) + '_' + str(x), self.image_added)

                if self.regretchance < 2:
                    self.regretchance += 1
                if self.regretchance == 2:
                    self.regretButton['state'] = NORMAL

                self.canvas_bottom.delete('image_added_sign')
                self.image_added_sign = self.canvas_bottom.create_oval(
                    event.x - dx + round(dx / self.dd) * self.dd + 0.5 * self.dd,
                    event.y - dy + round(dy / self.dd) * self.dd + 0.5 * self.dd,
                    event.x - dx + round(dx / self.dd) * self.dd - 0.5 * self.dd,
                    event.y - dy + round(dy / self.dd) * self.dd - 0.5 * self.dd, width=3, outline='#3ae')
                self.canvas_bottom.addtag_withtag('image', self.image_added_sign)
                self.canvas_bottom.addtag_withtag('image_added_sign', self.image_added_sign)

                black, white = self.convert()
                if self.present == 0:  # 黑子
                    self.create_pW()
                    self.del_pB()
                    self.present = 1
                    token_list, _ = law.take_out(white, black)
                    log = 'B'+'['+chr(ord('A')+x-1)+chr(ord('S')-y+1)+']'
                else:  # 白子
                    self.create_pB()
                    self.del_pW()
                    self.present = 0
                    token_list, _ = law.take_out(black, white)
                    log = 'W'+'['+chr(ord('A')+x-1) +chr(ord('S')-y+1) +']'
                self.rob_list = [each[0] * args.ways + each[1] for each in token_list]
                self.kill(self.rob_list)
                print(log)
                self.gohistory.append(log)

                black, white = self.convert()
                self.black_histry.appendleft(copy.deepcopy(black))
                self.white_histry.appendleft(copy.deepcopy(white))

                self.autoDown()

    # 恢复位置列表list_to_recover为b_or_w指定的棋子
    def recover(self, list_to_recover, b_or_w):
        if len(list_to_recover) > 0:
            for i in range(len(list_to_recover)):
                self.positions[list_to_recover[i][0]][list_to_recover[i][1]] = b_or_w + 1
                self.image_added = self.canvas_bottom.create_image(
                    20 * self.size + (list_to_recover[i][1] - 1) * self.dd - 5 * self.p,
                    20 * self.size + (list_to_recover[i][0] - 1) * self.dd + 4 * self.p,
                    image=self.photoWBD_list[b_or_w])
                self.canvas_bottom.addtag_withtag('image', self.image_added)
                self.canvas_bottom.addtag_withtag('position_' + str(list_to_recover[i][0]) + '_' + str(list_to_recover[i][1]),
                                                  self.image_added)

    # 杀死位置列表killList中的棋子，即删除图片，位置值置0
    def kill(self, killList):
        for one in killList:
            self.positions[one//args.ways+1][one%args.ways+1] = 0
            self.canvas_bottom.delete('position_' + str(one//args.ways+1) + "_" + str(one%args.ways+1))

    # 键盘快捷键退出游戏
    def keyboardQuit(self, event):
        self.quit()

    def convert(self):
        black = np.zeros((args.ways, args.ways))
        white = np.zeros((args.ways, args.ways))
        for i in range(args.ways):
            for j in range(args.ways):
                if self.positions[i+1][j+1] == 1:
                    black[i, j] = 1
                elif self.positions[i+1][j+1] == 2:
                    white[i, j] = 1
        return black, white

    def autoDown(self):
        self.canvas_bottom.unbind('<Motion>')
        self.canvas_bottom.unbind('<Button-1>')
        self.regretButton['state'] = DISABLED
        self.startButton['state'] = DISABLED
        self.passmeButton['state'] = DISABLED
        self.replayButton['state'] = DISABLED
        self.quitButton['state'] = DISABLED
        self.canvas_bottom.delete('image_added_sign')
        input = []
        if self.ourBlack:
            self.present = 0
            for i in range(8):
                input.append(copy.deepcopy(self.black_histry[i]).reshape((1, args.ways, args.ways)))
            for i in range(8):
                input.append(copy.deepcopy(self.white_histry[i]).reshape((1, args.ways, args.ways)))
            input.append(np.ones((1, args.ways, args.ways)))
            input.append(np.zeros((1, args.ways, args.ways)))
            player = 0
        else:
            self.present = 1
            for i in range(8):
                input.append(copy.deepcopy(self.white_histry[i]).reshape((1, args.ways, args.ways)))
            for i in range(8):
                input.append(copy.deepcopy(self.black_histry[i]).reshape((1, args.ways, args.ways)))
            input.append(np.zeros((1, args.ways, args.ways)))
            input.append(np.ones((1, args.ways, args.ways)))
            player = 1
        input = np.vstack(input)
        best = MCT2(self.model, player=player, net_env=input.copy().reshape((1, 18, 19, 19)),
                   rob_list=self.rob_list, args=args, send1=self.send1, receve2=self.receve2)
        succ_flag, [me, en], rob_list = law.check([input[0].copy(), input[8].copy()], best, self.rob_list, args)
        self.last_2_positions = copy.deepcopy(self.last_1_positions)
        self.last_1_positions = copy.deepcopy(self.positions)  # 保存上一步
        self.black_2_trash = copy.deepcopy(self.black_1_trash)
        self.black_1_trash = copy.deepcopy(self.black_histry[-1])
        self.white_2_trash = copy.deepcopy(self.white_1_trash)
        self.white_1_trash = copy.deepcopy(self.white_histry[-1])
        self.rob_2_trash = copy.deepcopy(self.rob_1_trash)
        self.rob_1_trash = copy.deepcopy(self.rob_list)
        self.rob_list = copy.deepcopy(rob_list)
        if self.present ==0:
            log = 'B'
        else:
            log = 'W'
        x = best // args.ways
        y = best % args.ways
        if best != 361:
            self.positions[x + 1][y + 1] = self.present + 1
            log += '['+chr(ord('A')+int(y)) +chr(ord('S')-int(x)) +']'
            self.kill(self.rob_list)
            self.image_added = self.canvas_bottom.create_image(
                y * self.dd + 20 * self.size + 4 * self.p,
                x * self.dd + 20 * self.size - 5 * self.p,
                image=self.photoWBD_list[self.present])
            self.canvas_bottom.addtag_withtag('image', self.image_added)
            # 棋子与位置标签绑定，方便“杀死”
            self.canvas_bottom.addtag_withtag('position_' + str(x + 1) + '_' + str(y + 1), self.image_added)
            self.image_added_sign = self.canvas_bottom.create_oval(
                y * self.dd + 20 * self.size + 0.5 * self.dd,
                x * self.dd + 20 * self.size + 0.5 * self.dd,
                y * self.dd + 20 * self.size - 0.5 * self.dd,
                x * self.dd + 20 * self.size - 0.5 * self.dd, width=3, outline='#3ae')
            self.canvas_bottom.addtag_withtag('image', self.image_added_sign)
            self.canvas_bottom.addtag_withtag('image_added_sign', self.image_added_sign)
        else:
            log += 'passSSSssSSsSSs'
        print(log)
        if best != 361: self.gohistory.append(log)

        
        black, white = self.convert()
        self.black_histry.appendleft(copy.deepcopy(black))
        self.white_histry.appendleft(copy.deepcopy(white))
        if self.regretchance < 2:
            self.regretchance += 1
        if self.regretchance == 2:
            self.regretButton['state'] = NORMAL

        if self.present == 0:  # 黑子
            self.create_pW()
            self.del_pB()
            self.present = 1
        else:  # 白子
            self.create_pB()
            self.del_pW()
            self.present = 0

        self.canvas_bottom.bind('<Motion>', self.shadow)
        self.canvas_bottom.bind('<Button-1>', self.getDown)
        self.startButton['state'] = NORMAL
        self.passmeButton['state'] = NORMAL
        self.replayButton['state'] = NORMAL
        self.quitButton['state'] = NORMAL

def Server(net, pin, pout):
    while True:
        input = pin.recv()
        score = net(tensor(input, dtype=torch.float).cuda())[1].item()
        pout.send(score)

def callback():
    pass

# 声明全局变量，用于新建Application对象时切换成不同模式的游戏
global mode_num, newApp
mode_num = args.ways
newApp = False
if __name__ == '__main__':
    # 循环，直到不切换游戏模式
    torch.cuda.set_device(args.gpu_id)
    assert torch.cuda.is_available(), logging.error('NO AVAILABLE GPU DEVICE')
    app = Application(mode_num)
    app.title('GO')
    app.protocol("WM_DELETE_WINDOW", callback)
    app.mainloop()

