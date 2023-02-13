import gzip
import os
import random
import time

import numpy as np
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from Arg import Args

"""Text format of training data
16 lines of hexadecimal string, each 361 bits longs, corresponding to the 16 board positions;
16 rows of data organization
     Every 8 consecutive rows of a player’s historical hand: [0-7][8-15]
     Assuming that the current Agent is playing white chess, then:
         [0-7] The history of white chess [8-15] is the history of black chess
         After the agent moves, [0-7] [8-15] exchange, ie [0-7] is the history of black chess and [8-15] is the history of white chess
     !important
         If the white chess moves, then the black chess may be taken by the white chess. This is equivalent to the black chess making no move, but the black chess state has changed.
         Therefore, after the white moves, the state of the black moves must be updated before joining the team again.
         If there is no game taking place, the result is 0 1 and the board is in the same state.
1 line with 1 number indicating who is to move;
1 line with 362 floating numbers, indicating the MCTS probabilities at the end of the search;(last represent pass)
1 line with either 1 or -1, corresponding to the outcome of the self-play game for the player to move.
"""
arg = Args()

DATA_ITEM_LINES = arg.DATA_ITEM_LINES
MINI_BATCH_SIZE = arg.MINI_BATCH_SIZE
MAX_STEP_NUM = arg.MAX_STEP_NUM

filepaths = []
for root, dirs, files in os.walk('./pool_/'):
    for file in files:
        filepaths.append(root + "/" + file)
length = len(filepaths) // 10


class WDataset(Dataset):
    """
     root：Storage address root path
     n < 10
    """

    def __init__(self, n):
        """
        filepaths = []
        for root, dirs, files in os.walk(root):
            for file in files:
                filepaths.append(root + "/" + file)
        random.shuffle(filepaths)
        print('all paths have been loaded!')
        self.path = filepaths
        self.num = len(filepaths)
        self.queue = mp.Queue(4096)
        workers = 4
        print("using {} worker process(es)...".format(workers))
        for list in self.nlists(workers):
            p = mp.Process(target=self.get_samples, args=(list, self.queue))
            p.start()
        """
        self.lines = []
        self.num = 0
        if n == 9:
            self.data_files = filepaths[n * length:]
        else:
            self.data_files = filepaths[n * length: n * length + length]

        for file in tqdm(self.data_files):
            with gzip.open(file, 'r') as f:
                lines = f.readlines()
                self.num += len(lines) // DATA_ITEM_LINES
                self.lines.extend(lines)
        print('load gz!')

    def nlists(self, n):
        length = self.num // n + 1
        for i in range(0, self.num, length):
            if i + length < self.num:
                yield self.path[i:i + length]
            else:
                yield self.path[i:]

    @staticmethod
    def get_samples(chunks, queue):
        while True:
            for chunk in chunks:
                with gzip.open(chunk, 'r') as chunk_file:
                    lines = chunk_file.readlines()
                    sample_count = len(lines) // DATA_ITEM_LINES
                    for index in range(sample_count):
                        sample_index = index * DATA_ITEM_LINES
                        sample = lines[sample_index:sample_index + DATA_ITEM_LINES]
                        # object ==> string
                        str_sample = [str(line, 'ascii') for line in sample]
                        input_planes, probabilities, game_winner = convert(str_sample)
                        queue.put([input_planes, probabilities, game_winner])
        print("DONE!")

    def __getitem__(self, index):
        # [input_planes, probabilities, game_winner] = self.queue.get()

        sample_index = index * DATA_ITEM_LINES
        sample = self.lines[sample_index:sample_index + DATA_ITEM_LINES]
        str_sample = [str(line, 'ascii') for line in sample]
        input_planes, probabilities, game_winner = convert(str_sample)

        return input_planes, probabilities, game_winner[0]

    def __len__(self):
        # Return quantity
        return self.num


def remap(vertex, symmetry):
    """Apply symmetry
    origin: s, rotate: a, reflect: b
    8 conditions: s, a, a*a, a*a*a, b, b*a, b*a*a, b*a*a*a
    """
    assert 0 <= vertex < 361
    x = vertex % 19
    y = vertex // 19
    if symmetry >= 4:
        x, y = y, x
        symmetry -= 4
    if symmetry == 1 or symmetry == 3:
        x = 19 - x - 1
    if symmetry == 2 or symmetry == 3:
        y = 19 - y - 1
    return y * 19 + x


def augmentation(plane, symmetry):
    assert 0 <= symmetry < 8
    work_plane = [0.0] * 361
    # vertex: 0-360
    for vertex in range(0, 361):
        work_plane[vertex] = plane[remap(vertex, symmetry)]
    # len(board_configuration)==361, len(mcts_probabilities)==362
    if len(plane) == 362:
        # pass
        work_plane.append(plane[361])
    return work_plane


def convert(data_item):
    # Convert textual data to python lists.
    board_configuration = []
    colour_to_play = []
    mcts_probabilities = []
    game_winner = []
    input_planes = []

    for plane in range(DATA_ITEM_LINES - 3):
        # 360 first bits are 90 hex chars
        hex_string = data_item[plane][0:90]
        # Convert a hex string to an int
        integer = int(hex_string, 16)
        format_string = format(integer, "0>360b")
        last_bit = data_item[plane][90]
        assert last_bit == "0" or last_bit == "1"
        format_string += last_bit
        plane = [float(char) for char in format_string]
        board_configuration.append(plane)
    assert len(board_configuration) == 16
    input_planes.extend(board_configuration)

    """Two colour_to_play planes
    Either 0 if black is to play or 1 if white is to play.

    The original AlphaGo Zero design has a slight imbalance in that 
    it is easier for the white player to see the board edge
    (due to how padding works in neural networks).
    """
    colour_to_play.append(float(data_item[DATA_ITEM_LINES - 3]))
    assert colour_to_play == [0.0] or colour_to_play == [1.0]
    if colour_to_play == [1.0]:
        input_planes.append([0.0] * 361)
        input_planes.append([1.0] * 361)
    else:
        input_planes.append([1.0] * 361)
        input_planes.append([0.0] * 361)

    # 18 input_planes = 16 board_configuration( [[], [] ,..., []] ) + 2 colour_to_play( [] )
    assert len(input_planes) == 18

    for prob in data_item[DATA_ITEM_LINES - 2].split():
        mcts_probabilities.append(float(prob))
    assert len(mcts_probabilities) == 19 * 19 + 1

    game_winner.append(float(data_item[DATA_ITEM_LINES - 1]))
    assert game_winner == [1.0] or game_winner == [-1.0]

    """Data augmentation
    The rules of Go are invariant under rotation and reflection, this knowledge
    has been used in AlphaGo Zero both by augmenting the dataset during training to
    include rotations and reflections of each position, and to sample random rotations
    or reflections of the position during MCTS.
    """
    symmetry = random.randrange(8)
    # symmetry = 0
    sym_input_planes = [augmentation(plane, symmetry) for plane in input_planes]  # 18x361
    sym_mcts_probabilities = augmentation(mcts_probabilities, symmetry)  # 362

    input_planes = np.zeros((18, 19, 19), dtype='float32')
    for i in range(18):
        for j in range(361):
            input_planes[i, j // 19, j % 19] = sym_input_planes[i][j]
    probabilities = np.array(sym_mcts_probabilities, dtype='float32')

    return input_planes, probabilities, game_winner


if __name__ == '__main__':
    start = time.time()
    trainloader = DataLoader(WDataset(0), batch_size=arg.MINI_BATCH_SIZE,
                             shuffle=True,
                             num_workers=0)
    for batch_idx, (inputs, targets, _) in enumerate(trainloader):
        print(batch_idx)
        end = time.time()
        print(end - start)
