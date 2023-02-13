import numpy as np

flag = False


def dfs(x, y, visit, board):
    global flag
    visit[x][y] = 1  # Indicates that this point has been searched
    directions = [[x - 1, y], [x + 1, y], [x, y - 1],
                  [x, y + 1]]  # Define the four directions up, down, left, and right
    for dx, dy in directions:
        if dx < 0 or dx > 18 or dy < 0 or dy > 18:
            continue  # Skip this direction
        elif visit[dx][dy] == 0 and flag == False:  # Determine whether this point has been searched
            if board[dx][dy] == 0:  # This point is an empty position, that is, the original chess piece is angry, stop searching
                flag = True
                return
            elif board[dx][dy] == -1 * board[x][y]:  # Opposing piece, skip this direction
                continue
            elif board[dx][dy] == board[x][y]:  # Own chess piece, recursive execution
                dfs(dx, dy, visit, board)
    return


def is_alive(x, y, board):  # Determine if you are angry
    global flag
    visit = [[0 for i in range(19)] for j in range(19)]
    flag = False
    dfs(x, y, visit, board)
    return flag


def take_out(enemy, me):  # take
    """
Make a judgment on whether the board is angry or not after each game
Only judge the opponent's pieces
Prevent your own pawns from being picked up by mistake
After the take operation of the opponent's piece is completed,
it is necessary to judge whether or not the own piece is angry.
If there is still no "qi" piece, then it can be asserted that there has been a "suicide" behavior.
	"""
    if np.max(me) == 0:
        print("NULL OP")
        return [[]], True
    suicide_flag = False
    board = np.zeros((19, 19))  # Enemy -1, own 1
    for i in range(19):
        for j in range(19):
            if enemy[i][j] == 1:
                board[i][j] = -1
            elif me[i][j] == 1:
                board[i][j] = 1

    token_list = []
    for i in range(19):
        for j in range(19):
            if board[i][j] == -1 and not (is_alive(i, j, board)):
                token_list.append([i, j])

    for each in token_list:
        board[each[0]][each[1]] = 0
    for i in range(19):
        for j in range(19):
            if board[i][j] == 1 and not (is_alive(i, j, board)):
                suicide_flag = True
                return token_list, suicide_flag
    return token_list, suicide_flag


def judge(enemy, me, color):  # color=0 means your own sunspots, otherwise whites
    board = np.zeros((19, 19))  # Black 1 white 1 empty 0
    if color == 0:
        for i in range(19):
            for j in range(19):
                if enemy[i][j] == 1:
                    board[i][j] = -1
                elif me[i][j] == 1:
                    board[i][j] = 1
    else:
        for i in range(19):
            for j in range(19):
                if enemy[i][j] == 1:
                    board[i][j] = 1
                elif me[i][j] == 1:
                    board[i][j] = -1


def check(board, idx, rob_list, arg):
    """
The act of trying
     @param arg:
     @param board: The board where both sides are to be placed 0-360 [board_self, board_2]
     @param idx: The index of the board to be placed 0<=idx<=361
     @param rob_list: The list of last eaten each in rob_list in [0,360]
     @return: succ_flag new_board rob_list
    """
    ways = arg.ways
    # 361 means not to fall
    if idx == 361:
        return True, board, []
    # Check whether there is a chess piece in 0-360
    if board[0][idx // ways][idx % ways] == 1 or board[1][idx // ways][idx % ways] == 1:
        return False, board, rob_list
    # Try to play chess, check if there is gas and jail
    board[0][idx // ways][idx % ways] = 1
    token_list, suicide_flag = take_out(board[1], board[0])
    # If you have been eaten here in the previous step and you can eat the opponent immediately after the drop, you cannot drop it
    if idx in rob_list and len(token_list) > 0:
        board[0][idx // ways][idx % ways] = 0
        return False, board, rob_list
    if suicide_flag:
        board[0][idx // ways][idx % ways] = 0
        return False, board, rob_list
    else:
        # take
        for each in token_list:
            board[1][each[0]][each[1]] = 0
        if len(token_list)==0:
            rob_list = []
        else:
            rob_list = [each[0] * ways + each[1] for each in token_list]
        return True, [board[0], board[1]], rob_list
