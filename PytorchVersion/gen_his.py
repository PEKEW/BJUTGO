import time


def prefix(off, on, winner, editor="腾讯会议", game_name="2021 CCGC") -> str:
    """
    @param game_name: 比赛名称
    @param editor: 打谱者
    @param off: 先手名称 str
    @param on:  后手名称 str
    @param winner: int 1 if 先手胜利 else  0
    @return: str
    """
    pre_str = "(;[GO]"
    off = "[" + off + "]"
    on = "[" + on + "]"
    assert winner == "1" or winner == "0", '参数winner错误 : 应该为(int) 1 if 先手胜利 else  0'
    winner_str = "[先手胜]" if winner == "1" else "[后手胜]"
    time_str = "[" + time.strftime("%Y-%m-%d %H:%M", time.localtime()) + " " + editor + "]"
    game_str = "[" + game_name + "]"
    finnal_str = pre_str + off + on + winner_str + time_str + game_str + ";"
    return finnal_str


def read_file(path="/Users/pkwang/Desktop/1.txt"):
    try:
        f = open(path)
        flow = f.read()
        flow = flow.replace('\n', ';')
        f.close()
        return flow
    except Exception as e:
        print(e)


def ionfix():
    return ")"


def write_file(hist, path="default.txt"):
    try:
        f = open(path, 'w+')
        f.write(hist)
        f.close()
    except Exception as e:
        print(e)
        exit(-1)
    print("棋谱已生成, 详见{}".format(path))


if __name__ == '__main__':
    off = input("先手队伍名称: ")
    on = input("后手队伍名称: ")
    winner = input("先手胜输入1， 否则0: ")
    # editor = input("比赛地点: ")
    path = input("日志路径: ")
    hist = prefix(off, on, winner) + read_file(path) + ionfix()
    out_str = off + " vs " + on + "-" + ["后", "先"][int(winner)] + "手胜.txt"
    write_file(hist, path=out_str)
