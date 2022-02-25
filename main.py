# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。
box = [[1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, 1], [-1, -1, -1, -1, -1, -1]]
loc = [0, 5]


def find_exit(loc, box):
    if loc[0] == len(box) or loc[1] == -1:
        return loc[1]

    row = loc[0]
    column = loc[1]
    cur = box[row][column]
    if cur == 1:
        if column == len(box[row])-1:
            loc[1] = -1
        else:
            if box[row][column + 1] == 1:
                loc[0] += 1
                loc[1] += 1
            else:
                loc[1] = -1
    else:
        if column == 0:
            loc[1] = -1
        else:
            if box[row][column - 1] == -1:
                loc[0] += 1
                loc[1] -= 1
            else:
                loc[1] = -1
    return find_exit(loc, box)

# for i in range(len(box[0])):
print(find_exit(loc, box))
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
