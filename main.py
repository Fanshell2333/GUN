# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


matrix = [[76618,42558,65788,20503,29400,54116]]


result = []
if len(matrix) == 1:
    result = [min(matrix[0])]

for i in range(len(matrix)):
    min_idx = matrix[i].index(min(matrix[i]))
    flag = False
    for j in range(len(matrix)):
        if i != j:
            if matrix[i][min_idx] <= matrix[j][min_idx]:
                flag = False
                break
            else:
                flag = True
    if flag:
        result.append(matrix[i][min_idx])


print(result)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
