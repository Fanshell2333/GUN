# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。

import numpy as np


nums = [2, 3, 1, 4, 0]
diff_nums = np.diff(np.array(nums)).tolist()
diff_nums.append(nums[-1])

for i in range(len(nums)):
    diff_nums.append()

print(diff_nums)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
