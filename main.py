# 这是一个示例 Python 脚本。

# 按 Shift+F10 执行或将其替换为您的代码。
# 按 双击 Shift 在所有地方搜索类、文件、工具窗口、操作和设置。


nums = [9, 4, 1, 7, 6, 4]
k = 3


def minimumDifference(nums, k):
    nums = sorted(nums)
    ans = nums[k - 1] - nums[0]
    for i in range(k, len(nums)):
        right = nums[i - k + 1]
        left = nums[i]
        ans = min(ans, left - right)
    return ans

minimumDifference(nums, k)
# 访问 https://www.jetbrains.com/help/pycharm/ 获取 PyCharm 帮助
