def minStartValue(nums):
    pre_sum = ans = 1
    for num in nums:
        cur_sum = pre_sum + num
        ans += -cur_sum + 1 if cur_sum < 1 else 0
        pre_sum += -cur_sum + 1 if cur_sum < 1 else 0
        pre_sum += num
    return ans


# nums = [-3, 2, -3, 4, 2]
# nums = [1, 2]
nums = [1, -2, -3]
print(minStartValue(nums))
