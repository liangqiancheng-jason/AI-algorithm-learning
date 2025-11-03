class Solution(object):
    """
    69. x 的平方根
    """

    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        if x <= 1:
            return x
        left, right = 1, x
        ans = 1
        while left <= right:
            mid = (left + right) // 2
            if mid * mid <= x:
                if mid * mid == x:
                    ans = mid
                left = mid + 1
            else:
                right = mid - 1
        return ans

    """
    34. 在排序数组中查找元素的第一个和最后一个位置
    """

    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """

        # 二分模板：找第一个 >= x 的位置
        def searchLeft(nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
            lo, hi = 0, len(nums) - 1
            ans = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] >= target:
                    if nums[mid] == target:
                        ans = mid  # 记录命中，但继续往左收缩
                    hi = mid - 1  # 收缩到左侧，保证是“第一次出现”
                else:  # nums[mid] < target
                    lo = mid + 1
            return ans

        def searchRight(nums, target):
            """
            :type nums: List[int]
            :type target: int
            :rtype: List[int]
            """
            lo, hi = 0, len(nums) - 1
            ans = -1
            while lo <= hi:
                mid = (lo + hi) // 2
                if nums[mid] <= target:
                    if nums[mid] == target:
                        ans = mid  # 记录命中，但继续往右找更晚的位置
                    lo = mid + 1  # 向右收缩
                else:  # nums[mid] > target
                    hi = mid - 1  # 向左收缩
            return ans

        if not nums:
            return [-1, -1]
        left = searchLeft(nums, target)
        # 若越界或不是目标值，说明不存在
        if left == len(nums) or nums[left] != target:
            return [-1, -1]

        right = searchRight(nums, target)
        return [left, right]

    """
    35. 搜索插入位置
    """
    def searchInsert(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (right - left) // 2 + left
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return left
    """
    704. 二分查找
    """
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left, right = 0, len(nums) - 1
        while left <=right:
            mid = (right - left) // 2 + left
            if nums[mid] == target:
                return mid
            elif nums[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1