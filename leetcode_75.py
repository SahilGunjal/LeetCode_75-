# This file contains 75 questions asked by companies in their coding interviews and their solutions. (Leetcode - 75)
# Author: Sahil Sanjay Gunjal.
# language: Python
# Note: Anyone can see and get the codes from this file. It's completely free and do share your best solutions if you
#       find any. Motive of this is to contribute to the community and enhance the learning opportunity for everyone.

"""
----------------------------------  Array and Strings  ----------------------------------
"""
"""
Question 1:
You are given two strings word1 and word2. Merge the strings by adding letters in alternating order, starting with
word1. If a string is longer than the other, append the additional letters onto the end of the merged string.
Return the merged string.
"""

class Solution:
    def mergeAlternately(self, word1: str, word2: str) -> str:
        len_1 = len(word1)
        len_2 = len(word2)
        str = ''
        if len_1 <= len_2:
            ptr2 = 0
            for i in range(len_1):
                str += word1[i]
                str += word2[ptr2]
                ptr2 += 1

            str = str + word2[ptr2:]

        else:
            ptr1 = 0
            for i in range(len_2):
                str += word1[ptr1]
                str += word2[i]
                ptr1 += 1

            str = str + word1[ptr1:]

        return str


"""
Question 2:
For two strings s and t, we say "t divides s" if and only if s = t + ... + t (i.e., t is concatenated with itself one or
 more times). Given two strings str1 and str2, return the largest string x such that x divides both str1 and str2.
"""

class Solution:
    def gcdOfStrings(self, str1: str, str2: str) -> str:
        l1, l2 = len(str1), len(str2)

        def checkGCD(l, str):
            if l1 % l != 0 or l2 % l != 0:
                return False

            temp1, temp2 = l1 // l, l2 // l

            if str1 == temp1 * str[:l] and str2 == temp2 * str[:l]:
                return True

        if l1 >= l2:
            for i in range(l2, 0, -1):
                if checkGCD(i, str2):
                    return str2[:i]

        else:
            for i in range(l1, 0, -1):
                if checkGCD(i, str1):
                    return str1[:i]

        return ""

"""
Question 3:
There are n kids with candies. You are given an integer array candies, where each candies[i] represents the number of 
candies the ith kid has, and an integer extraCandies, denoting the number of extra candies that you have.

Return a boolean array result of length n, where result[i] is true if, after giving the ith kid all the extraCandies, 
they will have the greatest number of candies among all the kids, or false otherwise.

Note that multiple kids can have the greatest number of candies.
"""
class Solution:
    def kidsWithCandies(self, candies: List[int], extraCandies: int) -> List[bool]:
        greatest_number = max(candies)
        total_students = len(candies)
        Ans = []
        for i in range(total_students):
            temp_candies = candies[i] + extraCandies
            if temp_candies >= greatest_number:
                Ans.append(True)
            else:
                Ans.append(False)

        return Ans

"""
Question 4:
You have a long flowerbed in which some of the plots are planted, and some are not. However, flowers cannot be planted 
in adjacent plots. Given an integer array flowerbed containing 0's and 1's, where 0 means empty and 1 means not empty, 
and an integer n, return true if n new flowers can be planted in the flowerbed without violating the no-adjacent-flowers
rule and false otherwise.
"""

class Solution:
    def canPlaceFlowers(self, flowerbed: List[int], n: int) -> bool:
        array_size = len(flowerbed)
        temp = n
        if n == 0:
            return True
        if array_size == 1 and flowerbed[0] == 0 and n == 1:
            return True

        if array_size > 1:
            if flowerbed[0] == 0 and flowerbed[1] == 0:
                flowerbed[0] = 1
                temp -= 1

            if temp == 0:
                return True

            for i in range(1, array_size - 1):
                if flowerbed[i] == 0:
                    if flowerbed[i - 1] == 0 and flowerbed[i + 1] == 0:
                        flowerbed[i] = 1
                        temp -= 1

                if temp == 0:
                    return True

            if flowerbed[array_size - 1] == 0 and flowerbed[array_size - 2] == 0 and temp == 1:
                return True

        return False

"""
Question 5:
Given a string s, reverse only all the vowels in the string and return it.
The vowels are 'a', 'e', 'i', 'o', and 'u', and they can appear in both lower and upper cases, more than once.
"""
class Solution:
    def reverseVowels(self, s: str) -> str:
        temp = []
        temp2 = [x for x in s]
        str_size = len(s)
        dict_set = {'a','e','i','o','u','A','E','O','I','U'}
        for i in range(str_size):
            if s[i] in dict_set:
                temp.append(s[i])

        for i in range(str_size):
            if s[i] in dict_set:
                temp2[i] = temp.pop()

        return ''.join(temp2)

"""
Question 6:
Given an input string s, reverse the order of the words.
A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.
Return a string of the words in reverse order concatenated by a single space.
Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only
have a single space separating the words. Do not include any extra spaces.
"""
class Solution:
    def reverseWords(self, s: str) -> str:
        temp = s.split()
        st = ''
        for i in range(len(temp)-1,-1,-1):
            if temp[i] != ' ':
                st = st + temp[i]
            if i != 0:
                st = st + ' '

        return st

"""
Question 7:
(Intuition: Calculate the pre and postfix of the number and then do the calculation)
Given an integer array nums, return an array answer such that answer[i] is equal to the product of all the elements of 
nums except nums[i]. The product of any prefix or suffix of nums is guaranteed to fit in a 32-bit integer.
You must write an algorithm that runs in O(n) time and without using the division operation.
"""
class Solution:
    def productExceptSelf(self, nums: List[int]) -> List[int]:
        size = len(nums)
        output = [0] * size
        pre = 1
        post = 1
        for i in range(size):
            if i == 0:
                output[i] = 1
            else:
                output[i] = nums[i - 1] * pre

            pre = output[i]

        for i in range(size - 1, -1, -1):
            if i == size - 1:
                post = 1
            else:
                post = nums[i + 1] * post
                output[i] = post * output[i]

        return output

"""
Question 8:
Increasing Triplet Subsequence:
Given an integer array nums, return true if there exists a triple of indices (i, j, k) such that 
i < j < k and nums[i] < nums[j] < nums[k]. If no such indices exists, return false.
"""
# Brute Force:O(n3)
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        for i in range(len(nums)):
            for j in range(i+1,len(nums)):
                if nums[j] > nums[i]:
                    for k in range(j+1,len(nums)):
                        if nums[k] > nums[j]:
                            return True

        return False

# O(n) Solution:
class Solution:
    def increasingTriplet(self, nums: List[int]) -> bool:
        num_size = len(nums)
        if num_size < 3:
            return False

        left = sys.maxsize
        mid = sys.maxsize

        for i in range(num_size):
            if nums[i] > mid:
                return True
            elif nums[i] < left:
                left = nums[i]
            elif nums[i] < mid and nums[i] > left:
                mid = nums[i]

        return False


"""
Question 9 : (String Comparison)
Given an array of characters chars, compress it using the following algorithm:
Begin with an empty string s. For each group of consecutive repeating characters in chars:
If the group's length is 1, append the character to s.
Otherwise, append the character followed by the group's length.
The compressed string s should not be returned separately, but instead, be stored in the input character array chars. 
Note that group lengths that are 10 or longer will be split into multiple characters in chars.
After you are done modifying the input array, return the new length of the array.
You must write an algorithm that uses only constant extra space.

Intuition: calculate the group one by one using while if the counter > 1 then again separate it and add to the chars
"""

class Solution:
    def compress(self, chars: List[str]) -> int:
        ans = 0
        char_size = len(chars)
        if char_size == 1:
            return 1

        i = 0
        while (i < char_size):
            j = i + 1
            while (j < char_size and chars[j] == chars[i]):
                j = j + 1

            count = j - i
            chars[ans] = chars[i]
            ans += 1

            if count > 1:
                char_arr = list(str(count))
                for char in char_arr:
                    chars[ans] = char
                    ans += 1

            i = j

        return ans

"""
----------------------------------  TWO POINTERS ---------------------------------- 
"""

"""
Question 10:(Move zeros)
Given an integer array nums, move all 0's to the end of it while maintaining the relative order of the non-zero elements
Note that you must do this in-place without making a copy of the array.
"""

class Solution:
    def moveZeroes(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        ind = 0
        size = len(nums)
        for i in range(size):
            if nums[i] != 0:
                nums[i], nums[ind] = nums[ind], nums[i]
                ind += 1


"""
Question 11: (Is Subsequence)
Given two strings s and t, return true if s is a subsequence of t, or false otherwise.
A subsequence of a string is a new string that is formed from the original string by deleting some (can be none) of the 
characters without disturbing the relative positions of the remaining characters. (i.e., "ace" is a subsequence of 
"abcde" while "aec" is not).
"""

class Solution:
    def isSubsequence(self, s: str, t: str) -> bool:
        s_ptr = 0
        size_t = len(t)
        size_s = len(s)
        if size_s == 0:
            return True

        for i in range(size_t):
            if s[s_ptr] == t[i]:
                s_ptr += 1

            if s_ptr == size_s:
                return True

        return False


"""
Question 12: (Container with most water)
You are given an integer array height of length n. There are n vertical lines drawn such that the two endpoints of the 
ith line are (i, 0) and (i, height[i]). Find two lines that together with the x-axis form a container, such that the 
container contains the most water.Return the maximum amount of water a container can store.
Notice that you may not slant the container.
Intuition: Brute force need O(n2) but using 2 ptr it is ~ O(n). Move ptr with small height and calculate the capacity.
"""

class Solution:
    def calculate_capacity(self, f, l, arr):
        x_diff = l - f
        y_diff = min(arr[f], arr[l])

        return x_diff * y_diff

    def maxArea(self, height: List[int]) -> int:
        arr_size = len(height)
        f_ptr = 0
        l_ptr = arr_size - 1
        max_capacity = float('-inf')
        while f_ptr < l_ptr:
            curr_capacity = self.calculate_capacity(f_ptr, l_ptr, height)
            if curr_capacity > max_capacity:
                max_capacity = curr_capacity

            if height[f_ptr] <= height[l_ptr]:
                f_ptr += 1
            else:
                l_ptr -= 1

        return max_capacity


"""
Question 13: (Max Number of K-Sum Pairs)
You are given an integer array nums and an integer k.
In one operation, you can pick two numbers from the array whose sum equals k and remove them from the array.
Return the maximum number of operations you can perform on the array.
Intuition: Can be done in O(n2) using brute force, better way can be done using sorting the array and then using 2 ptrs
O(nlogn). But using Counter --> hashmap , can be done using O(N) = Best Soltion ans = ans + min(h_map[item],h_map[k-item])
and then final ans//2, as it counts the items twice.
"""

class Solution:
    def maxOperations(self, nums: List[int], k: int) -> int:
        h_map = Counter(nums)
        max_count = 0
        for items in h_map:
            max_count += min(h_map[items], h_map[k - items])

        return max_count // 2


"""
----------------------------------  Sliding Window  ----------------------------------
"""

"""
Question 14: Maximum Average Subarray I
You are given an integer array nums consisting of n elements, and an integer k.
Find a contiguous subarray whose length is equal to k that has the maximum average value and return this value. 
Any answer with a calculation error less than 10-5 will be accepted.

Intuition: Can be done in O(nk) in brute force way but it is a problem of fix size window (sliding window) here we have 
given arr, find subarray, avg and given window size as well = k. So can be solved in O(n) using sliding window. 
"""

class Solution:
    def findMaxAverage(self, nums: List[int], k: int) -> float:
        max_avg = float('-inf')
        num_size = len(nums)
        initial_sum = 0
        for i in range(k):
            initial_sum += nums[i]

        if initial_sum / k > max_avg:
            max_avg = initial_sum / k

        for i in range(k, num_size):
            initial_sum = initial_sum + nums[i] - nums[i - k]
            if initial_sum / k > max_avg:
                max_avg = initial_sum / k

        return max_avg




















