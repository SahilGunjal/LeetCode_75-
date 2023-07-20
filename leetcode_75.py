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

Note:If you are using something repeatedly then it can be optimized. So look for such cases. Same in case of sliding 
window problems check for these conditions and use sliding window whenever possible. 
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


"""
Question 15: Maximum Number of Vowels in a Substring of Given Length
Given a string s and an integer k, return the maximum number of vowel letters in any substring of s with length k.
Vowel letters in English are 'a', 'e', 'i', 'o', and 'u'.
Intuition : Again it is a classic problem of sliding window here we can see need to find the substring, k is given.
Keep the window of size 3 and move and update max_vowel  
"""

class Solution:
    def maxVowels(self, s: str, k: int) -> int:
        s_size = len(s)
        max_vowel = 0
        temp = 0
        for i in range(k):
            if s[i] == 'a' or s[i] == 'e' or s[i] == 'i' or s[i] == 'o' or s[i] == 'u':
                temp += 1

        if temp > max_vowel:
            max_vowel = temp

        for i in range(k, s_size):
            if s[i - k] == 'a' or s[i - k] == 'e' or s[i - k] == 'i' or s[i - k] == 'o' or s[i - k] == 'u':
                temp -= 1

            if s[i] == 'a' or s[i] == 'e' or s[i] == 'i' or s[i] == 'o' or s[i] == 'u':
                temp += 1

            if temp > max_vowel:
                max_vowel = temp

        return max_vowel


"""
Question 16: Max Consecutive Ones III
Given a binary array nums and an integer k, return the maximum number of consecutive 1's in the array if you can 
flip at most k 0's.
Intuition : This one is tricky sliding window problem, here we have not given any fix size of a window. As per the 
condition we have to move the sliding window. We use 2 pointers here to manage this. (we kept number of 0's <= k
"""

class Solution:
    def longestOnes(self, nums: List[int], k: int) -> int:
        i = 0
        j = 0
        nums_size = len(nums)
        long_ones = 0
        k_count = 0
        while j < nums_size:
            if nums[j] == 0:
                k_count += 1

            if k_count > k:
                while k_count > k:
                    if nums[i] == 0:
                        k_count -= 1

                    i += 1

            if j - i + 1 > long_ones:
                long_ones = j - i + 1

            j += 1

        return long_ones

"""
Practice Problem of Sliding window: (Not in 75 Questions) : gfg: Longest Sub-Array with Sum K
Given an array containing N integers and an integer K., Your task is to find the length of the longest Sub-Array with 
the sum of the elements equal to the given value K. (Only Positive)
"""

class Solution:
    def lenOfLongSubarr(self, arr, n, k):
        # Complete the function

        i = 0
        j = 0
        sum = 0
        long_subarr = 0
        arr_size = len(arr)
        while (j < arr_size):
            sum += arr[j]

            if sum < k:
                j += 1

            elif sum == k:
                if j - i + 1 > long_subarr:
                    long_subarr = j - i + 1

                j += 1

            elif sum > k:
                while sum > k:
                    if arr[i] < 0:
                        sum += arr[i]
                    else:
                        sum -= arr[i]

                    i += 1

                j += 1

        return long_subarr


"""
If we have both positive and negative numbers and as well as zeros:
(solved using reverse engineering 
"""
class Solution:
    def lenOfLongSubarr(self, arr, n, k):
        # Complete the function
        long_sum = 0
        dic = dict()
        sum = 0
        arr_size = len(arr)
        for i in range(arr_size):
            sum += arr[i]

            if sum == k:
                long_sum = max(long_sum, i + 1)

            rem = sum - k

            if rem in dic:
                long_sum = max(long_sum, i - dic[rem])

            if sum not in dic:
                dic[sum] = i

        return long_sum

"""
Question 17: Longest Subarray of 1's After Deleting One Element
Given a binary array nums, you should delete one element from it.
Return the size of the longest non-empty subarray containing only 1's in the resulting array. Return 0 if there is no 
such subarray.
Intuition: It is again a sliding window problem and here we can maintain 2 pointers and i and j and same as the previous 
consecutive ones we can maintain k not greater than 1 as only 1 element must be there. if 1101011 is an example then j 
goes till the 4 and then again reduce k by moving i till k==1. then I will be at 3 and k = 1 and j=5. and calculate the
long_subarray each step. So output of this case will be 3.===> either i=0 and j = 4 (4-0-1) or i=3 and j = 6 but j==7
at the end of loop and (7-3-1) = 3  
"""

class Solution:
    def longestSubarray(self, nums: List[int]) -> int:
        i = 0
        j = 0
        nums_size = len(nums)
        long_sub = 0
        k = 0
        while (j < nums_size):
            if nums[j] == 0:
                k += 1

                if k > 1 and nums[j] == 0:
                    long_sub = max(long_sub, j - i - 1)

                    while (k > 1):
                        if nums[i] == 0:
                            k -= 1

                        i = i + 1

            j += 1

        long_sub = max(long_sub, j - i - 1)

        return long_sub


"""
----------------------------------  Prefix Sum  ----------------------------------
"""

"""
Question 18: 1732. Find the Highest Altitude
There is a biker going on a road trip. The road trip consists of n + 1 points at different altitudes. The biker starts 
his trip on point 0 with altitude equal 0. You are given an integer array gain of length n where gain[i] is the net gain
 n altitude between points i and i + 1 for all (0 <= i < n). Return the highest altitude of a point.
Intuition: Very simple, just see at by adding each point whether highest altitude increases. consider Y axis and keep 
adding and checking. 
"""
class Solution:
    def largestAltitude(self, gain: List[int]) -> int:
        highest_altitude = 0
        temp_alti = 0
        total_points = len(gain)
        for i in range(total_points):
            temp_alti += gain[i]
            highest_altitude = max(highest_altitude, temp_alti)

        return highest_altitude


"""
Question 19:  Find Pivot Index
Given an array of integers nums, calculate the pivot index of this array.
The pivot index is the index where the sum of all the numbers strictly to the left of the index is equal to the sum of 
all the numbers strictly to the index's right. If the index is on the left edge of the array, then the left sum is 0 
because there are no elements to the left. This also applies to the right edge of the array. Return the leftmost pivot 
index. If no such index exists, return -1.
Intuition: 
approach: o(n): space o(2n): time : store sum in array
Best: o(1): space o(2n): time : maintain left_sum and right_sum pointers
"""

# Approach:
class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        my_sum_arr = []
        sum = 0
        for i in range(len(nums)):
            sum = sum + nums[i]
            my_sum_arr.append(sum)

        if my_sum_arr[len(nums) - 1] - nums[0] == 0:
            return 0

        for i in range(1, len(my_sum_arr)):
            temp2 = my_sum_arr[len(nums) - 1] - my_sum_arr[i]

            if my_sum_arr[i - 1] == temp2:
                return i

        return -1

# Best Solution:

class Solution:
    def pivotIndex(self, nums: List[int]) -> int:
        lsum = 0
        rsum = sum(nums)

        for i in range(len(nums)):
            rsum = rsum - nums[i]

            if lsum == rsum:
                return i

            lsum = lsum + nums[i]

        return -1



"""
----------------------------------  Hashmap  ----------------------------------
"""

"""
Question 20: 2215. Find the Difference of Two Arrays
Given two 0-indexed integer arrays nums1 and nums2, return a list answer of size 2 where:
answer[0] is a list of all distinct integers in nums1 which are not present in nums2.
answer[1] is a list of all distinct integers in nums2 which are not present in nums1.
Note that the integers in the lists may be returned in any order.

Intuition: put the nums in 2 sets(hash) and check whether that element is in other set or not
"""

class Solution:
    def findDifference(self, nums1: List[int], nums2: List[int]) -> List[List[int]]:
        s1 = set(nums1)
        s2 = set(nums2)
        ans = [[] for _ in range(2)]
        for num in s1:
            if num not in s2:
                ans[0].append(num)

        for num in s2:
            if num not in s1:
                ans[1].append(num)

        return ans

"""
question 21: Unique Number of Occurrences
Given an array of integers arr, return true if the number of occurrences of each value in the array is unique or false 
otherwise.
2 approaches: 1. With more time complexity
"""
# Naive
class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        num_count = set()

        for num in arr:
            num_count.add(arr.count(num))

        if len(num_count) == len(set(arr)):
            return True

        else:
            return False


# Optimized

class Solution:
    def uniqueOccurrences(self, arr: List[int]) -> bool:
        count_dict = dict()

        for num in arr:
            if num not in count_dict:
                count_dict[num] = 1

            else:
                count_dict[num] += 1

        if len(set(count_dict.values())) == len(set(arr)):
            return True

        else:
            return False


"""
question 22: Determine if Two Strings Are Close
Two strings are considered close if you can attain one from the other using the following operations:

Operation 1: Swap any two existing characters.
For example, abcde -> aecdb
Operation 2: Transform every occurrence of one existing character into another existing character, 
and do the same with the other character.
For example, aacabb -> bbcbaa (all a's turn into b's, and all b's turn into a's)
You can use the operations on either string as many times as necessary.
Given two strings, word1 and word2, return true if word1 and word2 are close, and false otherwise.
Intuition: We need same characters in both the words + and their number of count of characters must be same in both
the cases not necessarily count of same characters. 
"""

class Solution:
    def closeStrings(self, word1: str, word2: str) -> bool:
        if len(word1) != len(word2):
            return False

        first_dict = dict()
        second_dict = dict()

        for char in word1:
            if char not in first_dict:
                first_dict[char] = 1

            else:
                first_dict[char] += 1

        for char in word2:
            if char not in second_dict:
                second_dict[char] = 1

            else:
                second_dict[char] += 1

        if set(word1) == set(word2) and set(first_dict.values()) == set(second_dict.values()):
            x = list(first_dict.values())
            y = list(second_dict.values())
            x.sort()
            y.sort()
            if x == y:
                return True

        else:
            return False


"""
Question 23: 2352. Equal Row and Column Pairs
Given a 0-indexed n x n integer matrix grid, return the number of pairs (ri, cj) such that row ri and column cj are equal.
A row and column pair is considered equal if they contain the same elements in the same order (i.e., an equal array).

Intuition: Here we can use logic of transpose and hashtable, store the row as string key in hashtable and then check if 
transpose of column is in dictionary/ hashtable if yes increase the counter. 
"""

class Solution:
    def equalPairs(self, grid: List[List[int]]) -> int:
        n = len(grid[0])
        trans_matrix_dict = dict()

        for rows in grid:
            temp_str = str(rows)
            if temp_str in trans_matrix_dict:
                trans_matrix_dict[temp_str] += 1

            else:
                trans_matrix_dict[temp_str] = 1

        count = 0
        for i in range(n):
            temp = []
            for j in range(n):
                temp.append(grid[j][i])

            temp_str = str(temp)
            if temp_str in trans_matrix_dict:
                count += trans_matrix_dict[temp_str]

        return count


"""
----------------------------------  Stack  ----------------------------------
"""

"""
Question 24: Removing Stars From a String
You are given a string s, which contains stars *.
In one operation, you can:
Choose a star in s.
Remove the closest non-star character to its left, as well as remove the star itself.
Return the string after all stars have been removed.

Note:
The input will be generated such that the operation is always possible.
It can be shown that the resulting string will always be unique.

Intuition: Naive + Better
"""

# Naive : Solve using just strings... time complexity : O(n) space: O(1)

class Solution:
    def removeStars(self, s: str) -> str:
        final_string = ''
        i = len(s) - 1
        s_count = 0
        while (i >= 0):
            if s_count > 0:
                if s[i] != '*':
                    s_count -= 1
                else:
                    s_count += 1

                i -= 1

            elif s[i] == '*':
                s_count += 1
                i -= 1

            elif s[i] != '*':
                final_string += s[i]
                i -= 1

        return final_string[::-1]


# Using Stack : Optimized time but used space complexity of O(n) = stack

class Solution:
    def removeStars(self, s: str) -> str:
        final_ans = deque()
        for char in s:
            if char == '*':
                final_ans.pop()
            else:
                final_ans.append(char)

        return "".join(final_ans)


"""
Question 25: 735. Asteroid Collision
We are given an array asteroids of integers representing asteroids in a row.
For each asteroid, the absolute value represents its size, and the sign represents its direction (positive meaning right
, negative meaning left). Each asteroid moves at the same speed.Find out the state of the asteroids after all collisions.
If two asteroids meet, the smaller one will explode. If both are the same size, both will explode. 
Two asteroids moving in the same direction will never meet.

Tricky Question: Conditions are easy but we have to think in better way. Use while loop correctly and break the loop 
whenever necessary.
"""
class Solution:
    def asteroidCollision(self, asteroids: List[int]) -> List[int]:
        final_asteroid = []

        for astro in asteroids:

            while len(final_asteroid) > 0 and final_asteroid[-1] > 0 and astro < 0:

                if abs(astro) == final_asteroid[-1]:
                    final_asteroid.pop()
                    break

                elif abs(astro) > final_asteroid[-1]:
                    final_asteroid.pop()
                    continue

                elif abs(astro) < final_asteroid[-1]:
                    break

            else:
                final_asteroid.append(astro)

        return final_asteroid

"""
Question 26: 394. Decode String
Given an encoded string, return its decoded string.
The encoding rule is: k[encoded_string], where the encoded_string inside the square brackets is being repeated exactly 
k times. Note that k is guaranteed to be a positive integer. You may assume that the input string is always valid; 
there are no extra white spaces, square brackets are well-formed, etc. Furthermore, you may assume that the original 
data does not contain any digits and that digits are only for those repeat numbers, k. For example, there will not be 
input like 3a or 2[4].
The test cases are generated so that the length of the output will never exceed 105.

Intuition: Tricky Question, look for the all conditions, [, ] , digit, char, use proper condition which are required.
It is leetcode medium question but not less than leetcode hard
"""

class Solution:
    def decodeString(self, s: str) -> str:
        str_stack = []
        num_stack = []
        num = 0
        ans = ''
        temp = ''

        for char in s:
            if char.isdigit():
                num = num * 10 + int(char)

            elif char == '[':
                num_stack.append(num)
                str_stack.append(ans)
                ans = ''
                num = 0

            elif char == ']':
                temp = ans
                ans = str_stack.pop()
                t_num = num_stack.pop()
                while (t_num > 0):
                    ans = ans + temp
                    t_num -= 1

            else:
                ans += char

        return ans

"""
----------------------------------  Queue  ----------------------------------
"""

"""
Question 27: Number of Recent Calls
You have a RecentCounter class which counts the number of recent requests within a certain time frame.

Implement the RecentCounter class:
RecentCounter() Initializes the counter with zero recent requests.
int ping(int t) Adds a new request at time t, where t represents some time in milliseconds, and returns the number of 
requests that has happened in the past 3000 milliseconds (including the new request). Specifically, return the number 
of requests that have happened in the inclusive range [t - 3000, t].
It is guaranteed that every call to ping uses a strictly larger value of t than the previous call.
"""

class RecentCounter:

    def __init__(self):
        self.que = deque()

    def ping(self, t: int) -> int:
        self.que.append(t)

        start_time = t - 3000

        while self.que[0] < start_time:
            self.que.popleft()

        return len(self.que)

# Your RecentCounter object will be instantiated and called as such:
# obj = RecentCounter()
# param_1 = obj.ping(t)

"""
Question 28: Dota2 Senate

Intuition: Use of 2 queues and add whatever is removing to the end as per the index of the R or D.
"""


class Solution:
    def predictPartyVictory(self, senate: str) -> str:
        senate = list(senate)
        d_que = deque()
        r_que = deque()

        for i in range(len(senate)):
            if senate[i] == 'D':
                d_que.append(i)
            else:
                r_que.append(i)

        while d_que and r_que:
            if d_que[0] < r_que[0]:
                d_que.append(d_que.popleft() + len(senate))
                r_que.popleft()
            else:
                r_que.append(r_que.popleft() + len(senate))
                d_que.popleft()

        if len(d_que) != 0:
            return 'Dire'
        else:
            return 'Radiant'

"""
----------------------------------  Linked-List  ----------------------------------
"""

"""
Question 29: 2095: Delete the Middle Node of a Linked List
Intuition: Good_Approach: count the total elements then go for deleting the middle element by index : time complexity: 
O(n + n/2) 
Better Approach: Maintain two pointers left and right : move right by 2 and left by 1 and once left reach the none, 
slow is just before the middle skip it and return the head. time complexity: O(n/2)
"""
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.
#         next = next

# GOOD Apporach
class Solution:
    def calculate_nodes(self, node):
        count = 0
        while node:
            count += 1
            node = node.next

        return count

    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        count = self.calculate_nodes(head)

        if count == 1 or count == 0:
            return None

        remove_index = count // 2
        i = 0
        node = head
        while i < remove_index - 1:
            node = node.next
            i += 1

        node.next = node.next.next

        return head


# Best Approach: Here the catch is only that look for if initially there is just one element then we have to return None
# depend on the where fast.next is None or fast.next.next is None we have to use prev and slow condition
# if after loop fast.next is None means slow is on the middle so use prev, (odd)
# else if fast.next.next is None means slow is before the middle so use slow (even)


# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteMiddle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        fast = head
        slow = head

        if fast.next is None:
            return None

        while fast.next and fast.next.next:
            prev = slow
            slow = slow.next
            fast = fast.next.next

        if fast.next is None:
            prev.next = slow.next

        else:
            slow.next = slow.next.next

        return head


"""
Question 30: 328. Odd Even Linked List
Given the head of a singly linked list, group all the nodes with odd indices together followed by the nodes with even 
indices, and return the reordered list. The first node is considered odd, and the second node is even, and so on.
Note that the relative order inside both the even and odd groups should remain as it was in the input.
You must solve the problem in O(1) extra space complexity and O(n) time complexity.
"""

# Approach 1:
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        odd_ptr = head
        odd_head = odd_ptr
        even_ptr = None

        if odd_ptr is None:
            return None

        if odd_ptr.next is None:
            return odd_head

        if odd_ptr.next.next is None:
            return odd_head

        even_head = head.next

        while odd_ptr.next and odd_ptr.next.next:
            if even_ptr is None:
                even_ptr = even_head
            else:
                even_ptr.next = even_ptr.next.next
                even_ptr = even_ptr.next

            odd_ptr.next = odd_ptr.next.next
            odd_ptr = odd_ptr.next

        if odd_ptr.next is None:
            even_ptr.next = None
        elif odd_ptr.next.next is None:
            even_ptr.next = even_ptr.next.next

        odd_ptr.next = even_head
        return odd_head

# Approach 2:
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def oddEvenList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        odd_ptr = head
        odd_head = odd_ptr
        if odd_ptr is None:
            return None
        even_ptr = head.next
        even_head = even_ptr
        if even_ptr is None:
            return odd_ptr

        while odd_ptr.next and even_ptr.next:
            odd_ptr.next = even_ptr.next
            odd_ptr = odd_ptr.next

            even_ptr.next = odd_ptr.next
            even_ptr = even_ptr.next

        odd_ptr.next = even_head

        return odd_head

"""
Question 31: Reverse the LinkedList
Approach 1:  using 3 pointers O(N) Time O(1) Space
Approach 2: using another datastructure (stack or array) Time O(2n) and space O(n)
"""

# Approach 1 is best:

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        prev = None
        curr = head
        while curr:
            new = curr.next
            curr.next = prev
            prev = curr
            curr = new

        return prev


"""
Question 32: 2130: Maximum Twin Sum of a Linked List
In a linked list of size n, where n is even, the ith node (0-indexed) of the linked list is known as the twin of the 
(n-1-i)th node, if 0 <= i <= (n / 2) - 1. For example, if n = 4, then node 0 is the twin of node 3, and node 1 is the 
twin of node 2. These are the only nodes with twins for n = 4. The twin sum is defined as the sum of a node and its twin.
Given the head of a linked list with even length, return the maximum twin sum of the linked list.

Intuition: Reverse the 2nd half of the linkedlist and then traverse both the half's simultaneously check the sum and 
store the maximum sum return it after the loop ends.
"""

# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, node):
        prev = None
        curr = node
        while curr:
            new = curr.next
            curr.next = prev
            prev = curr
            curr = new

        return prev

    def pairSum(self, head: Optional[ListNode]) -> int:

        if head:
            forHead = head
            slow = head
            fast = head

            while fast.next.next:
                slow = slow.next
                fast = fast.next.next

            revHead = self.reverseList(slow.next)

        maxSum = 0
        while revHead:
            maxSum = max(maxSum, revHead.val + forHead.val)
            revHead = revHead.next
            forHead = forHead.next

        return maxSum


"""
----------------------------------  Binary-Tree-DFS  ----------------------------------
"""

"""
Question 33: 104. Maximum Depth of Binary Tree
Given the root of a binary tree, return its maximum depth.
A binary tree's maximum depth is the number of nodes along the longest path from the root node down to the farthest leaf 
node.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def calculateTheHeight(self,node):
        if node is None:
            return 0

        leftHeight = self.calculateTheHeight(node.left)
        rightHeight = self.calculateTheHeight(node.right)

        return 1 + max(leftHeight,rightHeight)

    def maxDepth(self, root: Optional[TreeNode]) -> int:
        return self.calculateTheHeig


"""
Question 34 : Leaf-Similar Trees
Consider all the leaves of a binary tree, from left to right order, the values of those leaves form a leaf value 
sequence.
Intuition:  Traverse each tree find all leaf nodes and compare if same return True else return False
TC: O(2N) and Space ~ O(2N) 
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findLeefNodes(self, node, l):
        if node is None:
            return
        self.findLeefNodes(node.left, l)
        self.findLeefNodes(node.right, l)

        if node.left is None and node.right is None:
            l.append(node.val)

    def leafSimilar(self, root1: Optional[TreeNode], root2: Optional[TreeNode]) -> bool:
        l1 = []
        l2 = []
        self.findLeefNodes(root1, l1)
        self.findLeefNodes(root2, l2)

        if l1 == l2:
            return True
        else:
            return False

"""
Question 35: 1448. Count Good Nodes in Binary Tree
Given a binary tree root, a node X in the tree is named good if in the path from root to X there are no nodes with a 
value greater than X.

Return the number of good nodes in the binary tree.
"""
# My approach: Take a stack and do DFS and take list too, if element >= maxNode(i.e. stack[-1]) from root then append
# stack and insert it in ans array as well. and when both left and right is checked properly then at end check if
# the node is top element of stack if yes pop it and continue.

# Time complexity: O(n) and space complexity : O(2n) : auxilary + stack ~ O(n)

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def traverseTree(self, node, stack, goodNodes):
        if node is None:
            return

        if len(stack) == 0:
            stack.append(node)
            goodNodes.append(node.val)
        elif stack[-1].val <= node.val:
            stack.append(node)
            goodNodes.append(node.val)

        self.traverseTree(node.left, stack, goodNodes)
        self.traverseTree(node.right, stack, goodNodes)

        if node == stack[-1]:
            stack.pop()

    def goodNodes(self, root: TreeNode) -> int:
        stack = deque()
        goodNodes = []
        self.traverseTree(root, stack, goodNodes)

        return len(goodNodes)


"""
More Better Approach : Time complexity : O(N) and space Complexity - O(n) : Queue
Approach: Go level by level put element in queue and with element put maximum value as well.
"""
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def goodNodes(self, root: TreeNode) -> int:
        totalGoodNodes = 0
        que = deque()
        que.append((root, float('-inf')))

        while que:
            currNode, maxVal = que.popleft()

            if currNode.val >= maxVal:
                totalGoodNodes += 1

            if currNode.left:
                que.append((currNode.left, max(maxVal, currNode.val)))

            if currNode.right:
                que.append((currNode.right, max(maxVal, currNode.val)))

        return totalGoodNodes

"""
Question 36: 437. Path Sum III
Given the root of a binary tree and an integer targetSum, return the number of paths where the sum of the values along 
the path equals targetSum. The path does not need to start or end at the root or a leaf, but it must go downwards 
(i.e., traveling only from parent nodes to child nodes). 

Approach : Take the all elements in the array and once the left and right of the element is done check the sum if obtained
then increase the counter and remove that node from the list. Repeat this and calculate the total paths.
Time Complexity = O(nlogn) and space complexity is O(n)
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def countPaths(self, node, count, nodeList, targetSum):
        if node is None:
            return

        nodeList.append(node)

        self.countPaths(node.left, count, nodeList, targetSum)
        self.countPaths(node.right, count, nodeList, targetSum)

        tempSum = 0
        for i in range(len(nodeList) - 1, -1, -1):
            tempSum += nodeList[i].val

            if tempSum == targetSum:
                count[0] += 1

        nodeList.pop()

    def pathSum(self, root: Optional[TreeNode], targetSum: int) -> int:
        count = [0]
        nodeList = []
        self.countPaths(root, count, nodeList, targetSum)

        return count[0]

"""
Question 37: 1372. Longest ZigZag Path in a Binary Tree

You are given the root of a binary tree.

A ZigZag path for a binary tree is defined as follow:

Choose any node in the binary tree and a direction (right or left).
If the current direction is right, move to the right child of the current node; otherwise, move to the left child.
Change the direction from right to left or from left to right.
Repeat the second and third steps until you can't move in the tree.
Zigzag length is defined as the number of nodes visited - 1. (A single node has a length of 0).

Return the longest ZigZag path contained in that tree.

Approach: We can traverse tree but we have to think if node is coming from left and we are going right then we have to
add +1 else its just the 1 and vice versa. So we have to maintain the current node. 
Time Complexity: O(n) and space : O(1) + auxiliary stack space 
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:

    def findTheMaxZigzag(self, node, comesFrom, maxLen, currLen):
        if node is None:
            return

        maxLen[0] = max(maxLen[0], currLen)
        if comesFrom == 'left':
            self.findTheMaxZigzag(node.left, 'left', maxLen, 1)
        else:
            self.findTheMaxZigzag(node.left, 'left', maxLen, currLen + 1)

        if comesFrom == 'right':
            self.findTheMaxZigzag(node.right, 'right', maxLen, 1)
        else:
            self.findTheMaxZigzag(node.right, 'right', maxLen, currLen + 1)

    def longestZigZag(self, root: Optional[TreeNode]) -> int:
        maxLen = [0]

        self.findTheMaxZigzag(root, 'left', maxLen, 0)
        self.findTheMaxZigzag(root, 'right', maxLen, 0)

        return maxLen[0]


"""
Question 38: 236. Lowest Common Ancestor of a Binary Tree
Given a binary tree, find the lowest common ancestor (LCA) of two given nodes in the tree.

According to the definition of LCA on Wikipedia: “The lowest common ancestor is defined between two nodes p and q as the
lowest node in T that has both p and q as descendants (where we allow a node to be a descendant of itself).”
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution:
    def findLCA(self, node, p, q, p_status, q_status, finalLCA):
        if node is None or node is p or node is q:
            return node

        l1 = self.findLCA(node.left, p, q, p_status, q_status, finalLCA)
        l2 = self.findLCA(node.right, p, q, p_status, q_status, finalLCA)

        if l1 and l2:
            return node

        if l1:
            return l1
        elif l2:
            return l2

        return None

    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':
        finalLCA = [None]
        p_status = False
        q_status = False
        return self.findLCA(root, p, q, p_status, q_status, finalLCA)

"""
----------------------------------  Binary-Tree-BFS  ----------------------------------
"""

"""
Question 39:  199. Binary Tree Right Side View
Given the root of a binary tree, imagine yourself standing on the right side of it, return the values of the nodes 
you can see ordered from top to bottom.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if root:
            que = deque()
            ansList = []
            que.append(root)
            while que:
                que_size = len(que)
                last_element = que[-1]
                ansList.append(last_element.val)
                for i in range(que_size):
                    curr = que.popleft()

                    if curr.left:
                        que.append(curr.left)
                    if curr.right:
                        que.append(curr.right)

            return ansList

"""
Question 40: 1161. Maximum Level Sum of a Binary Tree
Given the root of a binary tree, the level of its root is 1, the level of its children is 2, and so on.
Return the smallest level x such that the sum of all the values of nodes at level x is maximal.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxLevelSum(self, root: Optional[TreeNode]) -> int:
        max_sum = float('-inf')
        level = 0
        que = deque()
        que.append(root)
        j = 0
        while que:
            size = len(que)
            temp_sum = 0
            for i in range(size):
                curr = que.popleft()
                temp_sum += curr.val
                if curr.left:
                    que.append(curr.left)
                if curr.right:
                    que.append(curr.right)

            if max_sum < temp_sum:
                max_sum = temp_sum
                level = j + 1

            j += 1

        return level

"""
----------------------------------  Binary-Search-Tree  ----------------------------------
"""
"""
Question 41: 700. Search in a Binary Search Tree

You are given the root of a binary search tree (BST) and an integer val.
Find the node in the BST that the node's value equals val and return the subtree rooted with that node. If such a node 
does not exist, return null.
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def search(self,node,val):
        if node is None or node.val == val :
            return node
        elif node.val < val:
            return self.search(node.right,val)
        else:
            return self.search(node.left,val)

    def searchBST(self, root: Optional[TreeNode], val: int) -> Optional[TreeNode]:
        return self.search(root,val)


"""
Question 42: Delete a Node in BST
"""

# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def findrightMost(self, root):
        if root.right is None:
            return root

        return self.findrightMost(root.right)

    def helper(self, root):
        if root.left is None:
            return root.right
        elif root.right is None:
            return root.left
        else:
            rightSide = root.right
            rightMost = self.findrightMost(root.left)

            rightMost.right = rightSide

        return root.left

    def deleteNode(self, root: Optional[TreeNode], key: int) -> Optional[TreeNode]:
        if root is None:
            return None

        dummyHead = root
        if root.val == key:
            print('yes')
            return self.helper(root)

        while root:
            if key < root.val:
                if root.left is not None and root.left.val == key:
                    root.left = self.helper(root.left)
                else:
                    root = root.left

            else:
                if root.right is not None and root.right.val == key:
                    root.right = self.helper(root.right)
                else:
                    root = root.right

        return dummyHead



"""
----------------------------------  Graph-DFS  ----------------------------------
"""

"""
Question 43: 841. Keys and Rooms
There are n rooms labeled from 0 to n - 1 and all the rooms are locked except for room 0. Your goal is to visit all the 
rooms. However, you cannot enter a locked room without having its key.

When you visit a room, you may find a set of distinct keys in it. Each key has a number on it, denoting which room it 
unlocks, and you can take all of them with you to unlock the other rooms.

Given an array rooms where rooms[i] is the set of keys that you can obtain if you visited room i, return true if you 
can visit all the rooms, or false otherwise.

Intuition: Starting with the '0' it must visit all the nodes means all the rooms and if it can visit all the rooms
then only we can return true else false. Simple DFS but graph must not contain connected components if present then 
return false.

"""
class Solution:
    def dfs(self,rooms,visited,node):
        visited.add(node)
        for neighbors in rooms[node]:
            if neighbors not in visited:
                self.dfs(rooms,visited,neighbors)

    def canVisitAllRooms(self, rooms: List[List[int]]) -> bool:
        visited = set()

        self.dfs(rooms,visited,0)

        if len(visited) == len(rooms):
            return True
        else:
            return False


"""
Question 44: Number of provinces

There are n cities. Some of them are connected, while some are not. If city a is connected directly with city b, 
and city b is connected directly with city c, then city a is connected indirectly with city c.

A province is a group of directly or indirectly connected cities and no other cities outside of the group.

You are given an n x n matrix isConnected where isConnected[i][j] = 1 if the ith city and the jth city are directly 
connected, and isConnected[i][j] = 0 otherwise.

Return the total number of provinces.
"""

# Solved Earlier
class Solution:
    def dfs(self, start, adj_list, visited_array):
        visited_array[start] = 1
        for neigh in adj_list[start]:
            if visited_array[neigh] == 0:
                self.dfs(neigh, adj_list, visited_array)

    def findCircleNum(self, isConnected: List[List[int]]) -> int:

        total_vertices = len(isConnected)
        adj_list = [[] for i in range(total_vertices)]
        for i in range(total_vertices):
            for j in range(len(isConnected[i])):
                if i != j and isConnected[i][j] == 1:
                    adj_list[i].append(j)
                    adj_list[j].append(i)

        visited_array = [0] * total_vertices

        count = 0
        for i in range(total_vertices):
            if visited_array[i] == 0:
                self.dfs(i, adj_list, visited_array)
                count += 1

        return count


# Revision

class Solution:
    def dfs(self, i, adjList, visited):
        visited.add(i)
        for neighbor in adjList[i]:
            if neighbor not in visited:
                self.dfs(neighbor, adjList, visited)

    def findCircleNum(self, isConnected: List[List[int]]) -> int:
        visited = set()
        # convert the matrix into adj list
        adjList = [[] for i in range(len(isConnected))]
        for i in range(len(isConnected)):
            for j in range(len(isConnected)):
                if isConnected[i][j] == 1:
                    adjList[i].append(j)

        count = 0

        for i in range(len(adjList)):
            if i not in visited:
                count += 1
                self.dfs(i, adjList, visited)

        return count


"""
Question 45: 1466. Reorder Routes to Make All Paths Lead to the City Zero
There are n cities numbered from 0 to n - 1 and n - 1 roads such that there is only one way to travel between 
two different cities (this network form a tree). Last year, The ministry of transport decided to orient the roads in 
one direction because they are too narrow.

Roads are represented by connections where connections[i] = [ai, bi] represents a road from city ai to city bi.

This year, there will be a big event in the capital (city 0), and many people want to travel to this city.

Your task consists of reorienting some roads such that each city can visit the city 0. Return the minimum number of 
edges changed.

It's guaranteed that each city can reach city 0 after reorder.


Intuition: make the graph bidirectional by adding -ve nodes in place of nodes for opposite path and apply dfs/bfs.
"""

class Solution:
    def dfs(self, i, adjList, count, visited):
        visited.add(i)
        for neighbors in adjList[i]:
            if abs(neighbors) not in visited:
                if neighbors > 0:
                    print(neighbors)
                    count[0] += 1
                self.dfs(abs(neighbors), adjList, count, visited)

    def minReorder(self, n: int, connections: List[List[int]]) -> int:
        adjList = [[] for i in range(n)]
        count = [0]
        visited = set()
        for conn in connections:
            adjList[conn[0]].append(conn[1])
            adjList[conn[1]].append(-conn[0])

        self.dfs(0, adjList, count, visited)

        return count[0]

"""
Question 46: 399. Evaluate Division
You are given an array of variable pairs equations and an array of real numbers values, 
where equations[i] = [Ai, Bi] and values[i] represent the equation Ai / Bi = values[i]. Each Ai or Bi is a 
string that represents a single variable.

You are also given some queries, where queries[j] = [Cj, Dj] represents the jth query where you must find the answer 
for Cj / Dj = ?.

Return the answers to all queries. If a single answer cannot be determined, return -1.0.

Note: The input is always valid. You may assume that evaluating the queries will not result in division by zero and that
 there is no contradiction.
 
Intuition: Make a graph then we just have start node and endNode just multiply them to get the answer if either is not
present in the dictionary then return -1 or else apply bfs/dfs then multiply and return the value.
If the start and end are not connected then return -1 as well. 
"""

class Solution:
    def bfs(self, start, end, adjList):
        if start not in adjList or end not in adjList:
            return -1
        visited = set()
        que = deque()
        visited.add(start)
        que.append([start, 1])

        while que:
            node, weight = que.popleft()
            if node == end:
                return weight

            for neighbor in adjList[node]:
                if neighbor[0] not in visited:
                    visited.add(neighbor[0])
                    que.append([neighbor[0], neighbor[1] * weight])

        return -1

    def calcEquation(self, equations: List[List[str]], values: List[float], queries: List[List[str]]) -> List[float]:
        adjList = collections.defaultdict(list)

        for i, eq in enumerate(equations):
            a, b = eq
            adjList[a].append([b, values[i]])
            adjList[b].append([a, 1 / values[i]])

        print(adjList)
        ans = []
        for query in queries:
            ans.append(self.bfs(query[0], query[1], adjList))

        return ans


"""
----------------------------------  Graph-BFS  ----------------------------------
"""
"""
Question 47: 1926. Nearest Exit from Entrance in Maze
You are given an m x n matrix maze (0-indexed) with empty cells (represented as '.') and walls (represented as '+'). 
You are also given the entrance of the maze, where entrance = [entrancerow, entrancecol] denotes the row and column of 
the cell you are initially standing at.

In one step, you can move one cell up, down, left, or right. You cannot step into a cell with a wall, and you cannot 
step outside the maze. Your goal is to find the nearest exit from the entrance. An exit is defined as an empty cell 
that is at the border of the maze. The entrance does not count as an exit.

Return the number of steps in the shortest path from the entrance to the nearest exit, or -1 if no such path exists.

Intuition : Simple BFS problem just check its left right up and down and store the steps. Apply BFS
"""

class Solution:
    def nearestExit(self, maze: List[List[str]], entrance: List[int]) -> int:
        rows = len(maze)
        cols = len(maze[0])

        que = deque()
        visited = [[0 for i in range(cols)] for i in range(rows)]
        # print(visited)
        que.append([entrance[0], entrance[1], 0])
        visited[entrance[0]][entrance[1]] = 1
        while que:
            curr_i, curr_j, steps = que.popleft()

            if (curr_i == 0 or curr_i == rows - 1 or curr_j == 0 or curr_j == cols - 1) and [curr_i,
                                                                                             curr_j] != entrance:
                return steps

            if curr_i + 1 < rows:
                if visited[curr_i + 1][curr_j] == 0:
                    if maze[curr_i + 1][curr_j] == '.':
                        que.append([curr_i + 1, curr_j, steps + 1])
                        visited[curr_i + 1][curr_j] = 1

            if curr_i - 1 >= 0:
                if visited[curr_i - 1][curr_j] == 0:
                    if maze[curr_i - 1][curr_j] == '.':
                        que.append([curr_i - 1, curr_j, steps + 1])
                        visited[curr_i - 1][curr_j] = 1

            if curr_j + 1 < cols:
                if visited[curr_i][curr_j + 1] == 0:
                    if maze[curr_i][curr_j + 1] == '.':
                        que.append([curr_i, curr_j + 1, steps + 1])
                        visited[curr_i][curr_j + 1] = 1

            if curr_j - 1 >= 0:
                if visited[curr_i][curr_j - 1] == 0:
                    if maze[curr_i][curr_j - 1] == '.':
                        que.append([curr_i, curr_j - 1, steps + 1])
                        visited[curr_i][curr_j - 1] = 1

        return -1


"""
Question 48: 994. Rotting Oranges

You are given an m x n grid where each cell can have one of three values:

0 representing an empty cell,
1 representing a fresh orange, or
2 representing a rotten orange.
Every minute, any fresh orange that is 4-directionally adjacent to a rotten orange becomes rotten.

Return the minimum number of minutes that must elapse until no cell has a fresh orange. If this is impossible, return 
-1.

Intuition: 
"""

# Previously Solved

from collections import deque

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        visited = [[0 for i in range(cols)] for j in range(rows)]

        que = deque()
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    que.append((i, j, 0))
                    visited[i][j] = 1
        t = 0
        while que:
            current_orange = que.popleft()
            i = current_orange[0]
            j = current_orange[1]
            time = current_orange[2]
            t = max(t, time)

            if i - 1 >= 0:
                if visited[i - 1][j] == 0:
                    if grid[i - 1][j] == 1:
                        grid[i - 1][j] = 2
                        visited[i - 1][j] = 1
                        que.append((i - 1, j, time + 1))

            if i + 1 < rows:
                if visited[i + 1][j] == 0:
                    if grid[i + 1][j] == 1:
                        grid[i + 1][j] = 2
                        visited[i + 1][j] = 1
                        que.append((i + 1, j, time + 1))

            if j - 1 >= 0:
                if visited[i][j - 1] == 0:
                    if grid[i][j - 1] == 1:
                        grid[i][j - 1] = 2
                        visited[i][j - 1] = 1
                        que.append((i, j - 1, time + 1))

            if j + 1 < cols:
                if visited[i][j + 1] == 0:
                    if grid[i][j + 1] == 1:
                        grid[i][j + 1] = 2
                        visited[i][j + 1] = 1
                        que.append((i, j + 1, time + 1))

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 1:
                    return -1

        return t


# Practice

class Solution:
    def orangesRotting(self, grid: List[List[int]]) -> int:
        rows = len(grid)
        cols = len(grid[0])
        que = deque()
        visited = [[0 for i in range(cols)] for j in range(rows)]
        time = 0
        for i in range(rows):
            for j in range(cols):
                if grid[i][j] == 2:
                    visited[i][j] = 1
                    que.append([i, j, time])

        final_ans = 0
        while que:
            curr_i, curr_j, time = que.popleft()
            final_ans = max(final_ans, time)
            if curr_i + 1 < rows:
                if visited[curr_i + 1][curr_j] == 0 and grid[curr_i + 1][curr_j] != 0:
                    visited[curr_i + 1][curr_j] = 1
                    que.append([curr_i + 1, curr_j, time + 1])

            if curr_i - 1 >= 0:
                if visited[curr_i - 1][curr_j] == 0 and grid[curr_i - 1][curr_j] != 0:
                    visited[curr_i - 1][curr_j] = 1
                    que.append([curr_i - 1, curr_j, time + 1])

            if curr_j + 1 < cols:
                if visited[curr_i][curr_j + 1] == 0 and grid[curr_i][curr_j + 1] != 0:
                    visited[curr_i][curr_j + 1] = 1
                    que.append([curr_i, curr_j + 1, time + 1])

            if curr_j - 1 >= 0:
                if visited[curr_i][curr_j - 1] == 0 and grid[curr_i][curr_j - 1] != 0:
                    visited[curr_i][curr_j - 1] = 1
                    que.append([curr_i, curr_j - 1, time + 1])

        for i in range(rows):
            for j in range(cols):
                if grid[i][j] != 0 and visited[i][j] == 0:
                    return -1

        return final_ans


"""
----------------------------------  Heap / Priority Queue  ----------------------------------
"""