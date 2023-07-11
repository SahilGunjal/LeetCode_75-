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


