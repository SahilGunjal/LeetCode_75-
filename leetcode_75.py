# This file contains 75 questions asked by companies in their coding interviews and their solutions. (Leetcode - 75)
# Author: Sahil Sanjay Gunjal.
# language: Python
# Note: Anyone can see and get the codes from this file. It's completely free and do share your best solutions if you
#       find any. Motive of this is to contribute to the community and enhance the learning opportunity for everyone.


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