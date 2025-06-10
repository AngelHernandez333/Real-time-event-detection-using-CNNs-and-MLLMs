import unittest

def hIndex(heights):
    index=0
    water=0
    while True:
        highest=heights[index]
        for i in range(index+1,len(heights)):
            print(f"Checking height at index {i}: {heights[i]} against highest: {highest}")
            if heights[i] >= highest:
                index = i
                break
            else:
                water += highest - heights[i]
        print(f"Final index: {index}, Total water trapped: {water}")
        if index >= len(heights) - 1:
            break
        print(f"Water trapped: {water}")

    return water
            
#
class TestHIndex(unittest.TestCase):
    def test_hIndex(self):
        test_cases = [
            ([0,1,0,2,1,0,1,3,2,1,2,1], 6),
        ]
        
        for citations, expected in test_cases:
            with self.subTest(citations=citations):
                result = hIndex(citations)
                self.assertEqual(result, expected, f"Failed for {citations}. Expected {expected}, got {result}")

if __name__ == "__main__":
    unittest.main()