
from cleaner import *
import unittest

class TestCleaner(unittest.TestCase):
    def test_remove_punc(self):
        self.assertEqual(remove_punc("你好，世界！"), "你好世界")

    def test_word_seg(self):
        self.assertEqual(word_seg("我爱机器学习"), ['我', '爱', '机器学习'])

# hi()