from cleaner import *

hi()

def test_remove_english_punc(self):
    self.assertEqual(remove_english_punc("Hello, world!"), "Hello world")