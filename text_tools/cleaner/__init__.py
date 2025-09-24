import re, jieba

def remove_punc(text: str) -> str:
    return re.sub(r"[^\w\s]", "", text, flags=re.UNICODE)

def word_seg(text: str) -> list:
    return jieba.lcut(text)

def char_count(text: str) -> int:
    return len(text)

def hi():
    print("hello,world")