# -*- encoding:utf-8 -*-
import os
from uer.utils.constants import *

class Tokenizer(object):
    
    def __init__(self):
        pass

    def tokenize(self, text):
        raise NotImplementedError


class CharTokenizer(Tokenizer):
    
    def __init__(self):
        super().__init__()
        self.type = "CharTokenizer"

    def tokenize(self, text):
        return list(text.strip())


class SpaceTokenizer(Tokenizer):
   
    def __init__(self):
        super().__init__()
        self.type = "SpaceTokenizer"

    def tokenize(self, text):
        """
        Splitting the sentence into words according to spaces
        """
        return text.strip().split(" ")

