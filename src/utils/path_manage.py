import sys
import os

def _setup () :
    path = os.path.abspath(os.path.dirname(__file__))
    if path not in sys.path : sys.path.append(path)
