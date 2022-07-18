import textdistance as td
import Levenshtein

def calc_leven_distance(raw_string, old_raw_string):
    leven_dist = [Levenshtein.distance(s1, s2) for s1,s2 in zip(raw_string, old_raw_string)]
    return leven_dist

def lendiff(seq1, seq2):
    return len(seq1)- len(seq2)

def levenshtein(seq1, seq2):
    return td.levenshtein(seq1, seq2)

def bag(seq1, seq2):
    return td.bag(seq1, seq2)
