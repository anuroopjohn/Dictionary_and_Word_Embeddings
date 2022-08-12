# cbdafd86
# 835ef5a3a27b6f5991e83b750642845b

import sys
sys.path.append('..')
import os

muse_path = '../data/muse_dictionaries'

langs = ['fr','de','en','hi','es','zh']

files = os.listdir(muse_path)

_files = []
for i in files:
    if len(i.split('.'))==2:
        src, tgt = i[:-4].split('-')
        if src in langs and tgt in langs:
            _files.append(i)
files = _files
print (files)

words=[]
pairs= []
for f in files:
    lines = open(muse_path+'/'+f).readlines()
    src, tgt = f[:-4].split('-')
    sep = '\t'
    if ' ' in lines[0]: sep = ' '

    for w in lines:
        w1,w2 = w.rstrip().split(sep)
    words.append((w1,src))
    words.append((w2,tgt))

    print (lines[0],sep)
    words+=lines
    print (f'file {f}: {len(lines)}')

print (len(words))
print (words[:5])

words = list(set(words))
print (len(words))
print (words[:5])