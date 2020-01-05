import os
'''
epoch:1000
lr: 0.001
ph 225
pw 150
sp True
ld 0.5
batch_size 
'''
os.system('python parameter_tunning.py -ne 10 -lr 0.0001  -ld 0.5 -bs 128')
# os.system('python style_inference.py -ne 5 -lr 0.001  -ld 0.5')