import numpy as np
import pdb
from numpy import linalg as la
f1 = open('','r')
f2 = open('../800_sem.txt','r')
f = open('','w')
lines1 = f1.readlines()
lines2 = f2.readlines()
f1.close()
f2.close()
class_num = 800
for i in range(class_num):
    words = lines1[i].split(',')
    array1 = np.array(words)
    words = lines2[i].split(' ')
    words.remove(words[0])
    array2 = np.array(words)
    array1 = array1.astype(float)
    array1 = array1/la.norm(array1)
    array2 = array2[0:200].astype(float)
    array2 = array2/la.norm(array2)

    array = 100*(array1+array2)/2
    content = ''
    array = str(list(array))
    array = array[1:]
    array = array[:-1]
    content=array+'\n'
    f.write(content)
f.close()    
