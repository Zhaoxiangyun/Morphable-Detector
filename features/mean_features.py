import numpy as np
import pdb



f = open('','r')
lines = f.readlines()
num = list(np.zeros(200))
mean={}
for i in range(200):
   mean[i] = np.zeros(200)

labels = []
index = 0
for line in lines:
    words = line.split(',')
    value = np.array(words[:-1])
    value = value.astype(float)
    label = int(words[-1]) - 1
    l =  int((words[-1]))
    mean[label] = mean[label]+value
    num[label] = num[label] + 1
f.close()
print(num)

f1 = open('','w')
for i in range(200):
    mean[i] = mean[i]/num[i]
    content = str(list(mean[i]))
    content = content[1:]
    content = content[:-1]+'\n'
    f1.write(content)
f1.close()    
