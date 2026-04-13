import matplotlib.pyplot as plt
import numpy as np

def get_sal(filename):
	f1 = open(filename, 'r')
	loss = []
	i = 0
	for line in f1.readlines():
		if line.find('rgb Loss: ') > 0:
			i = i + 1
			print(i, '#Loss#', line)
			tmp = line.split('rgb Loss: ')[1].split(',')[0]
			loss.append(float(tmp))
	return loss



loss = get_sal('log.txt')
x = [i for i in range(0,len(loss))]

plt.figure(figsize=(15,15))
l1, = plt.plot(x, loss, label = 'line', color = 'r', linewidth = 1.0, linestyle = '-')
plt.xlabel('Iter /10')
plt.legend(handles = [l1], labels = ['Sal Loss'], loc = 'best')
plt.grid()




plt.savefig('loss.png')
plt.show()







