import numpy as np
import matplotlib.pyplot as plt

data = []
with open("t_data.txt", "r") as f:
	x = f.readline().rstrip().split(" ")
	while len(x) == 2:
		if len(x) != 0:
			data.append(x)
		x = f.readline().rstrip().split(" ")

data = np.array(data).astype('float64')
data = np.flip(data[data[:,0].argsort()], axis=0)

print(data)
print()

tp = 0
fp = 0

t_tot = np.count_nonzero(data[:,1])
f_tot = len(data) - t_tot

if data[0][1] == 1:
	tp = 1
else:
	fp = 1

x = [0]
y = [0]

for i in range(len(data)-1):
	print(tp, fp)
	if (data[i][1] == 1 and data[i+1][1] == 0):
		# falling edge
		fpr = fp / f_tot
		tpr = tp / t_tot
		print("falling edge! tp:", tpr, " fp:", fpr)
		x.append(fpr)
		y.append(tpr)

	if data[i+1][1] == 1:
		tp += 1
	else:
		fp += 1

fpr = fp / f_tot
tpr = tp / t_tot
x.append(fpr)
y.append(tpr)

plt.plot(x,y)

plt.ylim(-0.1,1.1)
plt.xlim(-0.1,1.1)
plt.xlabel("FP Rate")
plt.ylabel("TP Rate")
plt.title("ROC Curve for " + str(len(data)) + " Data Points")
plt.show()


