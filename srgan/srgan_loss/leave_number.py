import os
import time
import matplotlib.pyplot as plt
import numpy as np

# readline_all.py
rf = open("./test_50hzlosses.txt", 'r')
cnt = 1
ganloss_li, disloss_li = [], []
while True:
    line = rf.readline()
    ganloss = ""
    disloss = line[-9:]
    if not line: break
    if cnt < 10:
        ganloss = line[21:33]
    elif cnt >= 10 and cnt < 100:
        ganloss = line[22:34]
    else:
        ganloss = line[23:35]
    cnt += 1
    # print(ganloss, disloss)
    ganloss_li.append(float(ganloss))
    disloss_li.append(float(disloss))
    # time.sleep(1)
print(ganloss_li)
print(disloss_li)

y_ax = []
for i in range(len(ganloss_li)):
    y_ax.append("epoch " + str(i))
    
xti = []
for i in range(len(ganloss_li[:100])):
    if i % 10 == 0:
        xti.append(str(i))

plt.figure(figsize=(9,5))
plt.plot(range(len(ganloss_li[:100])),ganloss_li[:100])
plt.title('Generator Loss', fontsize=15, weight='bold')
plt.xlabel('epoch', fontsize=13)
plt.ylabel('loss', fontsize=13)
plt.grid(True)
# plt.xticks(range(len(ganloss_li[:100])), labels=xti)
# plt.yticks(np.arange(1, 6))
plt.show()

plt.figure(figsize=(9,5))
plt.plot(range(len(disloss_li[:100])),disloss_li[:100])
plt.title('Discriminator Loss', fontsize=15, weight='bold')
plt.xlabel('epoch', fontsize=13)
plt.ylabel('loss', fontsize=13)
plt.grid(True)
# plt.xticks(range(len(ganloss_li[:100])), labels=xti)
# plt.yticks(np.arange(1, 6))
plt.show()
rf.close()