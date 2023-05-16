import numpy as np

# table[i][j] = Q(s(i), a(j)). s(i) = {0 if A, 1 if B}, j = {1 if move, 0 if stay}
# table = np.random.normal(loc=0, scale=0.01, size=(2, 2))
table = np.zeros((2, 2))

a = 0.5
g = 0.8

s_cur = 0

iter = 200

gamma = a
for i in range(iter):
    action = np.argmax(table[s_cur])

    if table[s_cur][0] == table[s_cur][1]:
        # break tie
        # if np.random.rand(1) > 0.5:
        #     action = 1
        action = 1

    r = 1 - action
    s_next = s_cur
    # change state if want move
    if action == 1:
        s_next = 1 - s_cur

    table[s_cur][action] += a * (r + gamma * np.max(table[s_next]) - table[s_cur][action])
    gamma *= g

    s_cur = s_next

print(table)

