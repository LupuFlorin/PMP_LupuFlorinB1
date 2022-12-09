import random


file = r"C:\Users\flori\PycharmProjects\Lab10PMP\date2.csv"
f = open(file, "a")
for i in range(0, 500):
    rand1 = round(random.uniform(-1.99, 9.99), 3)
    rand2 = round(random.uniform(-1.99, 9.99), 3)
    f.write(str(rand1) + ' ')
    f.write(str(rand2) + '\n')
f.close()
