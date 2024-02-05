import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

abc = [1]
cba = [4]
#abc.append(3)
#cba.append(2)
plt.plot(abc,cba,'ko')
plt.show()
