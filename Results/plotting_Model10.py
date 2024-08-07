import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':12}
fig = plt.figure(figsize=(7, 7), dpi = 128)

# ==============================================================================================
#                                         data reading     
# ==============================================================================================

data2 = np.loadtxt("./Redfield/Model10.txt", dtype = float)
plt.plot(data2[:, 0], data2[:, 1] - data2[:, 7], "-", linewidth = 2.0, color = 'red', label = "Redfield")

data2 = np.loadtxt("./TNL-2/Model10.txt", dtype = float)
plt.plot(data2[:, 0], data2[:, 1] - data2[:, 7], "-", linewidth = 2.0, color = 'blue', label = "TNL-2")

data2 = np.loadtxt("./benchmark/Model10.dat", dtype = float)
plt.plot(data2[:, 0], data2[:, 1], "o", linewidth = 2.0, color = 'navy', label = "MCTDH")

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
time = 15.0             # x-axis range: (0, time)
y1, y2 = - 1.1, 1.0     # y-axis range: (y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(5)
x_minor_locator = MultipleLocator(1)
y_major_locator = MultipleLocator(0.5)
y_minor_locator = MultipleLocator(0.1)

# x-axis and LHS y-axis
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
ax.xaxis.set_minor_locator(x_minor_locator)
ax.yaxis.set_major_locator(y_major_locator)
ax.yaxis.set_minor_locator(y_minor_locator)
ax.tick_params(which = 'major', length = 8, labelsize = 10)
ax.tick_params(which = 'minor', length = 4)

x1_label = ax.get_xticklabels()
[x1_label_temp.set_fontname('Times New Roman') for x1_label_temp in x1_label]
y1_label = ax.get_yticklabels()
[y1_label_temp.set_fontname('Times New Roman') for y1_label_temp in y1_label]

plt.tick_params(labelsize = 20, which = 'both', direction = 'in')
plt.xlim(0.0, time)
plt.ylim(y1, y2)

# RHS y-axis
ax2 = ax.twinx()
ax2.yaxis.set_major_locator(y_major_locator)
ax2.yaxis.set_minor_locator(y_minor_locator)
ax2.tick_params(which = 'major', length = 8)
ax2.tick_params(which = 'minor', length = 4)
ax2.axes.yaxis.set_ticklabels([])

plt.tick_params(which = 'both', direction = 'in')
plt.ylim(y1, y2)

# name of x, y axis and the panel
ax.set_xlabel('time / a.u.', font = 'Times New Roman', size = 20)
ax.set_ylabel(r'$\langle \sigma_z \rangle (t)$', font = 'Times New Roman', size = 20)
ax.set_title('Spin Boson Model 10', font = 'Times New Roman', size = 20)

# legend location, font & markersize
ax.legend(loc = 'upper right', prop = font, markerscale = 1)
plt.legend(frameon = False)

# plt.show()

plt.savefig("Spin Boson Ohmic Model 10.pdf", bbox_inches='tight')