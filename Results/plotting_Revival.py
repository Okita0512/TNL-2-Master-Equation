import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator, tick_params
import numpy as np
fig, ax = plt.subplots()

font = {'family':'Times New Roman', 'weight': 'roman', 'size':12}
fig = plt.figure(figsize=(10, 5), dpi = 128)

# ==============================================================================================
#                                         data reading     
# ==============================================================================================

data2 = np.loadtxt("./benchmark/Photon_echo_1.txt", dtype = float)
plt.plot(data2[:, 0], 1 - data2[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = 'blue', label = "P1 (exact)", alpha = 0.2)
plt.plot(data2[:, 0], data2[:, 1], 'o', markersize = 10, linewidth = 1, markerfacecolor = 'white', color = 'red', label = "P2 (exact)", alpha = 0.2)

data2 = np.loadtxt("./TNL-2/Photon_echo.txt", dtype = float)
plt.plot(data2[:, 0], data2[:, 1], "-", linewidth = 2.0, color = 'blue', label = "P1 (TNL-2)")
plt.plot(data2[:, 0], data2[:, 7], "-", linewidth = 2.0, color = 'red', label = "P2 (TNL-2)")

data2 = np.loadtxt("./Redfield/Photon_echo.txt", dtype = float)
plt.plot(data2[:, 0], data2[:, 1], "--", linewidth = 2.0, color = 'blue', label = "P1 (Redfield)", alpha = 0.5)
plt.plot(data2[:, 0], data2[:, 7], "--", linewidth = 2.0, color = 'red', label = "P2 (Redfield)", alpha = 0.5)

# data5 = np.loadtxt("./HEOM_1/Photon_echo.txt", dtype = float)
# plt.plot(data5[:, 0], data5[:, 7], "--", linewidth = 2.0, color = 'green', label = "HEOM_tier=1")

# ==============================================================================================
#                                      plotting set up     
# ==============================================================================================

# x and y range of plotting 
time = 2100.0             # x-axis range: (0, time)
y1, y2 = - 0.1, 1.1     # y-axis range: (y1, y2)

# scale for major and minor locator
x_major_locator = MultipleLocator(500)
x_minor_locator = MultipleLocator(100)
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
ax.set_ylabel(r'Population', font = 'Times New Roman', size = 20)
ax.set_title('Photon Echo', font = 'Times New Roman', size = 20)

# legend location, font & markersize
ax.legend(loc = 'center', prop = font, markerscale = 1, frameon = False)
# plt.legend(frameon = False)

# plt.show()

plt.savefig("Photon Echo.pdf", bbox_inches='tight')