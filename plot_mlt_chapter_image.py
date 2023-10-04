import matplotlib.pyplot as plt
import matplotlib

plt.rcParams["font.family"] = "Times New Roman"

cmap0 = matplotlib.cm.get_cmap("Oranges")
cmap1 = matplotlib.cm.get_cmap("Blues")
cmap2 = matplotlib.cm.get_cmap("Reds")
cmap3 = matplotlib.cm.get_cmap("Purples")
cmap4 = matplotlib.cm.get_cmap("Greys")

colors_regular = [cmap0(0.2), cmap1(0.3), cmap2(0.5), cmap3(0.7), cmap4(0.9)]
colors_dark = [cmap0(0.3), cmap1(0.4), cmap2(0.6), cmap3(0.8), cmap4(1.0)]

fig, ax = plt.subplots(1, 1)
leters_size = 20
y_limits = (1, 200)

p = [3, 4]
cpwl = [
    [1.0, 1.1, 1.2100000000000002, 1.3310000000000002, 1.4641000000000002, 1.61051, 1.7715610000000002, 1.9487171, 2.1435888100000002, 2.357947691, 2.5937424601, 2.8531167061100002, 3.1384283767210004, 3.4522712143931003, 3.1, 3.2, 3.3000000000000003, 3.4000000000000004],
    [1.0, 1.1, 1.2100000000000002, 1.3310000000000002, 1.4641000000000002, 1.61051, 1.7715610000000002, 1.9487171, 2.1435888100000002, 2.357947691, 2.1, 2.2, 2.3000000000000003],
]

err = [
    [0.8415241121582129, 0.8301105394289546, 0.7967853827408525, 0.7324768440473581, 0.6874775868744354, 0.5646827685523109, 0.39462731283819136, 0.3024501087599718, 0.2798604445038005, 0.17168231222328753, 0.1102209752703036, 0.07753974748838786, 0.05453078287106337, 0.04273809923384259, 0.056637810745222505, 0.05304431809864472, 0.05006798003515486, 0.048703101764712584],
    [0.5960931610457801, 0.576940253506065, 0.4413778168966874, 0.3507859571386087, 0.27353628447755024, 0.20630666808065928, 0.17610246876987454, 0.1089761318406112, 0.060879683369979294, 0.035507817488064426, 0.06754651329155496, 0.05983289279287807, 0.041660852428524485],
]

dts = [
    [0.003367003367003367, 0.00306044376434583, 0.0025559105431309905, 0.0022766078542970974, 0.0022896393817973667, 0.002073613271124935, 0.0017043033659991478, 0.0015414258188824663, 0.001404001404001404, 0.001272264631043257, 0.0011655011655011655, 0.0011651616661811825, 0.0009535160905840286, 0.0008674907829104315, 0.0010712372790573112, 0.0009480919649205973, 0.0009197516670498966, 0.0008902737591809482],
    [0.0017338534893801473, 0.0015760441292356187, 0.001324942033786022, 0.0011788977306218685, 0.0011795930404010617, 0.0010686615014694097, 0.0008828073273008167, 0.0007992007992007992, 0.0007276696379843551, 0.0006597394029358403, 0.0007649646203863071, 0.000716974368166338, 0.0006836438215689625],
]

runtimes = [
    [10.431009292602539, 8.564547061920166, 17.13622236251831, 44.329190731048584, 31.125303745269775, 38.018271923065186, 34.10909557342529, 52.48997616767883, 60.012927770614624, 83.77455735206604, 118.04440140724182, 115.85980796813965, 208.81342220306396, 259.4341731071472, 146.69270086288452, 195.18282413482666, 206.70982217788696, 215.8141529560089],
    [14.505264043807983, 17.703612327575684, 28.289212942123413, 38.25164842605591, 46.186219692230225, 111.35497188568115, 98.58344340324402, 116.20505619049072, 178.62511682510376, 229.5563042163849, 163.38052129745483, 163.56908702850342, 245.71752667427063],
]

cont = 0
for degree in p:
    m = cpwl[cont]

    error_percent = [i * 100 for i in err[cont]]

    ax.plot(m, error_percent, "o", label="MLT" + str(degree)+"tri", color=colors_dark[cont])
    cont += 1


ax.plot([0.5, 8.0], [5, 5], 'k--')
# ax .set(xlabel = "Grid-points-per-wavelength (G)", ylabel = "E %")
ax.set_title("Error with varying C")
ax.set(xlabel="Cells-per-wavelength (C)", ylabel="E %")
ax.set_yticks([3, 5, 10, 30, 100])
ax.set_xticks([1, 2, 4, 6, 8])
ax.set_xlim((0.0, 8.5))
ax.set_ylim(y_limits)
ax.set_yscale("log")
ax.legend()
ax.set_yticks([3, 5, 10, 30, 100])
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# ax.text(-0.1, 1.0, '(b)', transform=ax .transAxes, size=leters_size)
ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
fig.set_size_inches(9, 5)
plt.show()

# cont = 0
# p = [1, 2, 3, 4, 5]
# for degree in p:
#     g = copy.deepcopy(gpwl[cont])
#     for ite in range(len(gpwl[cont])):
#         g[ite] = old_to_new_g_converter("KMV", degree, gpwl[cont][ite])

#     error_percent = [i * 100 for i in err[cont]]

#     plt.plot(g, error_percent, "o-", label="KMV" + str(degree) + "tri")
#     cont += 1

# plt.plot([3.0, 13.0], [5, 5], "k--")
# plt.xlabel("Grid-points-per-wavelength (G)")
# plt.ylabel("E %")
# plt.title("Error with varying G")
# plt.legend(loc="lower left")
# plt.yscale("log")
# plt.yticks([1, 5, 10, 100])
# ax = plt.gca()
# ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
# plt.show()

# dp = 2
# print("for p=" + str(dp))
# gorigin = 11.7
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)

# dp = 3
# print("for p=" + str(dp))
# gorigin = 10.5
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)

# dp = 4
# print("for p=" + str(dp))
# gorigin = 10.5
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)

# dp = 5
# print("for p=" + str(dp))
# gorigin = 8.4
# gnew = old_to_new_g_converter("KMV", dp, gorigin)
# print(gnew)
