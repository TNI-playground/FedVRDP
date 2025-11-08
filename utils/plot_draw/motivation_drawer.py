import matplotlib.pyplot as plt

################### fang trimmed mean
# dp_accuracy =             [71.27, 51.91, 17.40, 11.70, 11.37]
# dp_variances_p=           [1.73, 3.85,  4.38,  3.15,  1.31]
# dp_variances_n=           [1.33, 2.04,  4.36,  1.75,  1.37]

# nodefend_accuracy =       []
# nodefend_variances_p=     []
# nodefend_variances_n=     []

# ours_accuracy =           [81.41, 79.35, 79.35, 74.62, 65.34]
# ours_variances_p=         [0.37, 0.50,  0.71,  0.62,  1.81]
# ours_variances_n=         [0.16, 0.34,  1.02,  0.73,  1.53]

################## agr

# dp_accuracy =             [71.27, 69.83, 45.25, 17.91, 12.04, 11.09]
# dp_variances_p=           [1.73,  1.22,  4.10,  5.54,  0.58,  0.67]
# dp_variances_n=           [1.33,  1.29,  3.81,  6.74,  1.39,  0.58]
dp_accuracy =             [71.27, 45.25, 17.91, 12.04, 11.09]
dp_variances_p=           [1.73,  4.10,  5.54,  0.58,  0.67]
dp_variances_n=           [1.33,  3.81,  6.74,  1.39,  0.58]

# nodefend_accuracy =       [85.94, 71.07, 36.77, 10.00, 10.00, 10.00]
# nodefend_variances_p=     [0.43,  1.25,  13.81, 0,     0,     0]
# nodefend_variances_n=     [0.42,  1.68,  9.13,  0,     0,     0]
nodefend_accuracy =       [85.94, 36.77, 10.00, 10.00, 10.00]
nodefend_variances_p=     [0.43,  13.81, 0,     0,     0]
nodefend_variances_n=     [0.42,  9.13,  0,     0,     0]

bulyan_accuracy =         [79.83, 74.56, 66.95, 61.91, 59.79]
bulyan_variances_p=        [0.46,  1.20,  1.39,  2.87,  1.88]
bulyan_variances_n=        [0.37,  1.45,  2.20,  1.39,  0.70]

# ours_accuracy =           [81.41,      , 80.49, 78.95, 77.34, 74.97]
# ours_variances_p=         [0.37,       , 0.17,  0.60,  0.50,  1.56]
# ours_variances_n=         [0.16,       , 0.20,  0.22,  0.82,  1.22]

# Percentage of compromised clients
# compromised_clients = [0, 5, 10, 15, 20, 25]
compromised_clients = [0, 10, 15, 20, 25]


plt.errorbar(compromised_clients, dp_accuracy, yerr=[dp_variances_p, dp_variances_n], fmt='-o', label='DPFed', capsize=4, linewidth=1.5, markersize=5, alpha=0.5)
plt.errorbar(compromised_clients, nodefend_accuracy, yerr=[nodefend_variances_p, nodefend_variances_n], fmt='-o', label='FedAvg', capsize=4, linewidth=1.5, markersize=5, alpha=0.5)
plt.errorbar(compromised_clients, bulyan_accuracy, yerr=[bulyan_variances_p, bulyan_variances_n], fmt='-o', label='RobustAggregator', capsize=4, linewidth=1.5, markersize=5, alpha=0.5)
# plt.errorbar(compromised_clients, ours_accuracy, yerr=[ours_variances_p, ours_variances_n], fmt='-o', label='Ours', capsize=4, linewidth=1.5, markersize=5, alpha=0.5)


# Adding labels and title
plt.xlabel('Percentage of Compromised Clients (%)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy against Byzantine Attack')

# Customize gridlines
plt.grid(color='lightgray', linestyle='--')
# Adjust tick labels font size
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)

# Adding legend
plt.legend()

# Saving the plot as a PNG file
plt.savefig('utils/plot_draw/dp_ablation.png')

# Displaying the plot
plt.show()
