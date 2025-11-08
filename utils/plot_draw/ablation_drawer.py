import matplotlib.pyplot as plt

# Test accuracy data for each method at different percentages of compromised clients

# # Benign
# plot_name = 'benign'
# title = 'Benign Model'

# sparsefed_accuracy =      [89.80]
# flame_accuracy =          [84.38]
# median_accuracy =         [86.68]
# trimmedmean_accuracy =    [86.68]
# multikrum_accuracy =      [73.49]
# bulyan_accuracy =         [86.49]
# dp_accuracy =             [80.89]
# freerobustness_accuracy = [81.40]
# freerobustness_plus_accuracy = []

# noattack_accuracy, noattack_variance =                       [85.94], []
# sparsefed_accuracy, sparsefed_variance =                     [88.80], []
# flame_accuracy, flame_variance =                             [82.21], []
# median_accuracy, median_variance =                           [],[]
# trimmedmean_accuracy, trimmedmean_variance =                 [],[]
# multikrum_accuracy, multikrum_variance =                     [],[]
# bulyan_accuracy, bulyan_variance =                           [],[]
# dp_accuracy, dp_variance =                                   [82.57],[]
# freerobustness_accuracy, freerobustness_variance =           [80.97],[]
# freerobustness_plus_accuracy, freerobustness_plus_variance = [],[]

# # Fang Attack
# plot_name = 'Fangpercentage'
# title = 'Fang Attack'

# nodefense_accuracy, nodefense_variance =                     [75, 53, 33, 18], []
# trimmedmean_accuracy, trimmedmean_variance =                 [80.13, 60.75, 40.33, 21.33], []
# median_accuracy, median_variance =                           [76.26, 54.09, 35.37, 19.58], []
# krum_accuracy, multikrum_variance =                          [24.50, 28.52, 29.44, 18.65], []
# bulyan_accuracy, bulyan_variance =                           [79.15, 70.45, 63.17, 59.61], []
# sparsefed_accuracy, sparsefed_variance =                     [69.41, 42.59, 23.76, 12.15], []
# # flame_accuracy, flame_variance =                             [], []
# # dp_accuracy, dp_variance =                                   [], []
# freerobustness_accuracy, freerobustness_variance =           [79.35, 77.54, 74.62, 65.34], []


# AGR Attac
plot_name = 'AGRpercentage'
title = 'AGR Attack'
nodefense_accuracy, nodefense_variance =                     [36, 9, 9, 9], []
trimmedmean_accuracy, trimmedmean_variance =                 [78.05, 49.88, 11.05, 10.00], []
median_accuracy, median_variance =                           [36.77, 10.00, 10.00, 10.00], []
krum_accuracy, multikrum_variance =                          [73.68, 72.24, 66.53, 49.21], []
bulyan_accuracy, bulyan_variance =                           [74.56, 66.95, 61.91, 59.79], []
sparsefed_accuracy, sparsefed_variance =                     [26.72, 10.38, 10.00, 10.00], []
# flame_accuracy, flame_variance =                             [], []
# dp_accuracy, dp_variance =                                   [], []
freerobustness_accuracy, freerobustness_variance =           [80.49, 78.95, 77.34, 74.97], []
# freerobustness_plus_accuracy, freerobustness_plus_variance = [], []

# Percentage of compromised clients
compromised_clients = [10, 15, 20, 25]

# Plotting the data
plt.plot(compromised_clients, nodefense_accuracy, marker='o', label='NoDefend')
plt.plot(compromised_clients, trimmedmean_accuracy, marker='o', label='TrimmedMean')
plt.plot(compromised_clients, median_accuracy, marker='o', label='Median')
plt.plot(compromised_clients, krum_accuracy, marker='o', label='Krum')
plt.plot(compromised_clients, bulyan_accuracy, marker='o', label='Bulyan')
plt.plot(compromised_clients, sparsefed_accuracy, marker='o', label='SparseFed')
# plt.plot(compromised_clients, flame_accuracy, marker='o', label='FLAME')
# plt.plot(compromised_clients, dp_accuracy, marker='o', label='DP')
plt.plot(compromised_clients, freerobustness_accuracy, marker='^', label='Ours')

# Adding labels and title
plt.xlabel('Percentage of Compromised Clients (%)')
plt.ylabel('Test Accuracy')
plt.title('Test Accuracy against ' + title)

# Adding legend
plt.legend(loc=(0.05,0.05))

# Saving the plot as a PNG file
plt.savefig('utils/plot_draw/' + plot_name + '_accuracy_plot.png')

# Displaying the plot
plt.show()
