import pandas as pd
import numpy as np

# Load the Excel file into a pandas DataFrame
df = pd.read_excel('Data.xlsx', sheet_name='Sheet1')

# Convert each column into a numpy array and skip the first row (assuming it's headers)
circle_number = df['Circle Number']
width = df['Width']
height = df['Height']
s = df['S']
factor = df['Factor']
square_width = df['Square Width']
square_height = df['Square Height']

# create a dataframe for the measure error which is 0.25 that is the same size as the other columns
measure_error = np.full(len(circle_number), 0.25)


new_width = width * (factor / square_width) * s
new_height = height * (factor / square_height) * s 






# Apply the error propagation formula
relative_error_width = measure_error / width
relative_error_square_width = measure_error / square_width

# Total relative error for each new length
total_relative_error = np.sqrt(relative_error_width**2 + relative_error_square_width**2)

# Calculate the propagated error in the new length
error_new_width = new_width * total_relative_error





# error propagation formula

# Apply the error propagation formula
relative_error_height = measure_error / height
relative_error_square_height = measure_error / square_height

# Total relative error for each new length
total_relative_error = np.sqrt(relative_error_height**2 + relative_error_square_height**2)

# Calculate the propagated error in the new length
error_new_height = new_height * total_relative_error



import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data (replace these with your actual x and y lists)
x_values = new_width
y_values = new_height

# Define the Gaussian function
def gaussian_function(x, y_mu, mu, sigma):
    return y_mu * np.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

# Fit the Gaussian function to the data
params_gaussian, covariance_matrix = curve_fit(gaussian_function, x_values, y_values, p0=[max(y_values), np.mean(x_values), 1])

# Extract the fitted parameters for Gaussian
y_mu_gaussian, mu_gaussian, sigma_gaussian = params_gaussian

# Generate the fitted curves
x_fitted_curve = np.linspace(min(x_values), max(x_values), 1000)
y_fitted_gaussian = gaussian_function(x_fitted_curve, y_mu_gaussian, mu_gaussian, sigma_gaussian)

# make the plot bigger
plt.figure(figsize=(10, 6))

# Plot the original scatter points
plt.scatter(x_values, y_values, color='blue', label='Data points')

# Plot error bars
plt.errorbar(x_values, y_values, xerr=error_new_width, yerr=error_new_height, fmt='o', color='red', alpha=0.3)

# Plot the fitted Gaussian distribution curve
plt.plot(x_fitted_curve, y_fitted_gaussian, 'r-', color='orange', label=f'Fitted Gaussian: $y(x)= {y_mu_gaussian:.2f} e^{{-(x-{mu_gaussian:.2f})^2/(2*{sigma_gaussian:.2f}^2)}}$')

# Customize the plot
plt.title('Oval Width vs Height Gaussian Curve Fitting')
plt.xlabel('Calibrated Oval Width')
plt.ylabel('Calibrated Oval Height')
plt.legend()

# Show the plot
plt.show()




param_errors = np.sqrt(np.diag(covariance_matrix))
error_y_mu_gaussian = param_errors[0]
error_mu_gaussian = param_errors[1]
error_sigma_gaussian = param_errors[2]

print('Fitted y_mu:', y_mu_gaussian)
print('Fitted mu:', mu_gaussian)
print('Fitted sigma:', sigma_gaussian)
print('---------------------------------')
print('Error y_mu:', error_y_mu_gaussian)
print('Error mu:', error_mu_gaussian)
print('Error sigma:', error_sigma_gaussian)







# Calculate the Chi-squared value
chi_squared = np.sum(((y_values - gaussian_function(x_values, y_mu_gaussian, mu_gaussian, sigma_gaussian))) ** 2  / gaussian_function(x_values, y_mu_gaussian, mu_gaussian, sigma_gaussian))
print('Chi-squared:', chi_squared)







import scipy.stats as stats
import scipy.special

cdf = scipy.special.chdtrc(9,chi_squared)
print('CDF:', cdf)



# Make the residuals plot
plt.figure(figsize=(10, 6))
residuals = y_values - gaussian_function(x_values, y_mu_gaussian, mu_gaussian, sigma_gaussian)

# find the residual error
residual_error = np.sqrt(error_new_height**2 + 0.25**2)
plt.errorbar(x_values, residuals, yerr=residual_error, fmt='o', color='red', alpha=0.3)

plt.scatter(x_values, residuals, color='blue')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Calibrated Oval Width')
plt.ylabel('Residuals')
plt.show()




import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# Data (replace these with your actual x and y lists)
x_values = new_width
y_values = new_height

# Define the Lognormal function
def lognormal_function(x, y_mu, mu, sigma):
    return (y_mu / (x * sigma * np.sqrt(2 * np.pi))) * np.exp(-((np.log(x) - mu) ** 2) / (2 * sigma ** 2))

# Fit the Lognormal function to the data
params_lognormal, covariance_matrix = curve_fit(lognormal_function, x_values, y_values, p0=[max(y_values), np.mean(np.log(x_values)), 1])

# Extract the fitted parameters for Lognormal
y_mu_lognormal, mu_lognormal, sigma_lognormal = params_lognormal

# Generate the fitted curves
x_fitted_curve = np.linspace(min(x_values), max(x_values), 1000)
y_fitted_lognormal = lognormal_function(x_fitted_curve, y_mu_lognormal, mu_lognormal, sigma_lognormal)

# make the plot bigger
plt.figure(figsize=(10, 6))

# Plot the original scatter points
plt.scatter(x_values, y_values, color='blue', label='Data points')

# Plot error bars
plt.errorbar(x_values, y_values, xerr=error_new_width, yerr=error_new_height, fmt='o', color='red', alpha=0.3)

# Plot the fitted Lognormal distribution curve
plt.plot(x_fitted_curve, y_fitted_lognormal, 'g-', color='orange', label=f'Fitted Lognormal: $y(x)= {y_mu_lognormal:.2f} / (x \cdot {sigma_lognormal:.2f} \cdot \sqrt{{2\pi}}) e^{{-(\ln(x)-{mu_lognormal:.2f})^2/(2*{sigma_lognormal:.2f}^2)}}$')

# Customize the plot
plt.title('Oval Width vs Height Lognormal Curve Fitting')
plt.xlabel('Calibrated Oval Width')
plt.ylabel('Calibrated Oval Height')
plt.legend()

# Show the plot
plt.show()





# Calculate the parameter errors
param_errors = np.sqrt(np.diag(covariance_matrix))
error_y_mu_lognormal = param_errors[0]
error_mu_lognormal = param_errors[1]
error_sigma_lognormal = param_errors[2]

print('Fitted y_mu:', y_mu_lognormal)
print('Fitted mu:', mu_lognormal)
print('Fitted sigma:', sigma_lognormal)
print('---------------------------------')
print('Error y_mu:', error_y_mu_lognormal)
print('Error mu:', error_mu_lognormal)
print('Error sigma:', error_sigma_lognormal)




# Calculate the Chi-squared value
chi_squared = np.sum(((y_values - lognormal_function(x_values, y_mu_lognormal, mu_lognormal, sigma_lognormal))) ** 2 / lognormal_function(x_values, y_mu_lognormal, mu_lognormal, sigma_lognormal))
print('Chi-squared:', chi_squared)




cdf = scipy.special.chdtrc(9,chi_squared)
print('CDF:', cdf)



# Make the residuals plot
plt.figure(figsize=(10, 6))
residuals = y_values - lognormal_function(x_values, y_mu_lognormal, mu_lognormal, sigma_lognormal)

# plot residuals error
residual_error = np.sqrt(error_new_height**2 + error_new_width**2)
plt.errorbar(x_values, residuals, yerr=residual_error, fmt='o', color='red', alpha=0.3)

plt.scatter(x_values, residuals, color='blue')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Calibrated Oval Width')
plt.ylabel('Residuals')
plt.show()



# Data (replace these with your actual x and y lists)
x_values = new_width
y_values = new_height

# Define the Laplacian function
def laplace_function(x, y_mu, mu, sigma):
    return y_mu * np.exp(-np.abs(x - mu) / sigma)

# Fit the Laplacian function to the data points
params, covariance_matrix = curve_fit(laplace_function, x_values, y_values, p0=[max(y_values), np.mean(x_values), 1])

# Extract the fitted parameters
y_mu_fitted, mu_fitted, sigma_fitted = params

# Generate the fitted curve
x_fitted_curve = np.linspace(min(x_values), max(x_values), 1000)
y_fitted_curve = laplace_function(x_fitted_curve, y_mu_fitted, mu_fitted, sigma_fitted)

# make the plot bigger
plt.figure(figsize=(10, 6))

# Plot the original scatter points
plt.scatter(x_values, y_values, color='blue', label='Data points')

# Plot the fitted Laplace distribution curve
plt.plot(x_fitted_curve, y_fitted_curve, 'r-', 
         label=f'Fitted Laplace: $y(x)= {y_mu_fitted:.2f} e^{{-|x-{mu_fitted:.2f}|/{sigma_fitted:.2f}}}$', color='orange')

#plot the error bars but make error bars fainter than the data points
plt.errorbar(x_values, y_values, xerr=error_new_width, yerr=error_new_height, fmt='o', color='red', alpha=0.3)
# Customize the plot
plt.title('Oval Width vs Height Laplacian Curve Fitting')
plt.xlabel('Oval Width Calibrated')
plt.ylabel('Oval Height Calibrated')
plt.legend()

# Show the plot
plt.show()




# Calculate the parameter errors
param_errors = np.sqrt(np.diag(covariance_matrix))
error_y_mu = param_errors[0]
error_mu = param_errors[1]
error_sigma = param_errors[2]

print('Fitted y_mu:', y_mu_fitted)
print('Fitted mu:', mu_fitted)
print('Fitted sigma:', sigma_fitted)
print('---------------------------------')
print('Error y_mu:', error_y_mu)
print('Error mu:', error_mu)
print('Error sigma:', error_sigma)


# Calculate the Chi-squared value
chi_squared = np.sum(((y_values - laplace_function(x_values, y_mu_fitted, mu_fitted, sigma_fitted))) ** 2 /  laplace_function(x_values, y_mu_fitted, mu_fitted, sigma_fitted))
print('Chi-squared:', chi_squared)


cdf = scipy.special.chdtrc(9,chi_squared)


# Make the residuals plot
plt.figure(figsize=(10, 6))
residuals = y_values - laplace_function(x_values, y_mu_fitted, mu_fitted, sigma_fitted)

# calculate the residual error bars
residual_error = np.sqrt(error_new_width**2)

plt.errorbar(x_values, residuals, yerr=residual_error, fmt='o', color='red', alpha=0.3)

plt.scatter(x_values, residuals, color='blue')
plt.axhline(y=0, color='black', linestyle='--')
plt.title('Residuals Plot')
plt.xlabel('Calibrated Oval Width')
plt.ylabel('Residuals')
plt.show()









