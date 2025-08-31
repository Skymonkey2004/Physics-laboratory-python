#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 01:50:41 2025

@author: Yash
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Constants
d = 3e-1  # FPI length in cm (3 mm)
conversion_factor = 0.1  # T/A for Phywe magnets

# Red light data
D1_red = 1.6167e-6  # Diameter D1 in meters
D2_red = 4.5523e-6  # Diameter D2 in meters

# Blue light data
D1_blue = 2.8477e-6  # Diameter D1 in meters
D2_blue = 4.7036e-6  # Diameter D2 in meters

# Red light measurements (Table 1)
current_red = np.array([1.0, 1.25, 1.75, 2.26, 2.75, 3.25, 3.75, 3.99])  # Current in A
B_red = current_red * conversion_factor  # Magnetic field in T
Da_red = np.array([4.6800, 4.7231, 4.7934, 4.8858, 4.9063, 5.0253, 5.0776, 5.0644]) * 1e-6  # Diameter Da in meters
Db_red = np.array([4.3952, 4.3344, 4.2393, 4.1652, 4.1109, 4.0367, 4.0113, 3.9634]) * 1e-6  # Diameter Db in meters

# Blue light measurements (Table 2)
current_blue = np.array([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])  # Current in A
B_blue = current_blue * conversion_factor  # Magnetic field in T
Da_blue = np.array([4.8263, 4.9360, 4.9577, 5.0353, 5.3353, 5.3035, 5.2409]) * 1e-6  # Diameter Da in meters
Db_blue = np.array([4.3907, 4.3190, 4.8250, 4.1154, 3.9264, 3.8376, 3.6765]) * 1e-6  # Diameter Db in meters

# Function to calculate Zeeman splitting (Delta nu)
def calculate_zeeman_splitting(Da, Db, D1, D2, d):
    return (1 / (2 * d)) * ((Da**2 - Db**2) / (D2**2 - D1**2))

red_g_J = 1
blue_g_J = 2

def zeeman_splitting_an(B, g_J):
    mu_B = 5.7883818060 * 1e-5  # in eV/T
    E = []
    for b in B:
        E_mag = 2* g_J * mu_B * b * 8065.54  # Convert eV to cm^-1
        E.append(E_mag)
    return np.array(E)

# Calculate Zeeman splitting for red and blue light
delta_nu_red = calculate_zeeman_splitting(Da_red, Db_red, D1_red, D2_red, d)
delta_nu_blue = calculate_zeeman_splitting(Da_blue, Db_blue, D1_blue, D2_blue, d)

# Perform linear fits
slope_red, intercept_red, _, _, _ = linregress(B_red, delta_nu_red)
slope_blue, intercept_blue, _, _, _ = linregress(B_blue, delta_nu_blue)

# Calculate the ratio of slopes
slope_ratio = slope_red / slope_blue

delta_nu_red_an = zeeman_splitting_an(B_red, red_g_J)
delta_nu_blue_an = zeeman_splitting_an(B_blue, blue_g_J)

slope_red_an, intercept_red_an, _, _, _ = linregress(B_red, delta_nu_red_an)
slope_blue_an, intercept_blue_an, _, _, _ = linregress(B_blue, delta_nu_blue_an)

# Function to calculate standard deviation
def standard_deviation(y_measured, y_fit):
    residuals = y_measured - y_fit
    return np.sqrt(np.sum(residuals**2) / len(y_measured))

# Calculate standard deviation for experimental fits
y_fit_red = slope_red * B_red + intercept_red
y_fit_blue = slope_blue * B_blue + intercept_blue
std_dev_red = standard_deviation(delta_nu_red, y_fit_red)
std_dev_blue = standard_deviation(delta_nu_blue, y_fit_blue)


# Print the results
print(f"Slope for red light (K_red): {slope_red}")
print(f"Slope for blue light (K_blue): {slope_blue}")
print(f"Ratio of slopes (K_red / K_blue): {slope_ratio}")
print(f"Standard deviation for red light (experimental): {std_dev_red}")
print(f"Standard deviation for blue light (experimental): {std_dev_blue}")

# Plot the results
plt.figure(figsize=(10, 6))

# Plot analytical fits
plt.plot(B_red, delta_nu_red_an, marker="x", color="red",
        label=f"analytical red: y = {slope_red_an:.3f}x + {intercept_red_an:.3f}")
plt.plot(B_blue, delta_nu_blue_an, marker="x", color="blue",
        label=f"analytical blue: y = {slope_blue_an:.3f}x + {intercept_blue_an:.3f}")

# Red light plot
plt.plot(B_red, delta_nu_red, 'ro', label='Red Light Data')
plt.plot(B_red, slope_red * B_red + intercept_red, 'r-', label=f'Red Light Fit: $K_{{red}} = {slope_red:.2f}$')

# Blue light plot
plt.plot(B_blue, delta_nu_blue, 'bo', label='Blue Light Data')
plt.plot(B_blue, slope_blue * B_blue + intercept_blue, 'b-', label=f'Blue Light Fit: $K_{{blue}} = {slope_blue:.2f}$')

# Labels and title
plt.xlabel('Magnetic Field $B$ (T)')
plt.ylabel(r"Zeeman splitting $\Delta \nu$ [cm$^{-1}$]")
plt.title('Zeeman Splitting experimental & analytical vs Magnetic Field')
plt.legend()
plt.grid(True)
plt.show()