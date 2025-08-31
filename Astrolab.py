#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  2 18:14:16 2025

@author: Yash
"""


import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import trapz
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Load the spectrum data
# Assuming the spectrum data is in two columns: wavelength (Å) and flux
wavelength, flux = np.loadtxt('//Users/Yash/Downloads/FYSB 23/Astro lab/spectrum.dat', unpack=True)

# Define the wavelength range of interest
wavelength_min = 6392
wavelength_max = 6402

# Filter the data for the specified range
mask = (wavelength >= wavelength_min) & (wavelength <= wavelength_max)
wavelength_range = wavelength[mask]
flux_range = flux[mask]

# Plot the spectrum
plt.figure(figsize=(10, 6))
plt.plot(wavelength_range, flux_range, label='Spectrum')
plt.xlabel('Wavelength (Å)' , fontsize = 14)
plt.ylabel('Flux' , fontsize = 14)
plt.title('Spectrum between 6392 Å and 6402 Å', fontsize = 14)
plt.grid(True)

# Mark the prominent iron lines
plt.axvline(x=6393.6, color='r', linestyle='--', label='6393.6 Å Fe line')
plt.axvline(x=6400, color='g', linestyle='--', label='6400 Å Fe line')

plt.legend()
plt.show()

# Determine which line to use for analysis
# Typically, the line with the deeper absorption (lower flux) is chosen for analysis
flux_at_6393_6 = flux_range[np.argmin(np.abs(wavelength_range - 6393.6))]
flux_at_6400 = flux_range[np.argmin(np.abs(wavelength_range - 6400))]

if flux_at_6393_6 < flux_at_6400:
    chosen_line = "6393.6 Å"
    reason = "deeper absorption (lower flux)"
else:
    chosen_line = "6400 Å"
    reason = "deeper absorption (lower flux)"

print(f"Chosen line for analysis: {chosen_line} because it has {reason}.")

# Load the spectrum data from the file
# Assuming the file has two columns: wavelength (Å) and flux
data = np.loadtxt('/Users/Yash/Downloads/FYSB 23/Astro lab/spectrum.dat')  # Replace 'spectrum.dat' with the actual file path
wavelength = data[:, 0]  # First column: wavelength
flux = data[:, 1]  # Second column: flux

# Define the wavelength range for the line (6393.0 Å to 6394.0 Å)
lambda_min = 6393.18
lambda_max = 6394.02

# Filter the data to the specified wavelength range
mask = (wavelength >= lambda_min) & (wavelength <= lambda_max)
wavelength_range = wavelength[mask]
flux_range = flux[mask]

# Calculate the integrand: 1 - F(lambda)
integrand = 1 - flux_range

# Calculate the line strength S using the trapezoidal rule
S_trapezoidal = np.trapz(integrand, wavelength_range)


# Print the results
print(f"Line strength S (trapezoidal rule): {S_trapezoidal}")

# Load the spectrum data from the file
data = np.loadtxt("/Users/Yash/Downloads/FYSB 23/Astro lab/spectrum.dat")
wavelength_spectrum = data[:, 0]  # Wavelength in Å
flux_spectrum = data[:, 1]  # Normalized flux

# Define the line list data
line_list = np.array([
    [6393.6004, -1.452, 2.433],
    [6416.9331, -1.109, 4.796],
    [6430.8450, -2.005, 2.176],
    [6481.8698, -2.981, 2.279],
    [6494.9804, -1.268, 2.404],
    [6496.4656, -0.530, 4.796],
    [6498.9383, -4.687, 0.958],
    [6518.3657, -2.438, 2.832],
    [6533.9281, -1.360, 4.559],
    [6546.2381, -1.536, 2.759],
    [6574.2266, -5.004, 0.990]
])



# Function to compute line strength with updated delta_lambda
def compute_line_strength(wavelength, delta_lambda=0.42):
    # Filter the spectrum data to the specified wavelength range
    mask = (wavelength_spectrum >= (wavelength - delta_lambda)) & (wavelength_spectrum <= (wavelength + delta_lambda))
    if np.sum(mask) == 0:
        return 0
    S = trapz(1 - flux_spectrum[mask], wavelength_spectrum[mask])
    return abs(S)

# Calculate line strengths
line_strengths = []
for line in line_list:
    lambda_0 = line[0]
    S = compute_line_strength(lambda_0)
    line_strengths.append([lambda_0, S])

line_strengths = np.array(line_strengths)

# Display line strengths with units
print("Line Strengths (Wavelength, S [Å]):")
print(line_strengths)

chi = line_list[:, 2]
log_gf = line_list[:, 1]


# Calculate log(S/λ) using the line strengths and central wavelengths
log_S_lambda = np.log10(line_strengths[:, 1] / line_strengths[:, 0])  # log(S/λ)

# Calculate log(gf * λ) using the central wavelengths
log_gf_lambda = log_gf + np.log10(line_strengths[:, 0])  # log(gf * λ)

# Corrected y-axis term: log(S/λ) - log(gf * λ)
log_S_lambda_gf = log_S_lambda - log_gf_lambda

# Linear Regression
slope, intercept, r_value, p_value, std_err = linregress(chi, log_S_lambda_gf)
T = -np.log10(np.e) / (slope * 8.617e-5)

# Plot
plt.figure(figsize=(8, 5))
plt.scatter(chi, log_S_lambda_gf, label="Data Points", color="blue")
plt.plot(chi, slope * chi + intercept, label=f"Fit: Slope = {slope:.4f}", color="red")
plt.xlabel("Excitation Potential (χ) [eV]", fontsize=12)
plt.ylabel(r"$\log_{10} (S/\lambda) - \log_{10} (gf)$", fontsize=12)
plt.title("Task 6: Excitation Potential vs. Spectral Line Strength", fontsize=12)
plt.legend()
plt.grid()
plt.show()

print(f"Temperature: {T:.2f} K")