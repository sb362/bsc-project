"""
Requires: numpy, matplotlib, uncertainties, sigfig
"""

from uncertainties import ufloat as uf
from uncertainties import unumpy as unp

import numpy as np

import matplotlib.pyplot as plot

import sigfig

import warnings

"""
Α α, Β β, Γ γ, Δ δ, Ε ε, Ζ ζ, Η η, Θ θ, Ι ι, Κ κ, Λ λ, Μ μ,
Ν ν, Ξ ξ, Ο ο, Π π, Ρ ρ, Σ σ/ς, Τ τ, Υ υ, Φ φ, Χ χ, Ψ ψ, Ω ω
"""

# Plot & axis titles
x_title = "Magnetic field (G)"
y_title = "Splitting (GHz)"
plot_title = "Zeeman splitting as a function of magnetic field for neon, λ = 607.4 nm"

# Raw data

# Current (mA)
raw_x = [
	144, 314, 537, 843,
    159, 322, 514, 841,
    152, 310, 524, 838
]

# Splitting (multiples of FSR)
raw_y = [
	0.25, 0.50, 0.75, 1.00,
	0.25, 0.50, 0.75, 1.00,
	0.25, 0.50, 0.75, 1.00
]

fsr = 3e8 / (2 * 10.08e-3)
print("Free spectral range: ", fsr)

transform_x = lambda x: 249 + 8.9299 * x + 3.9277e-3 * x ** 2 - 5.5726e-6 * x ** 3
transform_y = lambda y: y * fsr

# Uncertainties
err_x = [1.00] * len(raw_x)
err_y = [0.00] * len(raw_y)

I = lambda x: x
def unrawify(raw_x, raw_y, err_x, err_y, transform_x=transform_x, transform_y=transform_y):
	if not isinstance(err_x, list): # number
		err_x = np.array(len(raw_x) * [err_x])
	if not isinstance(err_y, list): # number
		err_y = np.array(len(raw_y) * [err_y])

	x = transform_x(unp.uarray(raw_x, err_x))
	y = transform_y(unp.uarray(raw_y, err_y))

	return x, y

def linear_plot_with_error(x, y, fig_label, data_label, x_min, x_max, y_min, y_max):
	filtered = np.array(list(
		filter(
			lambda i: x_min <= i[0] <= x_max and y_min <= i[1] <= y_max,
			zip(unp.nominal_values(x), unp.nominal_values(y))
		))
	)

	if filtered.size == 0:
		warnings.warn("x- and/or y-limits are too small - using all data points for fit")

		fil_x, fil_y = unp.nominal_values(x), unp.nominal_values(y)

		x_min = min(fil_x)
		x_max = max(fil_x)
		y_min = min(fil_y)
		y_max = max(fil_y)

		set_axes_limits = False # Let matplotlib decide
	else:
		fil_x, fil_y = filtered[:, 0], filtered[:, 1]
		set_axes_limits = True

	coeff, cov = np.polyfit(fil_x, fil_y, 1, cov=True)
	fit_x = np.linspace(x_min, x_max)
	fit_y = np.polyval(coeff, fit_x)
	std_err = np.sqrt(np.diag(cov))

	print(fig_label, data_label, "fit parameters:", unp.uarray(coeff, std_err))

	fit_legend  = "m = " + sigfig.round(coeff[0], uncertainty=std_err[0]) + ", "
	fit_legend += "c = " + sigfig.round(coeff[1], uncertainty=std_err[1])

	# Create figure
	fig = plot.figure(fig_label)

	# Get current axes
	axes = fig.gca()
	axes.plot(fit_x, fit_y, linestyle='dashed', linewidth=2, markersize=12, label=fit_legend)

	# Plot with error bars
	axes.errorbar(
		unp.nominal_values(x), unp.nominal_values(y),
		xerr=unp.std_devs(x), yerr=unp.std_devs(y),
		fmt=".",
		label=data_label
	)
	axes.set_xlabel(x_title)
	axes.set_ylabel(y_title)
	axes.set_title(plot_title)

	if set_axes_limits:
		axes.set_xlim([x_min, x_max])
		axes.set_ylim([y_min, y_max])
	
	axes.minorticks_on() # Required for minor gridlines
	axes.grid(which="major")
	axes.grid(which="minor", linestyle=':', linewidth='0.5', color='black')

	plot.legend()

# Fit
x_min, x_max = 1000, 7500
y_min, y_max = 3, 16

x, y = unrawify(raw_x, raw_y, err_x, err_y)
y /= 1e9

print(x, "\n", y, "\n")
linear_plot_with_error(x, y, "Figure 1", "Data", x_min, x_max, y_min, y_max)

plot.show()
