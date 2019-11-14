import numpy as np
from matplotlib import pyplot as plt
from numpy import (abs, any, arange, array, complex128, convolve, cos, exp,
                   flipud, float32, float64, floor, int64, linspace, log,
                   maximum, minimum, ndarray, pi, place, polyval, real,
                   setdiff1d, sin)
from numpy import sort as npsort
from numpy import sqrt, tan, where, zeros
from numpy.fft import fft, fftshift

# define precision
__ComplexType__ = complex128
__IntegerType__ = int64
__FloatType__ = float64

# initialize global variables
VARIABLES = {}

# ------------------ complex probability function -----------------------
# define static data
zone = __ComplexType__(1.0e0 + 0.0e0j)
zi = __ComplexType__(0.0e0 + 1.0e0j)
tt = __FloatType__([0.5e0,1.5e0,2.5e0,3.5e0,4.5e0,5.5e0,6.5e0,7.5e0,8.5e0,9.5e0,10.5e0,11.5e0,12.5e0,13.5e0,14.5e0])
pipwoeronehalf = __FloatType__(0.564189583547756e0)


w0 = 1000.
GammaD = 0.005
Gamma0 = 0.2
Gamma2 = 0.01 * Gamma0
Delta0 = 0.002
Delta2 = 0.001 * Delta0
nuVC = 0.2
eta = 0.5
Dw = 1.
ww = arange(w0-Dw, w0+Dw, 0.01) # GRID WITH THE STEP 0.01

# now l1 contains values of HT profile calculates on the grid ww


def cef(x, y, N):
	# Computes the function w(z) = exp(-zA2) erfc(-iz) using a rational
	# series with N terms. It is assumed that Im(z) > 0 or Im(z) = 0.
	z = x + 1.0j * y
	M = 2 * N;
	M2 = 2 * M;
	k = arange(-M + 1, M)  # '; # M2 = no. of sampling points.
	L = sqrt(N / sqrt(2));  # Optimal choice of L.
	theta = k * pi / M;
	t = L * tan(theta / 2);  # Variables theta and t.
	# f = exp(-t.A2)*(LA2+t.A2); f = [0; f]; # Function to be transformed.
	f = zeros(len(t) + 1);
	f[0] = 0
	f[1:] = exp(-t ** 2) * (L ** 2 + t ** 2)
	# f = insert(exp(-t**2)*(L**2+t**2),0,0)
	a = real(fft(fftshift(f))) / M2;  # Coefficients of transform.
	a = flipud(a[1:N + 1]);  # Reorder coefficients.
	Z = (L + 1.0j * z) / (L - 1.0j * z);
	p = polyval(a, Z);  # Polynomial evaluation.
	w = 2 * p / (L - 1.0j * z) ** 2 + (1 / sqrt(pi)) / (L - 1.0j * z);  # Evaluate w(z).
	return w


# weideman24 by default
#weideman24 = lambda x,y: cef(x,y,24)
weideman = lambda x,y,n: cef(x,y,n)

def hum1_wei(x,y,n=24):
    t = y-1.0j*x
    cerf=1/sqrt(pi)*t/(0.5+t**2)
    """
    z = x+1j*y
    cerf = 1j*z/sqrt(pi)/(z**2-0.5)
    """
    mask = abs(x)+y<15.0
    if any(mask):
        w24 = weideman(x[mask],y[mask],n)
        place(cerf,mask,w24)
    return cerf.real,cerf.imag

VARIABLES['CPF'] = hum1_wei
#VARIABLES['CPF'] = cpf

def pcqsdhc(sg0, GamD, Gam0, Gam2, Shift0, Shift2, anuVC, eta, sg):
    # -------------------------------------------------
	#      "pCqSDHC": partially-Correlated quadratic-Speed-Dependent Hard-Collision
	#      Subroutine to Compute the complex normalized spectral shape of an
	#      isolated line by the pCqSDHC model
	#
	#      Reference:
	#      H. Tran, N.H. Ngo, J.-M. Hartmann.
	#      Efficient computation of some speed-dependent isolated line profiles.
	#      JQSRT, Volume 129, November 2013, Pages 199–203
	#      http://dx.doi.org/10.1016/j.jqsrt.2013.06.015
	#
	#      Input/Output Parameters of Routine (Arguments or Common)
	#      ---------------------------------
	#      T          : Temperature in Kelvin (Input).
	#      amM1       : Molar mass of the absorber in g/mol(Input).
	#      sg0        : Unperturbed line position in cm-1 (Input).
	#      GamD       : Doppler HWHM in cm-1 (Input)
	#      Gam0       : Speed-averaged line-width in cm-1 (Input).
	#      Gam2       : Speed dependence of the line-width in cm-1 (Input).
	#      anuVC      : Velocity-changing frequency in cm-1 (Input).
	#      eta        : Correlation parameter, No unit (Input).
	#      Shift0     : Speed-averaged line-shift in cm-1 (Input).
	#      Shift2     : Speed dependence of the line-shift in cm-1 (Input)
	#      sg         : Current WaveNumber of the Computation in cm-1 (Input).
	#
	#      Output Quantities (through Common Statements)
	#      -----------------
	#      LS_pCqSDHC_R: Real part of the normalized spectral shape (cm)
	#      LS_pCqSDHC_I: Imaginary part of the normalized spectral shape (cm)
	#
	#      Called Routines: 'CPF'      (Complex Probability Function)
	#      ---------------  'CPF3'      (Complex Probability Function for the region 3)
	#
	#      Called By: Main Program
	#      ---------
	#
	#     Double Precision Version
	#
	# -------------------------------------------------

	# sg is the only vector argument which is passed to function

	if type(sg) not in set([array, ndarray, list, tuple]):
         sg = array([sg])

	number_of_points = len(sg)
	Aterm_GLOBAL = zeros(number_of_points, dtype=__ComplexType__)
	Bterm_GLOBAL = zeros(number_of_points, dtype=__ComplexType__)

	cte = sqrt(log(2.0e0)) / GamD
	rpi = sqrt(pi)
	iz = __ComplexType__(0.0e0 + 1.0e0j)

	c0 = __ComplexType__(Gam0 + 1.0e0j * Shift0)
	c2 = __ComplexType__(Gam2 + 1.0e0j * Shift2)
	c0t = __ComplexType__((1.0e0 - eta) * (c0 - 1.5e0 * c2) + anuVC)
	c2t = __ComplexType__((1.0e0 - eta) * c2)

	# PART1
	if abs(c2t) == 0.0e0:
		Z1 = (iz * (sg0 - sg) + c0t) * cte
		xZ1 = -Z1.imag
		yZ1 = Z1.real
		WR1, WI1 = VARIABLES['CPF'](xZ1, yZ1)
		Aterm_GLOBAL = rpi * cte * __ComplexType__(WR1 + 1.0e0j * WI1)
		index_Z1 = abs(Z1) <= 4.0e3
		index_NOT_Z1 = ~index_Z1
		if any(index_Z1):
			Bterm_GLOBAL = rpi * cte * ((1.0e0 - Z1 ** 2) * __ComplexType__(WR1 + 1.0e0j * WI1) + Z1 / rpi)
		if any(index_NOT_Z1):
			Bterm_GLOBAL = cte * (rpi * __ComplexType__(WR1 + 1.0e0j * WI1) + 0.5e0 / Z1 - 0.75e0 / (Z1 ** 3))
	else:
		# PART2, PART3 AND PART4   (PART4 IS A MAIN PART)

		# X - vector, Y - scalar
		X = (iz * (sg0 - sg) + c0t) / c2t
		Y = __ComplexType__(1.0e0 / ((2.0e0 * cte * c2t)) ** 2)
		csqrtY = (Gam2 - iz * Shift2) / (2.0e0 * cte * (1.0e0 - eta) * (Gam2 ** 2 + Shift2 ** 2))

		index_PART2 = abs(X) <= 3.0e-8 * abs(Y)
		index_PART3 = (abs(Y) <= 1.0e-15 * abs(X)) & ~index_PART2
		index_PART4 = ~ (index_PART2 | index_PART3)

		# PART4
		if any(index_PART4):
			X_TMP = X[index_PART4]
			Z1 = sqrt(X_TMP + Y) - csqrtY
			Z2 = Z1 + __FloatType__(2.0e0) * csqrtY
			xZ1 = -Z1.imag
			yZ1 = Z1.real
			xZ2 = -Z2.imag
			yZ2 = Z2.real
			SZ1 = sqrt(xZ1 ** 2 + yZ1 ** 2)
			SZ2 = sqrt(xZ2 ** 2 + yZ2 ** 2)
			DSZ = abs(SZ1 - SZ2)
			SZmx = maximum(SZ1, SZ2)
			SZmn = minimum(SZ1, SZ2)
			length_PART4 = len(index_PART4)
			WR1_PART4 = zeros(length_PART4)
			WI1_PART4 = zeros(length_PART4)
			WR2_PART4 = zeros(length_PART4)
			WI2_PART4 = zeros(length_PART4)
			index_CPF3 = (DSZ <= 1.0e0) & (SZmx > 8.0e0) & (SZmn <= 8.0e0)
			index_CPF = ~index_CPF3  # can be removed
			if any(index_CPF3):
				WR1, WI1 = cpf3(xZ1[index_CPF3], yZ1[index_CPF3])
				WR2, WI2 = cpf3(xZ2[index_CPF3], yZ2[index_CPF3])
				WR1_PART4[index_CPF3] = WR1
				WI1_PART4[index_CPF3] = WI1
				WR2_PART4[index_CPF3] = WR2
				WI2_PART4[index_CPF3] = WI2
			if any(index_CPF):
				WR1, WI1 = VARIABLES['CPF'](xZ1[index_CPF], yZ1[index_CPF])
				WR2, WI2 = VARIABLES['CPF'](xZ2[index_CPF], yZ2[index_CPF])
				WR1_PART4[index_CPF] = WR1
				WI1_PART4[index_CPF] = WI1
				WR2_PART4[index_CPF] = WR2
				WI2_PART4[index_CPF] = WI2

			Aterm = rpi * cte * (__ComplexType__(WR1_PART4 + 1.0e0j * WI1_PART4) - __ComplexType__(
				WR2_PART4 + 1.0e0j * WI2_PART4))
			Bterm = (-1.0e0 +
			         rpi / (2.0e0 * csqrtY) * (1.0e0 - Z1 ** 2) * __ComplexType__(WR1_PART4 + 1.0e0j * WI1_PART4) -
			         rpi / (2.0e0 * csqrtY) * (1.0e0 - Z2 ** 2) * __ComplexType__(WR2_PART4 + 1.0e0j * WI2_PART4)) / c2t
			Aterm_GLOBAL[index_PART4] = Aterm
			Bterm_GLOBAL[index_PART4] = Bterm

		# PART2
		if any(index_PART2):
			X_TMP = X[index_PART2]
			Z1 = (iz * (sg0 - sg[index_PART2]) + c0t) * cte
			Z2 = sqrt(X_TMP + Y) + csqrtY
			xZ1 = -Z1.imag
			yZ1 = Z1.real
			xZ2 = -Z2.imag
			yZ2 = Z2.real
			WR1_PART2, WI1_PART2 = VARIABLES['CPF'](xZ1, yZ1)
			WR2_PART2, WI2_PART2 = VARIABLES['CPF'](xZ2, yZ2)
			Aterm = rpi * cte * (__ComplexType__(WR1_PART2 + 1.0e0j * WI1_PART2) - __ComplexType__(
				WR2_PART2 + 1.0e0j * WI2_PART2))
			Bterm = (-1.0e0 +
			         rpi / (2.0e0 * csqrtY) * (1.0e0 - Z1 ** 2) * __ComplexType__(WR1_PART2 + 1.0e0j * WI1_PART2) -
			         rpi / (2.0e0 * csqrtY) * (1.0e0 - Z2 ** 2) * __ComplexType__(WR2_PART2 + 1.0e0j * WI2_PART2)) / c2t
			Aterm_GLOBAL[index_PART2] = Aterm
			Bterm_GLOBAL[index_PART2] = Bterm

		# PART3
		if any(index_PART3):
			X_TMP = X[index_PART3]
			xZ1 = -sqrt(X_TMP + Y).imag
			yZ1 = sqrt(X_TMP + Y).real
			WR1_PART3, WI1_PART3 = VARIABLES['CPF'](xZ1, yZ1)
			index_ABS = abs(sqrt(X_TMP)) <= 4.0e3
			index_NOT_ABS = ~index_ABS
			Aterm = zeros(len(index_PART3), dtype=__ComplexType__)
			Bterm = zeros(len(index_PART3), dtype=__ComplexType__)
			if any(index_ABS):
				xXb = -sqrt(X).imag
				yXb = sqrt(X).real
				WRb, WIb = VARIABLES['CPF'](xXb, yXb)
				Aterm[index_ABS] = (2.0e0 * rpi / c2t) * (
							1.0e0 / rpi - sqrt(X_TMP[index_ABS]) * __ComplexType__(WRb + 1.0e0j * WIb))
				Bterm[index_ABS] = (1.0e0 / c2t) * (-1.0e0 +
				                                    2.0e0 * rpi * (1.0e0 - X_TMP[index_ABS] - 2.0e0 * Y) * (
							                                    1.0e0 / rpi - sqrt(X_TMP[index_ABS]) * __ComplexType__(
						                                    WRb + 1.0e0j * WIb)) +
				                                    2.0e0 * rpi * sqrt(X_TMP[index_ABS] + Y) * __ComplexType__(
							WR1_PART3 + 1.0e0j * WI1_PART3))
			if any(index_NOT_ABS):
				Aterm[index_NOT_ABS] = (1.0e0 / c2t) * (
							1.0e0 / X_TMP[index_NOT_ABS] - 1.5e0 / (X_TMP[index_NOT_ABS] ** 2))
				Bterm[index_NOT_ABS] = (1.0e0 / c2t) * (-1.0e0 + (1.0e0 - X_TMP[index_NOT_ABS] - 2.0e0 * Y) *
				                                        (1.0e0 / X_TMP[index_NOT_ABS] - 1.5e0 / (
							                                        X_TMP[index_NOT_ABS] ** 2)) +
				                                        2.0e0 * rpi * sqrt(X_TMP[index_NOT_ABS] + Y) * __ComplexType__(
							WR1 + 1.0e0j * WI1))
			Aterm_GLOBAL[index_PART3] = Aterm
			Bterm_GLOBAL[index_PART3] = Bterm

	# common part
	LS_pCqSDHC = (1.0e0 / pi) * (
				Aterm_GLOBAL / (1.0e0 - (anuVC - eta * (c0 - 1.5e0 * c2)) * Aterm_GLOBAL + eta * c2 * Bterm_GLOBAL))
	return LS_pCqSDHC.real, LS_pCqSDHC.imag


# "naive" implementation for benchmarks
def cpf3(X, Y):
	# X,Y,WR,WI - numpy arrays
	if type(X) != ndarray:
		if type(X) not in set([list, tuple]):
			X = array([X])
		else:
			X = array(X)
	if type(Y) != ndarray:
		if type(Y) not in set([list, tuple]):
			Y = array([Y])
		else:
			Y = array(Y)

	zm1 = zone / __ComplexType__(X + zi * Y)  # maybe redundant
	zm2 = zm1 ** 2
	zsum = zone
	zterm = zone

	for tt_i in tt:
		zterm *= zm2 * tt_i
		zsum += zterm

	zsum *= zi * zm1 * pipwoeronehalf

	return zsum.real, zsum.imag

l1 = pcqsdhc(w0,GammaD,Gamma0,Gamma2,Delta0,Delta2,nuVC,eta,ww)[0]

plt.plot(ww,l1)
plt.show()
