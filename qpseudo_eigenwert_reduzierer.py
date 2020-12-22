##########################################################################
## 					PSEUDO-EIGENWERT-REDUZIERER							##
##########################################################################
from build_hamiltonian import build
import numpy as np
from lauf import Lauf
import timeit
import multiprocessing as mp


def print_exact(H):
	np.set_printoptions(precision=4, suppress=True)
	print("---------------")
	x = np.linalg.eig(H.toarray())
	v = x[1][:, np.argmin(x[0])]
	print(v)
	print("---------------")
	return v


def alles(hilf=[]):
	for it in range(1):
		hilfe_schritte = [[]]
		num_steps_fwd = 0
		num_steps_bck = 0

		#### Konstanten ####
		Ll = 4  # Länge des Systems
		N_u = Ll//2 + Ll % 2  # Elektronen mit Spin Up
		N_d = Ll//2  # Elektronen mit Spin Down
		# U = np.sqrt(2)
		# J = 1.0

		#### Variablen ####
		steps = 10000  # Die Schritte, die das Programm geht
		picks = 0  # Die Anzahl der Picks, siehe weiter unten
		prob = 250  # Die Wahrscheinlichkeit, energiereichere Zustände zu behalten
		# (In Promille)
		num_results = 1  # Anzahl Wiederholungen der Walks, damit das niedrigste
		# Ergebnis genommen werden kann
		smooth = 0  # Anzahl der "smooths", also "leerer Schritte" - siehe weiter unten

		H, basis_Ns = build(Ll, N_u, N_d)

		################################################################################

		# v = print_exact(H)

		L = Lauf(steps, H, basis_Ns, num_results, smooth,
		         picks, prob, hilf=hilf, all_steps_back=True)

		# print("steps = ", steps)
		print("Eigenwert = ", L.eigenwert)
		# print("Vektor dazu = ", L.eigenvektor)
		# print("Calculated Error = ", L.error)
		# print("Actual Error = ", np.linalg.norm(L.eigenvektor - v))
		# print("Zeit = ", L.zeit)
		# print("Average steps forward:", (L.num_steps_fwd/num_results)*100/steps, "%")
		# print("Average steps missed:", (L.num_steps_bck/num_results)*100/steps, "%")
		# print(" ")
		# print("step length: ", 0.2)
		# print(L.eigenwert[0])  # , " für N = ", Ll)
		# psi = L.eigenvektor
		# print(psi.T.dot(H.dot(psi)))
		# print(np.linalg.norm(v.T[0][0]-L.eigenvektor))

		# K = L.debugging_list
		# filename = "/home/hugo/Documents/Heidelberg/Quantenphysik/Jufo_Materialien/n4u2d2_s10000_" + \
		# 	str(hilf[0]) + "_" + str(hilf[1])
		# open(filename, "w").close()
		# with open(filename, 'w') as file_object:
		# 	for i in range(len(K)):
		# 		file_object.writelines(str(K[i]))


# Mittelhöhe, Amplitude, Activation p.
hilfen = [[0.5, 0.9], [0.5, 0.8], [0.5, 0.7], [0.5, 0.6], [
	0.5, 0.5], [0.5, 0.4], [0.5, 0.3], [0.5, 0.2], [0.5, 0.1],
	[0.4, 0.8], [0.4, 0.7], [0.4, 0.6], [0.4, 0.5], [
		0.4, 0.4], [0.4, 0.3], [0.4, 0.2], [0.4, 0.1],
		[0.3, 0.6], [0.3, 0.5], [0.3, 0.4], [0.3, 0.3], [
				0.3, 0.2], [0.3, 0.1], [0.2, 0.4], [0.2, 0.3], [0.2, 0.2], [0.2, 0.1], [
						0.1, 0.2], [0.1, 0.1]]

alles([0.1, 0.1])
# pool = mp.Pool(mp.cpu_count())
# pool.map(alles, [i for i in hilfen])
# pool.close()
