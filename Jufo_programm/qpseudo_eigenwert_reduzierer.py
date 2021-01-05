##########################################################################
## 					PSEUDO-EIGENWERT-REDUZIERER							##
##########################################################################
from build_hamiltonian import build
import numpy as np
from lauf import Lauf
import timeit
import multiprocessing as mp
from functions import Basisvektoren, print_exact, startpunkte


def alles(num_starting_points=None, hilf=[], ortho_startp=True):
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
	v = print_exact(H)
	################################################################################
	starting_points = startpunkte(
		basis_Ns, num_starting_points, ortho_startp=ortho_startp)

	for it in range(len(starting_points)):

		hilfe_schritte = [[]]
		num_steps_fwd = 0
		num_steps_bck = 0

		L = Lauf(starting_points[it], steps, H, basis_Ns, num_results, smooth,
		         picks, prob, hilf=hilf, all_steps_back=True, sim_ann=True)

		# print("steps = ", steps)
		print("Erwartungswert = ", L.eigenwert)
		# print("Vektor dazu = ", L.eigenvektor)
		# print("Calculated Error = ", L.error)
		# print("Actual Error = ", np.linalg.norm(L.eigenvektor - v))
		# print("Zeit = ", L.zeit)
		print("Average steps forward:", (L.num_steps_fwd/num_results)*100/steps, "%")
		print("Average steps missed:", (L.num_steps_bck/num_results)*100/steps, "%")
		print("Average of good steps:", (L.num_good_steps/num_results)*100/steps, "%")
		# print("step length: ", 0.2)
		# print(L.eigenwert[0])  # , " für N = ", Ll)--
		# psi = L.eigenvektor
		# print(psi.T.dot(H.dot(psi)))
		# print(np.linalg.norm(v.T[0][0]-L.eigenvektor))
		# print("Durchschnittl. Fehler: ", sum(L.debugging_list[0])/len(L.debugging_list[0]))
		print("Durchschnittl. Akzeptanzwahrscheinlichkeit: ",
		      sum(L.debugging_list[1])/len(L.debugging_list[1]))

		K = L.all_steps
		filename = "/home/hugo/Documents/Heidelberg/Quantenphysik/Jufo_Materialien/n4u2d2_s10000_ew_im_verlauf"
		open(filename, "w").close()
		with open(filename, 'w') as file_object:
			for i in range(len(K)):
				file_object.writelines(str(K[i]))
		print()
	return (L)


# Mittelhöhe, Amplitude, Activation p.
# hilfen = [[0.5, 0.9], [0.5, 0.8], [0.5, 0.7], [0.5, 0.6], [
# 	0.5, 0.5], [0.5, 0.4], [0.5, 0.3], [0.5, 0.2], [0.5, 0.1],
# 	[0.4, 0.8], [0.4, 0.7], [0.4, 0.6], [0.4, 0.5], [
# 		0.4, 0.4], [0.4, 0.3], [0.4, 0.2], [0.4, 0.1],
# 		[0.3, 0.6], [0.3, 0.5], [0.3, 0.4], [0.3, 0.3], [
# 				0.3, 0.2], [0.3, 0.1], [0.2, 0.4], [0.2, 0.3], [0.2, 0.2], [0.2, 0.1], [
# 						0.1, 0.2], [0.1, 0.1]]


L = alles(num_starting_points=1, ortho_startp=True)


# pool = mp.Pool(mp.cpu_count())
# pool.map(alles, [i for i in range(3)])
# pool.close()

################
# STUPID STUFF #
################
# import matplotlib.pyplot as plt
# import numpy as np
# from random import random, gauss
# x = np.arange(len(K[0]))
# x = list(x)
# # K[0], K[1] = K[0][::100], K[1][::100]
# # for i in range(len(K[0])):
# # 	if K[0][i] != "NaN":
# # 		x.append(i)
#
# for i in range(len(K[0])-1, 0, -1):
# 	if K[0][i] == "NaN":
# 		K[0] = K[0][:i] + K[0][i+1:]
# 		K[1] = K[1][:i] + K[1][i+1:]
# 		x = x[:i] + x[i+1:]
#
# fig, axs = plt.subplots()
# fig.tight_layout()
# # axs.scatter(x, K[0], label="Verschlechterung")
# print(np.shape(K[1]))
# axs.plot(x, x, label="Akzeptanzwahrscheinlichkeit")
# axs.legend()
#
#
# fig2, axs2 = plt.subplots()
# fig2.tight_layout()
# axs2.scatter(x, x, label="Verschlechterung")
# # axs.scatter(x, K[1], label="Akzeptanzwahrscheinlichkeit")
# axs2.legend()
#
# steps = 10000
# fig3 , axs3 = plt.subplots()
# fig3.tight_layout()
# K3 = []
# x = np.arange(steps)
# for i in range(steps):
# 	K3.append(L.curve.current(i))
# axs3.plot(x, K3)
# print(len(K[0]), steps)
# print(L.num_steps_fwd, L.num_steps_bck)
#
# plt.show()
