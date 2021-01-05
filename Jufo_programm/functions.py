import numpy as np
from random import choice, randrange, gauss, random
from scipy.constants import Boltzmann, pi


def print_exact(H):
	np.set_printoptions(precision=4, suppress=True)
	print("---------------")
	x = np.linalg.eig(H.toarray())
	v = x[1][:, np.argmin(x[0])]
	print(v)
	print(np.amin(x[0]))
	print("---------------")
	return


def Steps(steps, smooth, psi, basis_Ns, n, stepnum=1, curve=None):
	# warum brauch ich basis_Ns, wenn ich psi und damit len(psi) habe?
	# k = randrange(0, basis_Ns)
	vec = psi.copy()

	# Richtung des schrittes kann random sein, aber senkrecht zu psi
	# zufällige Richtung:
	v_richtung = np.random.rand(basis_Ns)
	# jetzt zufällige stelle benutzen, um den vektor senkrecht zu machen:
	index = randrange(0, basis_Ns)
	v_richtung[index] = 0
	v_richtung[index] = -v_richtung.dot(vec) / vec[index]

	# Haversine formula: central angle = großkreis-Abstand d / radius r
	# r = 1, daher theta = d (in radians)
	# Zudem ist step_length = d = curve.current(n) :
	theta = curve.current(n)

	# Nun den schritt auch gehen
	# Trigonometrie: tan(theta) = |vec_richtung|/|psi| , wobei |vec_richtung|
	# die gesuchte Größe ist.
	vec += (np.tan(theta) * np.linalg.norm(vec)
	        / np.linalg.norm(v_richtung)) * v_richtung

	n += 1
	return vec/np.linalg.norm(vec), n
######################################################################
# Ein tatsächlicher Schritt. Immer in eine zufällige Achsenrichtung. #
# 1 - pow((n)/(steps+smooth),3) ist hierbei eine Kurve, damit die    #
# Schritte mit der Zeit kleiner werden. Ich gehe davon aus, dass     #
# nach einer bestimmten Anzahl an Schritten der "Topf" des jeweiligen#
# Minimum bereits erreicht ist. Damit der Wert nun genau ist, wird   #
# dann nur noch mit kleinen Schritten die Genauigkeit verbessert.    #
######################################################################


class curve:
	def __init__(self, n, med=0.5, amp=0, actp=0, prt=True):
		self.n = n
		self.med = med
		self.amp = amp
		self.actp = actp
		if prt:
			print("[", med, ",", amp, ",", actp, "],")

	def current(self, i):
		if self.amp == 0:
			return (self.med)
		x = 1 - pow((i)/(self.n), 3)
		x *= self.amp
		x += self.med - 0.5*self.amp
		return (x)


def winkel(vec1, vec2):
	return(np.arccos(abs(vec1.T.dot(vec2))/(np.linalg.norm(vec1)*np.linalg.norm(vec2))))
	### Diese Methode misst den Winkel zwischen einem Vektor vor und nach    ###
	### Anwendung des Hamiltonians. 			 						     ###
	### So wird errechnet, wie nah ein Vektor an einem Eigenvektor dran ist. ###

# def Verteilung(psi):
# 	v=0
# 	l=0
# 	for i in range(len(psi)):
# 		v += psi[i]*Verteilung_zustand(i)
# 		l += psi[i]
# 	v/=l
# 	return(v)
#
#
# def Verteilung_zustand(i):
# 	v = 0
# 	for i in range(L):
# 		if ((basis[i]//2**L)//2**i)%2 + (basis[i]// 2**i)% 2  == 1:
# 			v += 1
# 	v/=L
# 	return(v)
#   ########################################################################
#   ## Diese beiden Methoden sollen einem Zustand einen Wert zwischen 0   ##
# 	## und 1 zuordnen, je nachdem wieviele einfache Besetzungen der       ##
#   ## Zustand hat. Das ist aber noch fehlerhaft. 				          ##
#   ########################################################################


def Temp(n):
	T_zero = 200
	return T_zero * 0.99**n


def Prob(diff, n, sim_ann):
	if not sim_ann:
		return 0.25
	boltz = 100000  # 1380649/16021766340
	return np.real(1 / (1 + np.exp((diff/(boltz * Temp(n))))))


def Smooth(smooth, psi, basis_Ns, n):
	for i in range(smooth):
		psi[randrange(0, basis_Ns)] += 1/n
		psi /= np.linalg.norm(psi)
		n += 1
	return psi, n
#######################################################################
# Diese Methode "rundet den Zustand ab"; sie geht Schritte, ohne zu   #
# prüfen, ob die Richtung korrekt ist. Dadurch sollte man mit einem   #
# möglichst durchmischten Zustand bereits beginnen. 				  #
#######################################################################


def Pick(picks, psi, H, basis_Ns):
	E = []
	V = []
	if picks == 0:
		return psi, psi.dot(H.dot(psi))
		# Dieser Teil verhindert bloß eine Fehlermeldung bei picks = 0.
	for i in range(picks):
		psi_new = 2 * np.random.rand(basis_Ns) - 1  # Neuer Zufallszustand
		psi_new /= np.linalg.norm(psi_new)  # Normieren
		E.append(psi_new.dot(H.dot(psi_new)))  # Energie ausrechnen
		V.append(psi_new)
		print(min(E))
	index = E.index(min(E))  # Geringste Energie heraussuchen und zurückgeben
	return V[index], E[index]
#########################################################################
# Zufälliges "Herumstochern" im Hilbertraum und auswählen des kleinsten #
# Elements. In der Theorie soll dadurch erreicht werden, dass der Walk  #
# bereits in einem möglichst tiefen Minimum beginnt. Wie gut das klappt #
# muss ich noch testen.													#
#########################################################################


def gram_schmidt_vektor(L):
	v = np.random.rand(len(L[0]))
	u = v.copy()
	for i in L:
		u -= (v.T.dot(i) / i.T.dot(i)) * i
	return u/np.linalg.norm(u)


def Basisvektoren(L, n):
	for i in range(n-len(L)):
		L.append(gram_schmidt_vektor(L))
	return L


def gleichverteilt(Laenge):
	vec = np.zeros(Laenge)
	for i in range(Laenge):
		vec[i] = gauss()
	return vec/np.linalg.norm(vec)


def startpunkte(basis_Ns, n=None, ortho_startp=True):
	if n == None:
		n = basis_Ns - 1

	if not ortho_startp:
		starting_points = [gleichverteilt(basis_Ns) for i in range(n)]
		return starting_points

	vec = np.random.rand(basis_Ns)*2 - np.ones(basis_Ns)
	starting_points = Basisvektoren([vec], n)
	return starting_points
