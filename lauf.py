import numpy as np
from random import choice, randrange, gauss
import timeit
from functions import Steps, Pick, Smooth, winkel, curve, Prob


class Lauf:
    def __init__(self, number_steps, H, basis_Ns, number_results=1, smooth=0, picks=0, prob=250, hilf=[], all_steps_back=False):
        #################################################
        #     VARIABLES
        #################################################
        ew2 = []   # Es gibt jeweils drei Stufen, in denen das Programm aussieben
        ew3 = []   # soll: nach jedem Schritt (ew1), nach jedem Walk(ew2) und
        # nach allen num_results Durchgängen. Das gleiche gilt für gs.
        gs2 = []
        gs3 = []   # gs = GroundStates. Speichert die energieärmsten Zustände.

        all_steps = []  # Hilfsliste mit allen Einzelschritten

        error = []  # Abweichung von einem Eigenwert
        s = []     # Zeit in Sekunden

        num_steps_bck = 0
        num_steps_fwd = 0
        K = []

        start = timeit.default_timer()  # Startzeitpunkt

        for j in range(number_results):  # Der jeweilige Walk wird initialisiert: #
            psi = np.zeros(basis_Ns)  # , dtype=complex)
            # Der erste Zustand wird initialisiert: zuerst einmal nur Nullen
            e1 = 0  # Dies wird die Energie sein
            e2 = 0
            n = 1  # Die Schritte werden zurückgesetzt
            # Nun ist "Psi" ein gültiger Zustand

            psi[randrange(0, len(psi))] = choice((1, -1))
            psi, ew1 = Pick(picks, psi, H, basis_Ns)

    	    # Zuerst die Picks, damit "Smooth" nicht überschrieben wird
    	    # An dieser Stelle wird "Smooth" aufgerufen
            psi, n = Smooth(smooth,  psi, basis_Ns, n)

    	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#
    	# Zurzeit ergibt es keinen Sinn, Smooth zu verwenden, wenn Picks verwendet #
    	# wird. Der einzige Vorteil, Smooth zu verwenden, ist also um die Rechen-  #
    	# zeit zu drücken, wenn (zu Testzwecken z.B.) die Genauigkeit der Ergeb-   #
    	# nisse nicht wichtig ist.												   #
    	#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!#

            psi_new = gs1 = psi  # gs1, genauso wie ew1, werden initialisiert

            if len(hilf) < 3:
                if len(hilf) < 2:
                    hilf.append(1)
                hilf.append(0)

            kurve = curve(number_steps, hilf[0], hilf[1], hilf[2], prt=False)

            for i in range(number_steps):  # Der aktive Teil des Walks startet:
                e1 = psi.T.dot(H.dot(psi))  # Energie des Zustands
    	        # Durch einen Schritt veränderter Zustand
                psi_new, n = Steps(number_steps, smooth,
                                   psi, basis_Ns, n, curve=kurve)
                # Auch von diesem die Energie nehmen
                e2 = psi_new.T.dot(H.dot(psi_new))

    	# Wenn die Energie nun größer ist:
    	# Mit abnehmender Wahrscheinlichkeit übernehmen
    	# Wenn die Energie nun kleiner ist:
    	# in jedem Fall übernehmen
                if e1 > e2 or randrange(0, 1000) < Prob(e2-e1, n):
                    psi = psi_new
                    e1 = e2
                    # hilfe_schritte[j].append(np.linalg.norm(v-psi.T))
                    num_steps_fwd += 1
    	            # print(np.linalg.norm(psi-v))
                    K.append(psi)
                else:
                    n -= 1
                    num_steps_bck += 1

                if all_steps_back:
                    all_steps.append(e1)

    		########################
    		# Die Rückfallvariable #
    		########################
                if min(ew1, e1) == e1:
                    ew1 = e1
                    gs1 = psi
    		###################################################################
    		# ew1 speichert so immer den geringsten Wert, der vorkam. Daher   #
    		# ist garantiert dass, falls ein Walk aus einem Minimum ins       #
    		# andere laufen und sich das als Fehler erweisen sollte, trotzdem #
    		# der tiefste Wert der vorkam gewählt wird.						  #
    		###################################################################
            ew2.append(ew1)  # Werte und die dazugehörigen #
            gs2.append(gs1)  # Zustände sammeln            #

        stop = timeit.default_timer()  # Endzeit
        s.append(stop-start)
        index = ew2.index(min(ew2))  # Den geringsten Wert suchen #
        ew3.append(ew2[index])

        gs3.append(gs2[index]/np.linalg.norm(gs2[index]))
        # Den dazugehörigen Vetor normieren und speichern #
        error.append(winkel(gs2[index], H.dot(gs2[index])))
        # Mit ihm einen Richtwert zur Abweichung von einem Eigenwert errechnen #
        self.error = error
        self.eigenwert = ew3
        self.eigenvektor = gs3[0]
        self.zeit = s
        self.num_steps_fwd = num_steps_fwd
        self.num_steps_bck = num_steps_bck
        self.all_steps = all_steps
        self.debugging_list = K
