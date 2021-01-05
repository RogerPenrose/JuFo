import numpy as np
from random import choice, random, gauss
import timeit
from functions import Steps, Pick, Smooth, winkel, curve, Prob


class Lauf:
    def __init__(self, starting_point, number_steps, H, basis_Ns, number_results=1, smooth=0, picks=0, prob=250, hilf=[], all_steps_back=False, sim_ann=True):
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
        num_good_steps = 0
        K1 = []
        K2 = []

        start = timeit.default_timer()  # Startzeitpunkt

        for j in range(number_results):  # Der jeweilige Walk wird initialisiert: #
            psi = starting_point
            # Der erste Zustand wird initialisiert: zuerst einmal nur Nullen
            e1 = 0  # Dies wird die Energie sein
            e2 = 0
            n = 1  # Die Schritte werden zurückgesetzt
            # Nun ist "Psi" ein gültiger Zustand

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

            if len(hilf) < 3:
                if len(hilf) < 2:
                    if len(hilf) < 1:
                        hilf.append(0.5)
                    hilf.append(1)
                hilf.append(0)

            kurve = curve(number_steps, hilf[0], hilf[1], hilf[2], prt=False)

            psi_new = gs1 = psi  # gs1, genauso wie ew1, werden initialisiert
            e1 = psi.T.dot(H.dot(psi))  # Energie des Zustands

            for i in range(number_steps):  # Der aktive Teil des Walks startet:

    	        # Durch einen Schritt veränderter Zustand
                psi_new, n = Steps(number_steps, smooth,
                                   psi, basis_Ns, n, curve=kurve)
                # Auch von diesem die Energie nehmen
                e2 = psi_new.T.dot(H.dot(psi_new))

                if e1 < e2:
                    K1.append(e2-e1)
                    K2.append(Prob(e2-e1, n, sim_ann))
                else:
                    K1.append("NaN")
                    K2.append(1)

            	# Wenn die Energie nun größer ist:
            	# Mit abnehmender Wahrscheinlichkeit übernehmen
            	# Wenn die Energie nun kleiner ist:
            	# in jedem Fall übernehmen

                if e1 > e2 or random() < Prob(e2-e1, n, sim_ann):
                    psi = psi_new
                    if e1 > e2:
                        num_good_steps += 1
                    num_steps_fwd += 1
                    e1 = e2

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
    		# der tiefste Wert, der vorkam, gewählt wird.					  #
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
        self.num_good_steps = num_good_steps
        self.all_steps = all_steps
        self.debugging_list = [K1, K2]
        self.curve = kurve
