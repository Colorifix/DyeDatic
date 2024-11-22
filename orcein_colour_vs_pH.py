import matplotlib.pyplot as plt
from src.ionisation import *

if __name__ == "__main__":
    """
    Equation for systematic error removal
    E_actual (in eV) = 0.397 + 0.731  * E_vert (in eV) for wB97X-D4  

    Hydroxyorcein |    Calculated energies      | Energies after systematic error removal
    S1111         |  2.609 (0.59); 3.279 (0.28) | 2.304 (0.59); 2.794 (0.28) 
    S1101         |                3.112 (0.85) | 2.679 (0.85)
    S1100         |                2.615 (1.15) | 2.308 (1.15)
    S0100         |                2.531 (1.19) | 2.246 (1.19)
    S0000         |  2.465 (0.86); 2.795 (0.31) | 2.198 (0.86); 2.44 (0.31)
    
    Aminoorcein   |    Calculated energies      | Energies after systematic error removal
    S1111         |  2.747 (0.24); 3.086 (0.56) | 2.405 (0.24); 3.026 (0.55)
    S1101         |                3.283 (0.52) | 2.797 (0.52)
    S1100         |                2.868 (1.07) | 2.493 (1.07)
    S0100         |                2.727 (1.09) | 2.390 (1.09)
    S1000         |                2.772 (1.11) | 2.442 (1.11)
    S0000         |  2.719 (1.05); 3.004 (0.03) | 2.384 (1.05); 2.593 (0.03)
    
    
    Orcein is a mixture of compounds. 
    For this work I considered two compounds: alpha-hydroxyorcein and alpha-aminoorcein
    Colour is predicted separately for both compounds and to 1:1 mixture of them
    
    """


    mol1 = Molecule("hydroxyorcein",
                    transition_energies=[(0, [2.304, 2.794]),  (2, [2.679]), (1, [2.308]),
                                         (3, [2.246]), (4, [2.198, 2.44])],
                                                                                                         # See Scheme S4 for more info
                    protonated_species=[(4, "Cc1cc(O)cc(O)c1c4c(O)cc3oc2cc(=O)cc(C)c2[nH+]c3c4C"),       # S1111
                                        (3, "Cc1cc(O)cc(O)c1c4c(O)cc3oc2cc(=O)cc(C)c2nc3c4C"),           # S1101
                                        (2, "Cc1cc(O)cc(O)c1c4c([O-])cc3oc2cc(=O)cc(C)c2nc3c4C"),        # S1100
                                        (1, "Cc1cc([O-])cc(O)c1c4c([O-])cc3oc2cc(=O)cc(C)c2nc3c4C"),     # S0100
                                        (0, "Cc1cc([O-])cc([O-])c1c4c([O-])cc3oc2cc(=O)cc(C)c2nc3c4C")], # S0000

                    oscillator_strengths=[(0, [0.59, 0.28]), (1, [0.85]), (2, [1.15]),
                                          (3, [1.19]), (4, [0.86, 0.31])],

                    pKa=[(0, -0.49), (1, 6.04), (2, 9.78), (3, 13.73)]
                )
    mol1.epsilon_from_osc_strength()
    mol1.hydroxyorcein_protonated_species()
    mol1.generate_colour_vs_pH()
    mol1.visualize_species_distribution()


    mol2 = Molecule("aminoorcein",
                    transition_energies=[(0, [2.405, 3.026]),  (1, [2.797]),  (2, [2.493]),
                                         (3, [2.390]), (4, [2.442]), (5, [2.384, 2.593])],
                                                                                                         #  See Scheme S5 for more info
                    protonated_species=[(5, "Cc1cc(O)cc(O)c1c4c([NH3+])cc3oc2cc(=O)cc(C)c2[nH+]c3c4C"),  #  S1111
                                        (4, "Cc1cc(O)cc(O)c1c4c([NH3+])cc3oc2cc(=O)cc(C)c2nc3c4C"),      #  S1101
                                        (3, "Cc1cc(O)cc(O)c1c4c(N)cc3oc2cc(=O)cc(C)c2nc3c4C"),           #  S1100
                                        (2, "Cc1cc(O)cc([O-])c1c4c(N)cc3oc2cc(=O)cc(C)c2nc3c4C"),        #  S0100
                                        (1, "Cc1cc([O-])cc(O)c1c4c(N)cc3oc2cc(=O)cc(C)c2nc3c4C"),        #  S1000
                                        (0, "Cc1cc([O-])cc([O-])c1c4c(N)cc3oc2cc(=O)cc(C)c2nc3c4C")],    #  S0000

                    oscillator_strengths=[(0, [0.24, 0.55]),  (1, [0.52]), (2, [1.07]),
                                          (3, [1.09]), (4, [1.11]), (5, [1.05, 0.03])],

                    pKa=[(0, -0.99), (1, 2.64), (2, 9.23), (3, 10.54)]
                    )
    mol2.epsilon_from_osc_strength()
    mol2.aminoorcein_protonated_species()
    mol2.generate_colour_vs_pH()
    mol2.visualize_species_distribution()

    # add two spectra together and then convert to colour
    mol2.spectra = [0.5 * (sp[0] + sp[1]) for sp in zip(mol1.spectra, mol2.spectra)]
    mol2.colours_vs_pH = [Spectrum(np.linspace(380, 780, 81), sp).rgb_to_hex() for sp in mol2.spectra]

    mol2.visualize_species_distribution()
