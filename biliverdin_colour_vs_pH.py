import matplotlib.pyplot as plt
from src.ionisation import *

if __name__ == "__main__":

    """
    Equation for systematic error removal
    E_actual (in eV) = 0.397 + 0.731 * E_vert (in eV) for wB97X-D4  

    Biliverdin     |    Calculated energies                                 | Energies after systematic error removal
    S11111         |                             2.234 (0.96); 3.787 (1.67) |                             2.030 (0.96); 3.165 (1.67)
    S11011         |               2.087 (1.74); 3.593 (0.05); 3.974 (0.10) |               1.922 (1.74); 3.023 (0.05); 3.302 (0.10)
    S10111         |               2.188 (1.26); 3.717 (1.20); 3.942 (0.17) |               1.996 (1.26); 3.114 (1.20); 3.279 (0.17)
    S10011         |               2.263 (1.69); 3.681 (0.04); 4.034 (0.27) |               2.051 (1.69); 3.088 (0.04); 3.346 (0.27)
    S10001         |               2.570 (1.89); 3.697 (0.22); 3.809 (0.29) |               2.276 (1.89); 3.100 (0.22); 3.181 (0.29)
    S00001         | 2.159 (1.98); 3.318 (0.04); 3.783 (0.24); 3.886 (0.15) | 1.975 (1.98); 2.822 (0.04); 3.162 (0.24); 3.238 (0.15)
    S10000         |               2.297 (0.58); 3.332 (0.78); 3.635 (0.59) |               2.076 (0.58); 2.832 (0.78); 3.054 (0.59) 
    S00000         |               2.266 (1.37); 3.215 (0.51); 3.598 (0.14) |               2.053 (1.37); 2.747 (0.51); 3.027 (0.14) 
    
    """
    mol = Molecule("biliverdin",
                   transition_energies=[(0, [2.053, 2.747, 3.027]),  (1, [2.076, 2.832, 3.054]),  (2, [1.975, 2.822, 3.162, 3.238]),
                                        (3, [2.276, 3.100, 3.181]), (4, [2.051, 3.088, 3.346]), (5, [1.996, 3.114, 3.279]),
                                        (6, [1.922, 3.023, 3.302]), (7, [2.030, 3.165])],

                   protonated_species=[(7, "C=Cc4c(C)c(=Cc3[nH+]c(=Cc2[nH]c(C=c1[nH]c(=O)c(C)c1C=C)c(C)c2CCC(=O)O)c(CCC(=O)O)c3C)[nH]c4=O"),       # S11111
                                       (6, "C=Cc4c(C)c(=Cc3[nH+]c(=Cc2[nH]c(C=c1[nH]c(=O)c(C)c1C=C)c(C)c2CCC(=O)[O-])c(CCC(=O)O)c3C)[nH]c4=O"),    # S11011
                                       (5, "C=Cc4c(C)c(=Cc3[nH+]c(=Cc2[nH]c(C=c1[nH]c(=O)c(C)c1C=C)c(C)c2CCC(=O)O)c(CCC(=O)[O-])c3C)[nH]c4=O"),    # S10111
                                       (4, "C=Cc4c(C)c(=Cc3[nH+]c(=Cc2[nH]c(C=c1[nH]c(=O)c(C)c1C=C)c(C)c2CCC(=O)[O-])c(CCC(=O)[O-])c3C)[nH]c4=O"), # S10011 See Scheme S6 for more information about species
                                       (3, "C=Cc4c(C)c(=Cc3nc(=Cc2[nH]c(C=c1[nH]c(=O)c(C)c1C=C)c(C)c2CCC(=O)[O-])c(CCC(=O)[O-])c3C)[nH]c4=O"),     # S10001
                                       (2, "C=Cc4c(C)c(=Cc3nc(=Cc2[nH]c(C=c1[n-]c(=O)c(C)c1C=C)c(C)c2CCC(=O)[O-])c(CCC(=O)[O-])c3C)[nH]c4=O"),     # S00001
                                       (1, "C=Cc4c(C)c(=Cc3nc(=Cc2[nH]c(C=c1[nH]c(=O)c(C)c1C=C)c(C)c2CCC(=O)[O-])c(CCC(=O)[O-])c3C)[n-]c4=O"),     # S10000
                                       (0, "C=Cc4c(C)c(=Cc3nc(=Cc2[nH]c(C=c1[n-]c(=O)c(C)c1C=C)c(C)c2CCC(=O)[O-])c(CCC(=O)[O-])c3C)[n-]c4=O")],    # S00000

                   oscillator_strengths=[(0, [1.37, 0.51, 0.14]),  (1, [0.58, 0.78, 0.59]),  (2, [1.98, 0.04, 0.24, 0.15]),
                                         (3, [1.89, 0.22, 0.29]), (4, [1.69, 0.04, 0.27]), (5, [1.26, 1.20, 0.17]),
                                         (6, [1.74, 0.05, 0.10]), (7, [0.96, 1.67])],

                   pKa=[(0, 3.87), (1, 4.46), (2, 5.86), (3, 10.44), (4, 11.04)]
                   )

    mol.epsilon_from_osc_strength()
    mol.biliverdin_protonated_species()
    mol.generate_colour_vs_pH()
    mol.visualize_species_distribution()


