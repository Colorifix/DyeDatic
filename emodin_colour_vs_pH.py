import matplotlib.pyplot as plt
from src.ionisation import *

if __name__ == "__main__":
    """
    Equation for systematic error removal
    E_actual (in eV) = 0.397 + 0.731 * E_vert (in eV) for wB97X-D4 

    Emodin  |                   Calculated                           | After systematic error removal
    S111    |                                           3.441 (0.35) | 2.912 (0.35)
    S110    |                             2.916 (0.28); 3.984 (0.12) | 2.528 (0.28); 3.31 (0.12)
    S100    |                             2.760 (0.25); 3.354 (0.08) | 2.414 (0.25); 2.849 (0.08)
    S010    |               2.653 (0.28); 3.419 (0.04); 3.465 (0.03) | 2.336 (0.28); 2.896 (0.04); 2.930 (0.03)
    S000    | 2.622 (0.26); 3.300 (0.03); 3.369 (0.02); 3.615 (0.10) | 2.313 (0.26); 2.809 (0.03); 2.860 (0.02); 3.04 (0.10)

    """
    mol = Molecule("emodin",
                   transition_energies = [(4, [2.912]),  (3, [2.528, 3.31]), (2, [2.414, 2.849]),
                                          (1, [2.336, 2.896, 2.930]), (0, [2.313, 2.809, 2.860, 3.04])],
                                                                                                        # See Scheme S2 for more info
                   protonated_species = [(4, "CC1=CC2=C(C(=C1)O)C(=O)C3=C(C2=O)C=C(C=C3O)O"),           # neutral                    (S111)
                                         (3, "CC1=CC2=C(C(=C1)O)C(=O)C3=C(C2=O)C=C(C=C3O)[O-]"),        # 3-hydroxy deprotonated     (S110)
                                         (2, "CC1=CC2=C(C(=C1)O)C(=O)C3=C(C2=O)C=C(C=C3[O-])[O-]"),     # 1,3-dihydroxy deprotonated (S100)
                                         (1, "CC1=CC2=C(C(=C1)[O-])C(=O)C3=C(C2=O)C=C(C=C3O)[O-]"),     # 3,8-dihydroxy deprotonated (S010)
                                         (0, "CC1=CC2=C(C(=C1)[O-])C(=O)C3=C(C2=O)C=C(C=C3[O-])[O-]")], # trihydroxy deprotonated    (S000)
                   oscillator_strengths = [(4, [0.35]),  (3, [0.28, 0.12]), (2, [0.25, 0.08]),
                                           (1, [0.28, 0.04, 0.03]), (0, [0.26, 0.03, 0.02, 0.10])],
                   pKa = [(0, 7.39), (1, 9.34), (2, 10.13)]
                   )
    mol.epsilon_from_osc_strength()
    mol.emodin_protonated_species()
    mol.generate_colour_vs_pH()
    mol.visualize_species_distribution()
