
class QMException(Exception):
    pass

class Calculation:

    """
    This class is going to store information about calculation input files
    The default calculation run is expected to be like this
    """

    multiplicity = 1 # make multiplicity a class variable

    #Define solvent properties as a class attribute
    dielec = {"dioxane": 2.25,
             "etoac": 6.02,
             "et2o": 4.33,
             "tetrachloroethane": 8.42,
             "butanol-1": 7.8}

    refrac = {"dioxane": 1.4224,
              "etoac": 1.3724,
              "et2o": 1.3524,
              "tetrachloroethane": 1.494,
              "butanol-1": 1.3993}

    # set up functional names list used in the paper
    functionals = ["PBE0", "wB97X-D4", "RI-B2PLYP", "CAM-B3LYP", "M06-2X", "BMK", "RI-SCS-PBE-QIDH", "RI-SCS-wPBEPP86"]

    def __init__(self, molname: str, method: str = "PBE0",
                 basis_set: str = "def2-SVP",
                 solvent: str = "Water", memory: int = 1000,
                 charge: int = 0, ncpus: int = 16) -> None:

        """
        Initialize class with the information necessary for any calculation
        By default ORCA 5 uses RIJCOSX approximation and auxilliary basis sets are
        :param molname: string of molecular identifier will be kept as a part of the basename
        :param method: DFT functional here to be used
        :param basis_set: basis set type
        :param memory: memory in MB for each process
        :param ncpus: number of cpu cores to be used in calculations
        :param charge: electric charge
        """

        self.molname = molname
        self.method = method
        self.basis_set = basis_set
        self.memory = memory
        self.charge = charge
        self.ncpus = ncpus
        # check if solvent within a list of predefine solvents
        if solvent not in ['Water', 'Acetonitrile', 'Acetone', 'Ammonia',
                           'Ethanol', 'Methanol', 'CH2Cl2', 'CCl4', 'DMF',
                           'DMSO', 'Pyridine', 'THF', 'Chloroform', 'Hexane', 'Toluene']:
            if solvent not in ["Dioxane", "Ethylacetate", "Ether", "tetrachloroethane", "butanol-1"]:
                raise QMException(f"Solvent {solvent} is not in a predefined list")

        # check if the chosen method belongs to the list of predefined functionals
        if method not in Calculation.functionals:
            raise QMException(f"Functional {method} should be taken from the following list: {', '.join(Calculation.functionals)}")

        self.solvent = solvent # use cpcm model

    def generate_pH_input(self, basis_set: str = 'def2-TZVP',
                          nroots: int = 5, tda: str = False) -> str:
        """
        A function to generate input for protonated/deprotonated species
        PBEh-3c optimisation forllow by wB97X-D4 TD-DFT calculation
        water is a default solvent
        :param basis_set: basis set identifier in ORCA
        :return a string containing all necessary keywords to run ORCA calculations
        """

        # add diffuse functions if the compound has a negative charge
        if int(self.charge) < 0:
            basis_set = "ma-" + basis_set
        input = f"! PBEh-3c def2/J RIJCOSX CPCM(Water) TightSCF TightOpt\n"
        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_gs_opt"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_xtb.xyz\n"
        input += f"\n"
        input += f"$new_job\n"
        input += f"! wB97XD4 {basis_set} def2/J RIJCOSX TIGHTSCF CPCM(Water)\n"
        input += f"%maxcore {str(self.memory * 2)}\n"
        input += f'%base "{str(self.molname)}_wb97xd4_solv_tddft"\n'
        input += f"%pal nprocs {str(self.ncpus // 2)} end\n"  # use less cpu with more memory
        input += f"%TDDFT nroots {nroots} tda {tda} end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_gs_opt.xyz\n"
        return input


    def generate_database_opt_pbe0(self, basis_set: str = 'def2-TZVP',
                                   nroots: int = 5, tda: bool = False) -> str:
        """
        A function to generate input for protonated/deprotonated species minimisation and inital TD-DFT calculations
        PBEh-3c optimisation followed by PBE0 TD-DFT calculation
        :param basis_set: basis set identifier in ORCA
        :param nroots: number of roots for TD-DFT solver
        :param tda: turns Tamm-Dancoff approximation
        :return a string containing all necessary keywords to run ORCA calculations
        """
        # add diffuse functions if the compound has a negative charge
        if int(self.charge) < 0:
            basis_set = "ma-" + basis_set

        # not internally parameterised solvents for CPCM are provided as a separate string
        if self.solvent in self.dielec.keys():
            input  = f"! PBEh-3c def2/J RIJCOSX CPCM NormalSCF NormalOpt\n"
            input += f"%cpcm epsilon {str(self.dielec[self.solvent])} refrac {str(self.refrac[self.solvent])} end\n"
        else:
            input = f"! PBEh-3c def2/J RIJCOSX CPCM({self.solvent}) NormalSCF NormalOpt\n"

        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_gs_opt"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_xtb.xyz\n"
        input += f"\n"
        input += f"$new_job\n"
        if self.solvent in self.dielec.keys():
            input += f"! PBE0 {basis_set} def2/J RIJCOSX TIGHTSCF CPCM\n"
            input += f"%cpcm epsilon {str(self.dielec[self.solvent])} refrac {str(self.refrac[self.solvent])} end\n"
        else:
            input += f"! PBE0 {basis_set} def2/J RIJCOSX TIGHTSCF CPCM({self.solvent})\n"

        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_pbe0_solv_tddft"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"%TDDFT nroots {nroots} tda {tda} end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_gs_opt.xyz\n"
        return input

    def generate_single_tddft(self, method, basis_set = 'def2-TZVP', nroots = 5, tda = False):
        """
        A function to generate input for protonated/deprotonated for functionals explored in the paper
        :param method: a functional from predefined set
        :param basis_set: basis set identifier in ORCA
        :param nroots: number of roots for TD-DFT solver
        :param tda: turns Tamm-Dancoff approximation
        :return a string containing all necessary keywords to run ORCA calculations
        """

        # add diffuse functions if the compound has a negative charge
        if int(self.charge) < 0:
            basis_set = "ma-" + basis_set

        # add auxilliary basis set for double hybrids
        if method in ["RI-B2PLYP", "RI-SCS-PBE-QIDH", "RI-SCS-wPBEPP86"]:
            method += f" {basis_set}/C"

        #certain functionals are taken from libxc
        if method not in ["CAM-B3LYP", "BMK"]:
            # not internally parameterised solvents for CPCM are provided as a separate string
            if self.solvent in self.dielec.keys():
                input = f"! {method} {basis_set} def2/J RIJCOSX CPCM TightSCF\n"
                input += f"%cpcm epsilon {str(self.dielec[self.solvent])} refrac {str(self.refrac[self.solvent])} end\n"
            else:
                input = f"! {method} {basis_set} def2/J RIJCOSX CPCM({self.solvent}) TightSCF\n"
        else:
            if self.solvent in self.dielec.keys():
                input = f"! RKS def2/J  {basis_set} RIJCOSX CPCM TightSCF\n"
                input += f"%cpcm epsilon {str(self.dielec[self.solvent])} refrac {str(self.refrac[self.solvent])} end\n"
                if method == "CAM-B3LYP":
                    input += f"%method method dft functional hyb_gga_xc_camh_b3lyp end\n"
                elif method == "BMK":
                    input += f"%method method dft functional hyb_gga_xc_b97_k end\n"
            else:
                input = f"! {method} {basis_set} def2/J RIJCOSX CPCM({self.solvent}) TightSCF\n"
        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_{method.split()[0].lower()}_solv_tddft"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"%TDDFT nroots {nroots} tda {tda} end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_gs_opt.xyz\n"
        return input

    def generate_single_tddft_nosolv(self, method, basis_set = 'def2-TZVP', nroots = 5, tda = False):
        """
        A function to generate input for protonated/deprotonated for functionals explored in the paper
        :param method: a functional from predefined set
        :param basis_set: basis set identifier in ORCA
        :param nroots: number of roots for TD-DFT solver
        :param tda: turns Tamm-Dancoff approximation
        :return a string containing all necessary keywords to run ORCA calculations
        """

        # add diffuse functions if the compound has a negative charge
        if int(self.charge) < 0:
            basis_set = "ma-" + basis_set

        # add auxilliary basis set for double hybrids
        if method in ["RI-B2PLYP", "RI-SCS-PBE-QIDH", "RI-SCS-wPBEPP86"]:
            method += f" {basis_set}/C"

        #certain functionals are taken from libxc
        if method not in ["CAM-B3LYP", "BMK"]:
            input = f"! {method} {basis_set} def2/J RIJCOSX TightSCF\n"
        else:
            input = f"! RKS {basis_set} def2/J RIJCOSX TightSCF\n"
            if method == "CAM-B3LYP":
                input += f"%method method dft functional hyb_gga_xc_camh_b3lyp end\n"
            elif method == "BMK":
                input += f"%method method dft functional hyb_gga_xc_b97_k end\n"
        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_{method.split()[0].lower()}_nosolv_tddft"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"%TDDFT nroots {nroots} tda {tda} end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_gs_opt.xyz\n"
        return input


    def generate_polarisation_input(self, method, basis_set = 'def2-TZVP'):
        """
        A function to generate input for protonated/deprotonated for functionals
        explored in the paper to calculate polarisation tensor
        :param method: a functional from predefined set
        :param basis_set: basis set identifier in ORCA
        :param nroots: number of roots for TD-DFT solver
        :param tda: turns Tamm-Dancoff approximation
        :return a string containing all necessary keywords to run ORCA calculations
        """

        # add diffuse functions if the compound has a negative charge
        if int(self.charge) < 0:
            basis_set = "ma-" + basis_set

        # add auxilliary basis set for double hybrids
        if method in ["RI-B2PLYP", "RI-SCS-PBE-QIDH", "RI-SCS-wPBEPP86"]:
            method += f" {basis_set}/C"

        #certain functionals are taken from libxc
        if method not in ["CAM-B3LYP", "BMK"]:
            # not internally parameterised solvents for CPCM are provided as a separate string
            if self.solvent in self.dielec.keys():
                input = f"! {method} {basis_set} def2/J RIJCOSX CPCM TightSCF\n"
                input += f"%cpcm epsilon {str(self.dielec[self.solvent])} refrac {str(self.refrac[self.solvent])} end\n"
            else:
                input = f"! {method} {basis_set} def2/J RIJCOSX CPCM({self.solvent}) TightSCF\n"
        else:
            if self.solvent in self.dielec.keys():
                input = f"! RKS def2/J  {basis_set} RIJCOSX CPCM TightSCF\n"
                input += f"%cpcm epsilon {str(self.dielec[self.solvent])} refrac {str(self.refrac[self.solvent])} end\n"
                if method == "CAM-B3LYP":
                    input += f"%method method dft functional hyb_gga_xc_camh_b3lyp end\n"
                elif method == "BMK":
                    input += f"%method method dft functional hyb_gga_xc_b97_k end\n"
            else:
                input = f"! {method} {basis_set} def2/J RIJCOSX CPCM({self.solvent}) TightSCF\n"
        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_{method.split()[0].lower()}_solv_polar"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"%elprop Polar 1 end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_gs_opt.xyz\n"
        return input


    def generate_polarisation_input_nosolv(self, method, basis_set = 'def2-TZVP'):
        """
        A function to generate input for protonated/deprotonated for functionals
        explored in the paper to calculate polarisation tensor
        :param method: a functional from predefined set
        :param basis_set: basis set identifier in ORCA
        :param nroots: number of roots for TD-DFT solver
        :param tda: turns Tamm-Dancoff approximation
        :return a string containing all necessary keywords to run ORCA calculations
        """

        # add diffuse functions if the compound has a negative charge
        if int(self.charge) < 0:
            basis_set = "ma-" + basis_set

        # add auxilliary basis set for double hybrids
        if method in ["RI-B2PLYP", "RI-SCS-PBE-QIDH", "RI-SCS-wPBEPP86"]:
            method += f" {basis_set}/C"

        #certain functionals are taken from libxc
        if method not in ["CAM-B3LYP", "BMK"]:
            input = f"! {method} {basis_set} def2/J RIJCOSX TightSCF\n"
        else:
            input = f"! RKS {basis_set} def2/J RIJCOSX TightSCF\n"
            if method == "CAM-B3LYP":
                input += f"%method method dft functional hyb_gga_xc_camh_b3lyp end\n"
            elif method == "BMK":
                input += f"%method method dft functional hyb_gga_xc_b97_k end\n"
        input += f"%maxcore {str(self.memory)}\n"
        input += f'%base "{str(self.molname)}_{method.split()[0].lower()}_nosolv_polar"\n'
        input += f"%pal nprocs {str(self.ncpus)} end\n"
        input += f"%elprop Polar 1 end\n"
        input += f"*xyzfile {self.charge} {Calculation.multiplicity} {self.molname}_gs_opt.xyz\n"
        return input