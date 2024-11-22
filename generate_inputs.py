from rdkit import Chem
import os
import pandas as pd
from src.optimize_xtb import confgen
from src.tautomers import enumerate_tautomers
from src.orca_inputs import Calculation
from loguru import logger

PIGMENTS = "data/pigments.csv" # a csv file with pigment SMILES and other related information
INPUT_DIR = "inputs"           # a directory to generate ORCA input files in sub dirs with colourants names

# conformation generation algorithm has problems with certain large molecules:
# salinixanthin, purpurogenone where structure generation were done with obabel

if __name__ == "__main__":
    # read pigment database
    pigments = pd.read_csv(PIGMENTS)
    pigments.reset_index(inplace = True)
    pigment_smiles = pigments["smiles"]

    # convert SMILES to rdkit Mol objects with the same order as in padnas DataFrame
    mols = [Chem.MolFromSmiles(smiles) for smiles in pigment_smiles]

    # Make sure all molecules are converted otherwise RuntimeError with molnames
    if any([mol is None for mol in mols]):
        none_idx = [mol is None for mol in mols]
        raise RuntimeError("The following molecules could not be read by rdkit: " +
                           ", ".join(pigments["name"].loc[none_idx]))

    logger.info(f"{len(mols)} molecules read and converted to SMILES")
    # then create a set of directories with names from molecular titles
    cnt = 0
    for mol in mols:
        name = pigments["name"].loc[cnt]
        logger.info(f"Current substance: {name} {cnt + 1} / {len(mols)}")

        # if not exists create a directory for a colourant input files
        path_mol = os.path.join(INPUT_DIR, name)
        if not os.path.exists(path_mol):
            os.mkdir(path_mol)
        os.chdir(path_mol)

        # get total molecular charge from SMILES
        total_charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])

        # not all solvents are available use water if the solvent is not parameterised
        if pigments["solvent"].loc[cnt] not in ["Acetone", "Acetonitrile", "Aniline", "Benzaldehyde", "Benzene", "CH2Cl2", "CHCl3",
                           "CS2", "Dioxane", "DMF", "DMSO", "Ether", "Ethylacetate", "Furane", "Hexadecane",
                           "Hexane", "Methanol", "Nitromethane", "Octanol", "Phenol", "Toluene", "THF", "Water"]:
            solvent = "h2o"
        else:
            solvent = pigments["solvent"].loc[cnt]

        # prepare a list of tautomers and get the first one as it has the lowest energy
        tautomers = enumerate_tautomers(mol, solvent = solvent) # not all solvents are available
        mol = tautomers.pop(0)

        # generate conformers and optimise geometry
        energy, opt_mol = confgen(name, mol)

        # prepare information to generate ORCA input files
        calculation = Calculation(name, charge = total_charge, solvent = pigments["solvent"].loc[cnt])

        # optimisation and pbe0 calculations
        opt_pbe0 = calculation.generate_database_opt_pbe0()
        with open("opt_pbe0.inp", "w") as f:
            f.write(opt_pbe0)

        # pbe0 calculations in vacuum
        pbe0_nosolv = calculation.generate_single_tddft_nosolv("PBE0")
        with open("pbe0_nosolv.inp", "w") as f:
            f.write(pbe0_nosolv)

        # wb97xd4 calculations
        wb97xd4 = calculation.generate_single_tddft("wB97X-D4")
        with open("wb97xd4.inp", "w") as f:
            f.write(wb97xd4)

        # wb97xd4 calculations in vacuum
        wb97xd4_nosolv = calculation.generate_single_tddft_nosolv("wB97X-D4")
        with open("wb97xd4_nosolv.inp", "w") as f:
            f.write(wb97xd4_nosolv)

        # wb97xd4 calculations
        wb97xd4 = calculation.generate_single_tddft("wB97X-D4")
        with open("wb97xd4.inp", "w") as f:
            f.write(wb97xd4)

        # wb97xd4 calculations in vacuum
        wb97xd4_nosolv = calculation.generate_single_tddft_nosolv("wB97X-D4")
        with open("wb97xd4_nosolv.inp", "w") as f:
            f.write(wb97xd4_nosolv)

        # b2plyp calculations
        b2plyp = calculation.generate_single_tddft("RI-B2PLYP")
        with open("b2plyp.inp", "w") as f:
            f.write(b2plyp)

        # b2plyp calculations in vacuum
        b2plyp_nosolv = calculation.generate_single_tddft_nosolv("RI-B2PLYP")
        with open("b2plyp_nosolv.inp", "w") as f:
            f.write(b2plyp_nosolv)

        # camb3lyp calculations
        camb3lyp = calculation.generate_single_tddft("CAM-B3LYP")
        with open("camb3lyp.inp", "w") as f:
            f.write(camb3lyp)

        # camb3lyp calculations in vacuum
        camb3lyp_nosolv = calculation.generate_single_tddft_nosolv("CAM-B3LYP")
        with open("camb3lyp_nosolv.inp", "w") as f:
            f.write(camb3lyp_nosolv)

        # m062x calculations
        m062x = calculation.generate_single_tddft("M06-2X")
        with open("m062x.inp", "w") as f:
            f.write(m062x)

        # m062x calculations in vacuum
        m062x_nosolv = calculation.generate_single_tddft_nosolv("M06-2X")
        with open("m062x_nosolv.inp", "w") as f:
            f.write(m062x_nosolv)

        # bmk calculations
        bmk = calculation.generate_single_tddft("BMK")
        with open("bmk.inp", "w") as f:
            f.write(bmk)

        # bmk calculations in vacuum
        bmk_nosolv = calculation.generate_single_tddft_nosolv("BMK")
        with open("bmk_nosolv.inp", "w") as f:
            f.write(bmk_nosolv)

        # pbeqidh calculations
        pbeqidh = calculation.generate_single_tddft("RI-SCS-PBE-QIDH")
        with open("pbeqidh.inp", "w") as f:
            f.write(pbeqidh)

        # pbeqidh calculations in vacuum
        pbeqidh_nosolv = calculation.generate_single_tddft_nosolv("RI-SCS-PBE-QIDH")
        with open("pbeqidh_nosolv.inp", "w") as f:
            f.write(pbeqidh_nosolv)

        # wpbepp86 calculations
        wpbepp86 = calculation.generate_single_tddft("RI-SCS-wPBEPP86")
        with open("wpbepp86.inp", "w") as f:
            f.write(wpbepp86)

        # wpbepp86 calculations in vacuum
        wpbepp86_nosolv = calculation.generate_single_tddft_nosolv("RI-SCS-wPBEPP86")
        with open("wpbepp86_nosolv.inp", "w") as f:
            f.write(wpbepp86_nosolv)

        # wb97xd4 calculations for polarisation tensor
        wb97xd4_polar = calculation.generate_polarisation_input("wB97X-D4")
        with open("wb97xd4_polar.inp", "w") as f:
            f.write(wb97xd4_polar)

        # wb97xd4 calculations for polarisation tensors in vacuum
        wb97xd4_polar_nosolv = calculation.generate_polarisation_input_nosolv("wB97X-D4")
        with open("wb97xd4_polar_nosolv.inp", "w") as f:
            f.write(wb97xd4_polar_nosolv)

        # wb97xd4 calculations with TDA approximation
        wb97xd4_tda = calculation.generate_single_tddft("wB97X-D4", tda = True)
        with open("wb97xd4_tda.inp", "w") as f:
            f.write(wb97xd4_tda)

        # wb97xd4 calculations with TDA approximation in vacuum
        wb97xd4_tda_nosolv = calculation.generate_single_tddft_nosolv("wB97X-D4", tda = True)
        with open("wb97xd4_tda_nosolv.inp", "w") as f:
            f.write(wb97xd4_tda_nosolv)

        # once input and geometry file are ready go to the parent directory
        os.chdir("../..")
        cnt += 1

