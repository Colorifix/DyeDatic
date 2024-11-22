import subprocess, glob, os
from loguru import logger
import re
from typing import Tuple, Union, Type
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from rdkit.Chem import AllChem, rdDistGeom

logger.add("../logs/optimization.log", format="{time} {level} {message}",
           filter="my_module", level="INFO", rotation="10 MB")

XTB_EXEC = "xtb"

class XTBException(Exception):
    pass

def run_xtb_optimization(input: Union[str, Mol],
                         solvent: str ="h2o",
                         charge: int = 0,
                         opt_criteria: str = "tight") -> Tuple[Mol, float, float]:
    """
    A wrapper to run xtb semiempirical quantum chemistry code to optimize conformation and get final energy
    :param input_file: xyz file containing geometry input for
    :param solvent: water by default
    :param charge: 0 by default
    :param opt_criteria: 'tight' by default, see other options in documentation
    :return: tuple of rdkit Mol with optimized geometry, the final energy of the optimized structure, and HOMO-LUMO gap
    """

    global XTB_EXEC

    # check if input is a string or a Mol object
    if type(input) == Mol:
        input_filename = input.GetProp("_Name") + "_mm.xyz"
        Chem.rdmolfiles.MolToXYZFile(input, filename=input_filename)
        # get total molecular charge for quantum calculations: overwrites the default value
        charge = sum([atom.GetFormalCharge() for atom in input_mol.GetAtoms()])

        result = subprocess.run([XTB_EXEC, input_filename, "--opt", opt_criteria,
                                 "--charge", str(charge), "--alpb", solvent],
                                 capture_output=True, text=True)

        logger.info(f"{input.GetProp('_Name')}: {str(result.stderr)}")
    # in case of a filename in string format
    else:
        result = subprocess.run([XTB_EXEC, input, "--opt", opt_criteria,
                                 "--charge", str(charge), "--alpb", solvent],
                                capture_output=True, text=True)

        logger.info(f"{input.split('_')[0]}: {str(result.stderr)}")

    # throw an exception if something nasty happens
    if result.returncode == 1:
        if type(input) == Mol:
            raise XTBException(f"{input.GetProp('_Name')}: {str(result.stderr)}")
        else:
            raise XTBException(f"{input.split('_')[0]}: {str(result.stderr)}")

    # use mol file instead of xyz to retain connectivity information
    mol = Chem.rdmolfiles.MolFromMolFile('xtbtopo.mol', sanitize=False, removeHs=False)

    # cleanup data in the directory
    for f in glob.glob("xtb*"):
        os.remove(f)
    os.remove("charges")
    os.remove("wbo")

    # get energy and HOMO-LUMO gap from the standard output
    line_with_energy = [line for line in result.stdout.split("\n") if re.findall(r"TOTAL ENERGY", line)].pop()
    line_with_hlgap = [line for line in result.stdout.split("\n") if re.findall(r"HL-Gap", line)].pop()
    hlgap = float(line_with_hlgap.split()[3])
    energy = float(line_with_energy.split()[3])
    return mol, energy, hlgap

def run_xtb_energy(input: Union[str, Mol],
                   solvent: str = "h2o",
                   charge: int = 0) -> Tuple[float, float]:
    """
    A wrapper to run xtb semiempirical quantum chemistry code to get energy of the given conformation
    :param input: either xyz file containing geometry input or Mol object
    :param solvent: water by default
    :param charge: 0 by default
    :return: final energy of the optimized structure and HOMO-LUMO gap
    """

    global XTB_EXEC

    # check if input is a string or a Mol object
    if type(input) == Mol:
        input_filename = input.GetProp("_Name") + "_mm.xyz"
        Chem.rdmolfiles.MolToXYZFile(input, filename=input_filename)
        # get total molecular charge for quantum calculations: overwrites the default value
        charge = sum([atom.GetFormalCharge() for atom in input_mol.GetAtoms()])

        result = subprocess.run([XTB_EXEC, input_filename,
                                 "--charge", str(charge), "--alpb", solvent],
                                capture_output=True, text=True)

        logger.info(f"{input.GetProp('_Name')}: {str(result.stderr)}")
    # in case of a filename in string format
    else:
        result = subprocess.run([XTB_EXEC, input,
                                 "--charge", str(charge), "--alpb", solvent],
                                capture_output=True, text=True)

        logger.info(f"{input.split('_')[0]}: {str(result.stderr)}")

    # throw an exception if something nasty happens
    if result.returncode == 1:
        if type(input) == Mol:
            raise XTBException(f"{input.GetProp('_Name')}: {str(result.stderr)}")
        else:
            raise XTBException(f"{input.split('_')[0]}: {str(result.stderr)}")

    # use mol file instead of xyz to retain connectivity information
    mol = Chem.rdmolfiles.MolFromMolFile('xtbtopo.mol', sanitize=False, removeHs=False)

    # cleanup data in the directory
    for f in glob.glob("xtb*"):
        os.remove(f)
    os.remove("charges")
    os.remove("wbo")

    # get energy and HOMO-LUMO gap from the standard output
    line_with_energy = [line for line in result.stdout.split("\n") if re.findall(r"TOTAL ENERGY", line)].pop()
    line_with_hlgap = [line for line in result.stdout.split("\n") if re.findall(r"HL-Gap", line)].pop()
    hlgap = float(line_with_hlgap.split()[3])
    energy = float(line_with_energy.split()[3])
    return energy, hlgap

def get_lowest_energies_xtb(mol: Mol,
                            solvent: str = "h2o",
                            num_conf: int = 10) -> float:
    """
    A high level function for of tautomer energies estimation. it should be noted the passed Mol instance
    will be modified to accomodate multiple conformers
    :param mol: instance of Mol class
    :param solvent: water by default
    :param opt_criteria: 'tight' by default, see other options in documentation
    :return: final energy of the optimized structure
    """

    # check if a molecule contains explicit hydrogens
    if not any([atom.GetSymbol() == "H" for atom in mol.GetAtoms()]):
        mol = Chem.AddHs(mol)

    charge = sum([atom.GetFormalCharge() for atom in mol.GetAtoms()])

    # embed molecular conformers and optimise their geometries
    param = rdDistGeom.ETKDGv3()
    cids = rdDistGeom.EmbedMultipleConfs(mol, num_conf, useRandomCoords = True)
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

    confs, conf_energies = [], []
    for conf_id in range(mol.GetNumConformers()):
        # create temporary xyz files for conformation energy estimation
        conf_copy = Chem.Mol(mol, confId=conf_id)
        Chem.rdmolfiles.MolToXYZFile(conf_copy, filename="temp.xyz")
        energy, hlgap = run_xtb_energy("temp.xyz", solvent=solvent, charge=charge)
        os.remove('temp.xyz')
        conf_energies.append(energy)

    return min(conf_energies)

def confgen(name: str,
            mol: Type[Mol]) -> Tuple[int, Type[Mol]]:
    """
    A specific function to generate pigments conformers using MMFF94s then optimize each using xtb
    with default parameters and get the lowest energy conformation
    :param name: molecular identifier taken from pigments file
    :param mol: instance of Mol class

    :return: final energy of the optimized structure with lowest energy
    :return: Mol object with 3D coordinates
    """

    # AddHs if not hydrogens in the molecule
    if not any([a.GetSymbol() == "H" for a in mol.GetAtoms()]):
        mol = Chem.AddHs(mol)

    # generate conformations for the lowest energy tautomer and optimise them in MMFF94s
    param = rdDistGeom.ETKDGv3()
    cids = rdDistGeom.EmbedMultipleConfs(mol, 30, param)
    AllChem.MMFFOptimizeMoleculeConfs(mol, numThreads=0, mmffVariant='MMFF94s')

    confs, conf_energies = [], []

    for conf_id in range(mol.GetNumConformers()):
        conf_copy = Chem.Mol(mol, confId=conf_id)
        Chem.rdmolfiles.MolToXYZFile(conf_copy, filename=os.path.join(os.getcwd(), name + "_mm.xyz"))
        conf, energy, _ = run_xtb_optimization(name + "_mm.xyz")
        confs.append(conf)
        conf_energies.append(energy)

    min_energy = min(conf_energies)
    lowest_energy_idx = conf_energies.index(min_energy)
    best_conf_xtb = confs[lowest_energy_idx]
    best_conf_mmff94 = Chem.Mol(mol, confId=lowest_energy_idx)

    # write both MMFF94 optimised and xtb optimized conformations with lowest energy
    Chem.rdmolfiles.MolToXYZFile(best_conf_mmff94, filename=os.path.join(os.getcwd(), name + "_mm.xyz"))
    Chem.rdmolfiles.MolToXYZFile(best_conf_xtb, filename=os.path.join(os.getcwd(), name + "_xtb.xyz"))

    return min_energy, best_conf_xtb


