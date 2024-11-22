from rdkit.Chem import Draw
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from src.convert_spectrum_to_colour import *
from typing import Tuple, List
import numpy as np
from rdkit.Chem.Draw import rdMolDraw2D
from PIL import Image
import io, base64, os


def wavelength2float(wstring: str) -> float:
    """
    A function to convert wavelengths string to float.
    The problem is there are frequently several values reported so if the string
    is not converted to float split it and return its maximal value
    :params wstring: a line potentially containing several values
    :return wavelength: a single float number - absorption peak in nanometres
    """
    try:
        wavelength = float(wstring)
        return wavelength
    except:
        wavelength = np.max([float(i) for i in wstring.split(";")])
        return wavelength

def epsilon2float(estring: str) -> float:
    """
    A function to convert extinction coefficient string to float.
    The problem is there are frequently several values reported so if the string
    is not converted to float split it and return its *last* value (different from wavelengths)
    :params estring: a line potentially containing several values
    :return wavelength: a single float number - extinction coefficient of the maximum absorption peak in M**(-1)*cm**(-1)
    """
    try:
        epsilon = float(estring)
        return epsilon
    except:
        # extinction coefficient information
        # is in the very end of the string
        epsilon = [float(i) for i in estring.split(";")][-1]
        return epsilon

def wavelength_to_rgb(wavelength_eV: float, gamma: float = 0.8) -> Tuple[float, float, float, float]:
    """
    Taken from http://www.noah.org/wiki/Wavelength_to_RGB_in_Python
    This converts a given wavelength of light to an
    approximate RGB color value. The wavelength must be given
    in nanometers in the range from 380 nm through 750 nm

    NB: wavelength is taken in eV then converted to nm
    """
    try:
       wavelength_eV = float(wavelength_eV)
    except:
        raise ValueError(f"{wavelength_eV} cannot be converted to float type")

    # convert to nm as all subsequent calculations are in nm
    wavelength = ev2nm(wavelength_eV)

    if wavelength >= 380 and wavelength <= 750:
        A = 1.
    else:
        A = 0.5
    if wavelength < 380:
        wavelength = 380.
    if wavelength > 750:
        wavelength = 750.
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B, A)


def nm2ev(wv):
    return 1239.8 / wv


def ev2nm(ev):
    return 1239.8 / ev


def DrawTautsFromList(mol_list: List, name: str, print_md = False) -> None:
    """
    A function to draw tautomers from the list and add its relative energy if anything stored
    Also prints Markdown path for the picture
    """

    cnt = 1
    for mol in mol_list:
        if mol.HasProp("energy"):
            Draw.MolToFile(mol, f"../figs/tauts_{name}{str(cnt)}.png", legend=mol.GetProp("energy"))
        else:
            Draw.MolToFile(mol, f"../figs/tauts_{name}{str(cnt)}.png")

        if print_md:
            print(f"![tauts_{name}{str(cnt)}]({os.path.abspath('..')}/figs/tauts_{name}{str(cnt)}.png)")
        cnt += 1


def is_tool(name):
    """Check whether `name` is on PATH and marked as executable."""

    from shutil import which

    return which(name) is not None


def gauss_spec(wavelengths, s = 0.4):
    visible_ev = np.linspace(1.63, 3.26, 163)
    sp = np.zeros(163)
    for wl in wavelengths:
        sp += 1/s * np.sqrt(4 * np.log(2)/np.pi) * np.exp(-4 * np.log(2) * (visible_ev - wl)**2 / s**2)
    return sp


def gauss_spec_with_extinction(wavelengths: List[List[float]],
                               fractions: np.ndarray,
                               extinction_coefficients: List[List[float]],
                               s: float,
                               concentration: float) -> None:
    """
    Given a
    """
    visible_ev = np.linspace(1.63, 3.26, 163)
    sp = np.zeros(163)

    if len(fractions) != len(wavelengths):
        raise RuntimeError('Number of wavelengths is not equal to the number of components in the mixture')

    for i in range(len(wavelengths)):
        wl = wavelengths[i] # wl is a list of wavelengths
        fraction = fractions[i]
        ext_coef = extinction_coefficients[i]
        if len(wl) != len(ext_coef):
            raise RuntimeError('Number of wavelengths is not equal to the number of extinction coefficients for the current component')

        for j in range(len(wl)):
            sp += concentration * ext_coef[j] * fraction * 1/s * np.sqrt(4 * np.log(2)/np.pi) * np.exp(-4 * np.log(2) * (visible_ev - wl[j])**2 / s**2)

    # sp = sp / np.max(sp)
    return sp

def generate_count_fps(smi: str, radius: int = 3, size: int = 2048) -> List[int]:
    """
    A wrapper for Morgan fingerprints generation
    It will through an error if an input SMILES is invalid
    :param smi: input SMILES string
    :param radius: radius for fragment search
    :param size: length of a vector for hashing
    :return: a list of integers (count of how many times a particular fragment can be found in a structure)
    """

    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        raise Exception(f"{smi} is not valid")
    fpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=size, useBondTypes=False)
    fp = fpgen.GetCountFingerprintAsNumPy(mol)
    return fp.tolist()


def generate_colours(wavelength_list):
    colours = []
    visible_ev = np.linspace(1.63, 3.26, 163)
    # if no data is available put 300 nm (colourless)

    for wavelength in wavelength_list:

        sp = gauss_spec([nm2ev(float(nm)) for nm in wavelength.split(";")])
        sp = sp / np.max(sp)

        # spectrum in nm is in a reversed order
        wavelengths, sp = zip(*sorted(zip(ev2nm(visible_ev), sp)))

        # interpolate spectrum to be able to convert with x, y, z functions
        sp = np.interp(np.linspace(380, 780, 81), wavelengths, sp)

        # compute the colour and put it into the container
        col = Spectrum(np.linspace(380, 780, 81), sp).rgb_to_hex()
        colours.append(col)
    return colours

def DrawMol(mol, title, molSize=(250, 250), kekulize=True):
    mc = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DCairo(*molSize)
    drawer.DrawMolecule(mc, legend = title)
    drawer.FinishDrawing()
    image = drawer.GetDrawingText()
    im = Image.open(io.BytesIO(image))
    return np.array(im)

def DrawMol_b64(mol, title, species, molSize=(300, 200), kekulize=True):
    mc = Chem.MolFromSmiles(mol)
    if kekulize:
        try:
            Chem.Kekulize(mc)
        except:
            mc = Chem.Mol(mol.ToBinary())
    if not mc.GetNumConformers():
        Chem.rdDepictor.Compute2DCoords(mc)

    drawer = rdMolDraw2D.MolDraw2DCairo(*molSize)
    # do not draw title and species
    #drawer.DrawMolecule(mc, legend = f"{title}\n{species}")
    drawer.DrawMolecule(mc)
    drawer.FinishDrawing()
    image = drawer.GetDrawingText()
    im = Image.open(io.BytesIO(image))

    # dump it to base64
    buffer = io.BytesIO()
    im.save(buffer, format="jpeg")
    encoded_image = base64.b64encode(buffer.getvalue()).decode()
    im_url = "data:image/jpeg;base64, " + encoded_image

    return im_url

def ndarray_to_b64(ndarray):
    """
    converts a np ndarray to a b64 string readable by html-img tags
    """
    img = cv2.cvtColor(ndarray, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode('.png', img)

