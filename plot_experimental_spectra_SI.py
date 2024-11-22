import matplotlib.gridspec as gridspec
from src.convert_spectrum_to_colour import *


# Visualise absorption spectra at different pH values for a set of 4 natural colourants
colourants = ["Emodin", "Quinalizarin", "Orcein", "Biliverdin"]
colourants_titles = ["Emodin", "Quinalizarin", "Alpha-hydroxyorcein", "Biliverdin"]


styles = [{"color": "yellow", "linestyle": "dashed"},
          {"color": "orange", "linestyle": "dashed"},
          {"color": "red", "linestyle": "dashed"},
          {"color": "green", "linestyle": "dashed"},
          {"color": "magenta", "linestyle": "dashed"},
          {"color": "blue", "linestyle": "dashed"},
          {"color": "yellow", "linestyle": "solid"},
          {"color": "orange", "linestyle": "solid"},
          {"color": "red", "linestyle": "solid"},
          {"color": "green", "linestyle": "solid"},
          {"color": "magenta", "linestyle": "solid"},
          {"color": "blue", "linestyle": "solid"}]


data = pd.read_csv("data/pigment_ph_SI.csv")
num = data._get_numeric_data()
num[num < 0] = 0

lambdas = np.linspace(380, 800, 211)

fig = plt.figure(figsize= (10, 8))

# use gridspec to split each of 4 plot into two smaller subplots
# for band shape and the corresponding colour
outer = gridspec.GridSpec(2, 2, hspace=0.1)

for j in range(4):

    mol = colourants_titles[j]
    inner = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=outer[j], wspace=0.1, hspace=0.1)
    spectra = plt.Subplot(fig, inner[0])
    
    for i in range(2, 13):
        spectra.plot(lambdas,
                     data[(data["compound"] == mol) & (data["pH"] == i)].values[0, 2:],
                     **styles[i-2],
                     label=f"pH = {str(i)}")

    spectra.set_title(colourants[j])
    colours = plt.Subplot(fig, inner[1])

    for i in range(2, 13):
        
        hex_rgb = Spectrum(lambdas, data[(data["compound"] == mol) & (data["pH"] == i)].values[0, 2:]).rgb_to_hex()
        circle = Circle(xy = (0.5 + (i - 2), 0.5), radius = 0.4, fc = hex_rgb)
        colours.add_patch(circle)
        colours.annotate(str(i), xy=(0.5 + (i -2), -0.2), va='center', ha='center', color='white')

    colours.set_xlim(0, 11)
    colours.set_ylim(-0.5, 1)
    colours.set_xticks([])
    colours.set_yticks([])
    colours.set_facecolor('k')
    # Make sure our circles are circular!
    colours.set_aspect("equal")

    fig.add_subplot(spectra)
    fig.add_subplot(colours)
    

lines, labels = spectra.get_legend_handles_labels()
plt.figlegend(lines, labels, loc='upper center', ncol = 6)
fig.supxlabel("$\\lambda$, nm")
fig.supylabel("Absorption")
plt.show()



