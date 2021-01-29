import matplotlib.pyplot as plt
import seaborn as sns


def init_plotpalette(totpalette, l=.4, s=.8):
    plotpalette = sns.hls_palette(totpalette, l=l, s=s).as_hex()
    sns.set_palette(plotpalette)
    return plotpalette

def default_rcparams(style='ticks'):
    sns.set_style(style)
    plt.rcParams['figure.figsize'] = [15,8]
    plt.rcParams['xtick.labelsize'] = 24
    plt.rcParams['ytick.labelsize'] = 24
    plt.rcParams['axes.linewidth'] = 1.6
    plt.rcParams['lines.linewidth'] = 3
    plt.rcParams['font.size'] = 20
    plt.rcParams['axes.labelsize'] = 25
    plt.rcParams['axes.titlesize'] = 30


plotpalette = init_plotpalette(8)
default_rcparams()


#plotpalette = sns.hls_palette(8, l=.4, s=.8).as_hex()
#sns.set_palette(plotpalette)
#plt.rcParams['figure.figsize'] = [15,8]
#plt.rcParams['xtick.labelsize'] = 24
#plt.rcParams['ytick.labelsize'] = 24
#plt.rcParams['axes.linewidth'] = 1.6
#plt.rcParams['lines.linewidth'] = 3
#plt.rcParams['font.size'] = 20
#plt.rcParams['axes.labelsize'] = 25
#plt.rcParams['axes.titlesize'] = 30
