"""Shared figure style for the paper: the paper's faces (Source Serif 4, IBM Plex Sans, IBM Plex Mono),
the Wong colour-blind-safe palette plus the site's tide accent, Tufte data-ink, vector PDF output.

Usage:
    from style import apply_style, C, save
    apply_style()
    fig, ax = plt.subplots(figsize=(W1, 2.4))   # W1 = text width in inches
    ...
    save(fig, 'name')                            # writes paper/figures/name.pdf
"""
import os
import matplotlib as mpl
from matplotlib import font_manager as fm

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.dirname(HERE)  # paper/figures
TEXMF = '/usr/local/texlive/2026/texmf-dist/fonts/opentype'
FONT_FILES = [
    'adobe/sourceserif/SourceSerif4-Regular.otf', 'adobe/sourceserif/SourceSerif4-RegularIt.otf',
    'adobe/sourceserif/SourceSerif4-Semibold.otf', 'adobe/sourceserif/SourceSerif4-Bold.otf',
    'ibm/plex/IBMPlexSans-Regular.otf', 'ibm/plex/IBMPlexSans-Medium.otf', 'ibm/plex/IBMPlexSans-SemiBold.otf',
    'ibm/plex/IBMPlexSans-Italic.otf', 'ibm/plex/IBMPlexMono-Regular.otf', 'ibm/plex/IBMPlexMono-Medium.otf',
]

# text width of the paper (15.4 cm) and a half width, in inches
W1 = 15.4 / 2.54
W2 = W1 / 2 - 0.1

# palette: Wong (2011) plus the paper's accents; one colour per role, used consistently
C = {
    'tide': '#0E7385',      # accent: the method being discussed, "holds"
    'ochre': '#A8720C',     # highlight, conditions
    'seal': '#A3392B',      # obstruction, "fails"
    'moss': '#3C7A3A',      # mergeable, invariant kept
    'ink': '#13233A', 'ink2': '#46566A', 'ink3': '#6F7D8C', 'rule': '#C7D0CC', 'paper2': '#EEF1EE',
    'blue': '#0072B2', 'orange': '#E69F00', 'green': '#009E73', 'vermillion': '#D55E00',
    'sky': '#56B4E9', 'purple': '#CC79A7', 'yellow': '#F0E442', 'gray': '#7F7F7F',
}
# modification kinds (same order as the site's legend)
KIND = {
    'additive-weight': C['blue'], 'multiplicative-weight': C['purple'], 'architectural-insertion': C['orange'],
    'input-or-activation-augmentation': C['vermillion'], 'selective': C['green'],
    'optimizer-or-training-procedure': C['sky'], 'composition-or-routing': C['yellow'], 'hybrid': C['gray'],
}


def apply_style():
    # TrueType conversions (fonts/*.ttf, made by otf2ttf.py) embed as Type 42; fall back to the OTF files
    ttf_dir = os.path.join(HERE, 'fonts')
    for f in FONT_FILES:
        ttf = os.path.join(ttf_dir, os.path.basename(f).replace('.otf', '.ttf'))
        p = ttf if os.path.exists(ttf) else os.path.join(TEXMF, f)
        if os.path.exists(p):
            fm.fontManager.addfont(p)
    mpl.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['IBM Plex Sans', 'DejaVu Sans'],
        'font.serif': ['Source Serif 4', 'DejaVu Serif'],
        'font.monospace': ['IBM Plex Mono', 'DejaVu Sans Mono'],
        'font.size': 8.5, 'axes.titlesize': 9.5, 'axes.labelsize': 8.5, 'xtick.labelsize': 7.5, 'ytick.labelsize': 7.5,
        'legend.fontsize': 7.5, 'legend.frameon': False,
        'mathtext.fontset': 'stix',             # Times-like maths, close to the paper's newtx
        'axes.edgecolor': C['ink3'], 'axes.labelcolor': C['ink'], 'axes.linewidth': 0.6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.titleweight': 'medium', 'axes.titlelocation': 'left',
        'xtick.color': C['ink2'], 'ytick.color': C['ink2'], 'xtick.major.width': 0.5, 'ytick.major.width': 0.5,
        'xtick.major.size': 2.5, 'ytick.major.size': 2.5,
        'grid.color': C['rule'], 'grid.linewidth': 0.4, 'grid.alpha': 0.8,
        'lines.linewidth': 1.3, 'lines.markersize': 4,
        'figure.dpi': 150, 'savefig.dpi': 600, 'savefig.bbox': 'tight', 'savefig.pad_inches': 0.02,
        'pdf.fonttype': 42, 'ps.fonttype': 42,
    })


def save(fig, name):
    path = os.path.join(OUT, name + '.pdf')
    fig.savefig(path)
    return path
