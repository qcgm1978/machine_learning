import matplotlib
from matplotlib.backends.backend_pgf import FigureCanvasPgf
matplotlib.backend_bases.register_backend('pdf', FigureCanvasPgf)

import matplotlib.pyplot as plt

pgf_with_latex = {
    "text.usetex": True,            # use LaTeX to write all text
    "pgf.rcfonts": False,           # Ignore Matplotlibrc
    "pgf.preamble": [
        r'\usepackage{color}'     # xcolor for colours
    ]
}
matplotlib.rcParams.update(pgf_with_latex)

plt.figure()
plt.ylabel(r'\textcolor{red}{Today} '+
           r'\textcolor{green}{is} '+
           r'\textcolor{blue}{cloudy.}')
plt.savefig("/Users/zhanghongliang/Documents/machine_learning/img/demo.pdf")
