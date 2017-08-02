"""
QuESt command line interface.
"""
import os
import argparse
import yaml
import quest
import quest.driver as driver
from quest.molecule import Molecule
from quest.mollib import mollib


parser = argparse.ArgumentParser(
    description="""
-------------------------------------------

  ██████╗         ███████╗███████╗
 ██╔═══██╗        ██╔════╝██╔════╝  ██║
 ██║   ██║██║  ██║█████╗  ███████╗██████╗
 ██║▄▄ ██║██║  ██║██╔══╝  ╚════██║╚═██╔═╝
 ╚██████╔╝╚█████╔╝███████╗███████║  ███║
  ╚══▀▀═╝  ╚════╝ ╚══════╝╚══════╝  ╚══╝

QuESt: Quantum Energy and Stuff
-------------------------------------------
    """,
    epilog="""
    """,
    formatter_class=argparse.RawDescriptionHelpFormatter)

default_params = quest.default_params

parser.add_argument('--molecule', '-m', type=str, default='h2o', metavar='',
                    help='Molecule file name (default: h2o)')
parser.add_argument('--parameters', '-p', type=str, default=default_params, metavar='',
                    help='Parameters file (default: parameters.yml)')
parser.add_argument('-qm', action='store_true', default=True,
                    help='Quantum mechanics (default: True)')
parser.add_argument('-mm', action='store_true', default=True,
                    help='Molecular mechanics (default: True)')

# Parse arguments
args = parser.parse_args()

# Read parameters
with open(args.parameters, 'r') as inp_file:
    params = yaml.load(inp_file)

if args.qm:
    qmp = params['qm']
    scf_energy, wfn = driver.compute_rhf(mollib[args.molecule],
                                         basis_name=qmp['basis_name'],
                                         numpy_memory=qmp['numpy_memory'],
                                         maxiter=qmp['maxiter'],
                                         E_conv=qmp['E_conv'],
                                         D_conv=qmp['D_conv'])

    mp2_energy = driver.compute_mp2(wfn)
    print('Total MP2 energy: %.5f' % mp2_energy)
