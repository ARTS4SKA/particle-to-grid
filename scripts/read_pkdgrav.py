import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from tqdm import tqdm

import astropy.constants as cst
from astropy.cosmology import Planck18 as cosmo

ρ_c = cosmo.critical_density0.to('Msun/Mpc**3').value / cosmo.h**2    # critical density of the universe in h²M☉/Mpc³


def homogeneous_gridding_partilces(filename, lbox, ngrid, meshsize=256, chunk_size=10000000, desc=''):
    # header for dark matter particl file    
    header_type = np.dtype([("a", ">f8"), ("N", ">i4"), ("Dims", ">i4"), ("Ngas", ">i4"), ("Ndark", ">i4"), ("Nstar", ">i4"), ("pad", ">i4"),])
    dark_type = np.dtype([("mass", ">f4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4"), ("vx", ">f4"), ("vy", ">f4"), ("vz", ">f4"), ("eps", ">f4"), ("phi", ">f4")])

    # read tipsy-like binary file
    tipsy = open(filename, "rb")

    # get header
    header = np.fromfile(tipsy, dtype=header_type, count=1)
    header = dict(zip(header_type.names, header[0]))

    # get redshift
    z = 1./header['a']-1.

    # for the regular gridding
    gridding_bins = np.linspace(0, lbox, meshsize+1)

    # Debug: print first few particles with their bin indices to compare with C++
    debug_printed = False
    for i0 in tqdm(range(0, ngrid**3, chunk_size), desc=desc):

        # read tipsy in chuncks
        dark = np.fromfile(tipsy, dtype=dark_type, count=chunk_size)
        dark = dark.view((">f4", len(dark.dtype.names)))

        # get position and mass of dark matter particle
        pos_dark = (dark[:,1:4] + 1/2) * lbox
        mass_dark = dark[:,0] * ρ_c * lbox**3

        if (not debug_printed) and pos_dark.shape[0] > 0:
            # Mirror the C++ CPP_DEBUG output for the first few particles
            debug_count = min(20, pos_dark.shape[0])
            cell_size = lbox / meshsize
            print(f"PY_DEBUG first {debug_count} particles:")
            for p in range(debug_count):
                x, y, z = pos_dark[p]
                m = mass_dark[p]
                # Compute bin indices: floor((pos / cell_size)), clamped to [0, meshsize-1]
                nx = x / cell_size
                ny = y / cell_size
                nz = z / cell_size
                i = int(np.floor(nx))
                j = int(np.floor(ny))
                k = int(np.floor(nz))
                i = max(0, min(meshsize - 1, i))
                j = max(0, min(meshsize - 1, j))
                k = max(0, min(meshsize - 1, k))
                print(f"PY_DEBUG p={p} x={x:.17e} y={y:.17e} z={z:.17e} mass={m:.17e} bin=({i},{j},{k})")
            debug_printed = True

        # gridded mass (in M☉/h)
        if(i0 == 0):
            gridded_mass = stats.binned_statistic_dd(sample=pos_dark, values=mass_dark, statistic='sum', bins=[gridding_bins, gridding_bins, gridding_bins]).statistic
        else:
            gridded_mass += stats.binned_statistic_dd(sample=pos_dark, values=mass_dark, statistic='sum', bins=[gridding_bins, gridding_bins, gridding_bins]).statistic

    tipsy.close()

    # get density field (in M☉/(h Mpc³) units)
    dens = gridded_mass / np.diff(gridding_bins)[0]**3
    print("Total mass (Python) =", dens.sum() * (lbox/meshsize)**3)

    return dens, z


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Homogeneous gridding of PKDGRAV/TIPSY dark matter particles onto a regular mesh."
    )
    parser.add_argument("filename", help="Input dark-matter particle file in tipsy-like format")
    parser.add_argument("--lbox", type=float, required=True, help="Box size (same units as positions in file)")
    parser.add_argument(
        "--ngrid",
        type=int,
        required=True,
        help="Cube root of the number of particles (e.g. 256 for 256^3 particles)",
    )
    parser.add_argument(
        "--meshsize",
        type=int,
        default=256,
        help="Number of grid cells per dimension for the Eulerian mesh (default: 256)",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=10_000_000,
        help="Number of particles to read per chunk (default: 10_000_000)",
    )
    parser.add_argument(
        "--desc",
        type=str,
        default="gridding",
        help="Description label for tqdm progress bar",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output text file to store the density field (one value per line)",
    )
    return parser


def main() -> None:
    parser = _build_arg_parser()
    args = parser.parse_args()

    dens, z = homogeneous_gridding_partilces(
        filename=args.filename,
        lbox=args.lbox,
        ngrid=args.ngrid,
        meshsize=args.meshsize,
        chunk_size=args.chunk_size,
        desc=args.desc,
    )

    if args.output:
        # Save as plain text: one value per line, flattened in C order, with cell index
        flat = dens.ravel()
        indices = np.arange(flat.size, dtype=int)
        data = np.column_stack((indices, flat))
        # Two columns: cell_index  density_value
        np.savetxt(args.output, data, fmt=["%d", "%.18e"])
        print(f"Saved density field (flattened, {flat.size} values) with indices to {args.output}")

    print(f"Redshift z = {z:.6f}")
    print(f"Density field shape: {dens.shape}")


if __name__ == "__main__":
    main()