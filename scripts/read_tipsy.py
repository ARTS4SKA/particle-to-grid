import numpy as np
import sys

def read_tipsy(filename):
    """!
    Read information from TIPSY files.

    The use of this is now discouraged as the preferred format is HDF5.
    However, it is mantained here just in case it is needed.

    Note that the naming convention is different than with HDF5 snapshots

    @param filename Name of the file to be read
    @return Tuple with header, gas, dark, star arrays
    """
    header_type = np.dtype([("time", ">f8"), ("N", ">i4"), ("Dims", ">i4"), ("Ngas", ">i4"), ("Ndark", ">i4"), ("Nstar", ">i4"), ("pad", ">i4"),])

    gas_type = np.dtype([("mass", ">f4"), ("x", ">f4"),("y", ">f4"), ("z", ">f4"), ("vx", ">f4"), ("vy", ">f4"), ("vz", ">f4"), ("rho", ">f4"), ("temp", ">f4"), ("hsmooth", ">f4"), ("metals", ">f4"), ("phi", ">f4"),])
    dark_type = np.dtype([("mass", ">f4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4"), ("vx", ">f4"), ("vy", ">f4"), ("vz", ">f4"), ("eps", ">f4"), ("phi", ">f4")])
    star_type = np.dtype([("mass", ">f4"), ("x", ">f4"), ("y", ">f4"), ("z", ">f4"), ("vx", ">f4"), ("vy", ">f4"), ("vz", ">f4"), ("metals", ">f4"), ("tform", ">f4"), ("eps", ">f4"), ("phi", ">f4"),])

    # read tipsy-like binary file
    tipsy = open(filename, "rb")

    # get header
    header = np.fromfile(tipsy, dtype=header_type, count=1)
    header = dict(zip(header_type.names, header[0]))
    
    # get fields
    gas = np.fromfile(tipsy, dtype=gas_type, count=header["Ngas"])
    dark = np.fromfile(tipsy, dtype=dark_type, count=header["Ndark"])
    star = np.fromfile(tipsy, dtype=star_type, count=header["Nstar"])

    # convert a numpy structured array to a regular 2D array (allows slicing like arr[:, i])
    gas = gas.view((">f4", len(gas.dtype.names)))
    dark = dark.view((">f4", len(dark.dtype.names)))
    star = star.view((">f4", len(star.dtype.names)))

    tipsy.close()
    return header, gas, dark, star

if __name__ == "__main__":
    # first cmdline argument: tipsy file name to read
    fname = sys.argv[1]

    header, gas, dark, star = read_tipsy(fname)


    print(f"  Time: {header['time']}")
    print(f"  Total particles (N): {header['N']}")
    print(f"  Dimensions: {header['Dims']}")
    print(f"  Gas particles: {header['Ngas']}")
    print(f"  Dark matter particles: {header['Ndark']}")
    print(f"  Star particles: {header['Nstar']}")

    print(f"\nParticle data shapes:")
    if header['Ngas'] > 0:
        print(f"  Gas array shape: {gas.shape}")
        print(f"  Gas data type: {gas.dtype}")
        if gas.size > 0:
            print(f"  Sample gas x positions (first 5): {gas[:min(5, len(gas)), 1]}")
    if header['Ndark'] > 0:
        print(f"  Dark matter array shape: {dark.shape}")
        print(f"  Dark matter data type: {dark.dtype}")
        if dark.size > 0:
            print(f"  Sample dark x positions (first 5): {dark[:min(5, len(dark)), 1]}")
    if header['Nstar'] > 0:
        print(f"  Star array shape: {star.shape}")
        print(f"  Star data type: {star.dtype}")
        if star.size > 0:
            print(f"  Sample star x positions (first 5): {star[:min(5, len(star)), 1]}")