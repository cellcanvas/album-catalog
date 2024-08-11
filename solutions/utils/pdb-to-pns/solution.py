###album catalog: cellcanvas

from album.runner.api import setup, get_data_path, get_args

env_file = """
channels:
  - conda-forge
dependencies:
  - python=3.10
  - biopython
  - scipy
  - numpy
"""

def run():
    import numpy as np
    from Bio.PDB import PDBParser
    from Bio.PDB.PDBList import PDBList
    from scipy.spatial import distance
    import os

    def fetch_pdb_file(pdb_id):
        pdbl = PDBList()
        pdbl.retrieve_pdb_file(pdb_id, file_format='pdb', pdir='.')
        return f"./pdb{pdb_id.lower()}.ent"

    def read_pdb_file(pdb_file):
        parser = PDBParser(QUIET=True)
        structure = parser.get_structure('', pdb_file)
        atoms = []
        for model in structure:
            for chain in model:
                for residue in chain:
                    for atom in residue:
                        atoms.append(atom.coord)
        return np.array(atoms)

    def generate_grid(atoms, grid_resolution, grid_spacing):
        min_coord = atoms.min(axis=0) - grid_spacing
        max_coord = atoms.max(axis=0) + grid_spacing
        grid_shape = np.ceil((max_coord - min_coord) / grid_resolution).astype(int)

        grid = np.zeros(grid_shape)

        for ix in range(grid_shape[0]):
            for iy in range(grid_shape[1]):
                for iz in range(grid_shape[2]):
                    point = min_coord + np.array([ix, iy, iz]) * grid_resolution
                    distances = distance.cdist([point], atoms, 'euclidean').min()
                    grid[ix, iy, iz] = 1 / (distances + 1e-12)

        return grid, min_coord

    def save_grid(grid, min_coord, grid_resolution, output_file):
        with open(output_file, 'w') as f:
            for ix in range(grid.shape[0]):
                for iy in range(grid.shape[1]):
                    for iz in range(grid.shape[2]):
                        point = min_coord + np.array([ix, iy, iz]) * grid_resolution
                        value = grid[ix, iy, iz]
                        f.write(f"{point[0]:.3f} {point[1]:.3f} {point[2]:.3f} {value:.5f}\n")

    args = get_args()
    pdb_id = args.pdb_id
    grid_resolution = args.grid_resolution
    grid_spacing = args.grid_spacing
    output_file = args.output_file

    pdb_file = fetch_pdb_file(pdb_id)
    atoms = read_pdb_file(pdb_file)
    grid, min_coord = generate_grid(atoms, grid_resolution, grid_spacing)
    save_grid(grid, min_coord, grid_resolution, output_file)

    os.remove(pdb_file)
    print(f"PNS file saved as {output_file}")

setup(
    group="utils",
    name="pdb-to-pns",
    version="0.0.2",
    title="Generate a PNS density image from PDB ID",
    description="Generate a Point Normal Surface (PNS) file from a PDB ID.",
    solution_creators=["Kyle Harrington"],
    cite=[{"text": "Cellcanvas and copick teams", "url": "https://cellcanvas.org/"}],
    tags=["PDB", "grid", "surface", "structural biology", "pns"],
    license="MIT",
    covers=[],
    album_api_version="0.5.1",
    args=[
        {"name": "pdb_id", "type": "string", "required": True, "description": "PDB ID of the structure."},
        {"name": "output_file", "type": "string", "required": True, "description": "Path to the output PNS file."},
        {"name": "grid_resolution", "type": "float", "required": False, "description": "Grid resolution for the PNS.", "default": 10.0},
        {"name": "grid_spacing", "type": "float", "required": False, "description": "Grid spacing around the structure.", "default": 50.0},
    ],
    run=run,
    dependencies={
        "environment_file": env_file
    },
)
