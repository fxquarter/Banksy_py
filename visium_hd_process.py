import os
import numpy as np
import pandas as pd
import scanpy as sc
import seaborn as sns
import anndata
import matplotlib.pyplot as plt
from matplotlib.pyplot import rc_context

import warnings

warnings.filterwarnings('ignore')

tissue_position_df = pd.read_parquet('newvisiumhd/square_008um/spatial/tissue_positions.parquet')
tissue_position_df.to_csv('newvisiumhd/square_008um/spatial/tissue_positions_list.csv', index=False, header=None)
adata = sc.read_visium('newvisiumhd/square_008um/', library_id='P1')
adata.obs['sample'] = 'P1'
adata.var_names_make_unique()
output_path = 'data/visium_hd_processed'
adata.write(os.path.join(output_path, 'visium_hd_adata.h5ad'))