from pytfa.redgem.lumpgem import LumpGEM
from pytfa.io.json import load_json_model, save_json_model
from cobra.io.json import save_json_model as save_json_model_cobra
from cobra import Reaction
from cobra.util import Zero

recon3d = load_json_model('GEM_Recon3_thermocurated_redHUMAN.json')
recon3d.solver.configuration.tolerances.feasibility = 1e-9
recon3d.solver.configuration.tolerances.optimality = 1e-9 


import json
with open('recon3D_gene_annotation.json', 'r') as f:
    gene_annotation = json.load(f)
    gene_annotation = gene_annotation['results']

# Confert list to dict using bigg ids
gene_annotation_dict = {g['bigg_id']: g for g in gene_annotation}

# Use the new gene annotation
rename_dict = dict()
for gene in recon3d.genes:
    gene_id = '_AT'.join(gene.id.split('.'))
    gene_annotation = gene_annotation_dict.get(gene_id, None)
    if gene_annotation:
        gene.name = gene_annotation['name']
        rename_dict[gene.id] = gene_id

# Annotate the model with the gene annotation ontology
from cobra.manipulation.modify import rename_genes
rename_genes(recon3d, rename_dict)


save_json_model_cobra(recon3d, 'Recon3D_New_Genes_Thermo_RXN.json')

