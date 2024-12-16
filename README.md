NPBench
=====

A turnkey library for benchmarking 3D biomolecular structure prediction models. 

# Installation

NPBench and it's dependencies are installed with the [Conda](https://docs.anaconda.com/miniconda/) package manager.

## From source

To build from source, you can clone the repository and follow with the normal building process:

```bash
git clone git@github.com:iambic-therapeutics/np-bench.git np-bench
cd np-bench

# Installing package with all dependencies from environment.yaml
make install
conda activate np-bench-env
```

If you would like to modify the code, we recommend installing the dev environment which supports formatting, type checking, and linting. Check out [Makefile](./Makefile) for available tools.

```bash
# Building dev environment (defined in environment-dev.yaml)
make install-dev
conda activate np-bench-dev

# Formating code
make format

# Running all pytests
make test

# Running static checks (formatting and mypy)
make checks
```

# Running benchmarks locally

NPBench currently supports evaluation of NeuralPLexer3 and related structure prediction models on two datasets, Posebusters and the Recent PDB Evaluation Set. Each has its own command and yields a distinct set of metrics:

| Dataset                | Command                    | Metrics                                                            |
|------------------------|----------------------------|--------------------------------------------------------------------|
| Posebusters            | `np-bench posebusters`     | Pocket-aligned RMSD \| Global TM-score \| Posebusters checks       |
| Recent PDB Evalulation | `np-bench recent-pdb-eval` | DockQ score \| Generalized RMSD \| Global and chain-wise TM-scores |

To perform benchmarking on these datasets, you will need to provide the paths to the dataset and the model predictions with particular file formats and folder structures.
Please make sure to convert your model predictions to the required file formats and folder structures before running the benchmarking commands.
Below are detailed usages of the two benchmarking commands and the corresponding description of the required folder structures.

Note that the model predictions do not need to be aligned with the reference structures; NPBench handles structure postprocessing. 

The Posebusters and Recent PDB Evaluation datasets can be downloaded from [Zenodo](https://zenodo.org/records/14498115).

## Usage

- For Posebusters benchmarking, you can run `np-bench posebusters --help` to get detailed usage for the CLI. 

```
Usage: np-bench posebusters [OPTIONS]                                                                                                                    
                                                                                                                                                        
Run local benchmarking on Posebusters-like dataset.                                                                                                      

The dataset folder must have the following structure:                                                                                                    

dataset_folder/                                                                                                                                          
├── target_1/                                                                                                                                            
│   ├── target_1_protein.pdb                                                                                                                             
│   └── target_1_ligand.sdf                                                                                                                              
├── target_2/                                                                                                                                            
│   ├── target_2_protein.pdb                                                                                                                             
│   └── target_2_ligand.sdf                                                                                                                              
├── ...                                                                                                                                                  

The predictions folder must have the following structure:                                                                                                

predictions_folder/                                                                                                                                      
├── target_1/                                                                                                                                            
│   ├── conf_0/                                                                                                                                          
│   │   ├── prot.pdb                                                                                                                                     
│   │   └── lig_0.sdf                                                                                                                                    
│   ├── conf_1/                                                                                                                                          
│   │   ├── prot.pdb                                                                                                                                     
│   │   └── lig_0.sdf                                                                                                                                    
|   ├── ...                                                                                                                                              
├── target_2/                                                                                                                                            
│   ├── conf_0/                                                                                                                                          
│   │   ├── prot.pdb                                                                                                                                     
│   │   └── lig_0.sdf                                                                                                                                    
│   ├── conf_1/                                                                                                                                          
│   │   ├── prot.pdb                                                                                                                                     
│   │   └── lig_0.sdf                                                                                                                                    
|   |-- ...                                                                                                                                              
├── ...     

The top-ranked conformers, when applicable shall be stored in a `best_LG1` folder under the target name subdirectory.                                                                                                                                             
                                                                                                                                                        
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --dataset           -d      TEXT     Path to the Posebusters dataset folder. [default: None] [required]                                                                     │
│ *  --predictions       -p      TEXT     Path to the predictions folder. [default: None] [required]                                                                             │
│    --num-conf          -n      INTEGER  Number of conformations to evaluate. [default: None]                                                                                   │
│    --conf-idx          -c      INTEGER  Conformer index to evaluate. [default: None]                                                                                           │
│    --score-top-ranked                   Score the top ranked conformations.                                                                                                    │
│    --use-cache                          Use cached results if available.                                                                                                       │
│    --help                               Show this message and exit.                                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

- For Recent PDB Evaluation benchmarking, run `np-bench recent-pdb-eval --help` to get detailed usage for the CLI. 

```
Usage: np-bench recent-pdb-eval [OPTIONS]                                                                                                                
                                                                                                                                                        
Run local benchmarking on Recent PDB Evaluation Set.     
                                                                                                                                                          
╭─ Options ──────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╮
│ *  --dataset           -d      TEXT     Path to the Recent PDB Evaluation Set. [default: None] [required]                                                                      │
│    --index             -i      TEXT     Path to the dataset index csv file. [default: results/recent_pdb_eval_set_v2_w_CASP15RNA_reduced.csv]                                  │
│ *  --predictions       -p      TEXT     Path to the prediction cif or pdb files. [default: None] [required]                                                                    │
│    --num-conf          -n      INTEGER  Number of conformations to evaluate. [default: None]                                                                                   │
│    --conf-idx          -c      INTEGER  Conformer index to evaluate. [default: None]                                                                                           │
│    --score-top-ranked                   Score the top ranked conformations.                                                                                                    │
│    --use-cache                          Use cached results if available.                                                                                                       │
│    --help                               Show this message and exit.                                                                                                            │
╰────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────╯
```

The reference index file containing 1,143 target is provided at `results/recent_pdb_eval_set_v2_w_CASP15RNA_reduced.csv`.
To create a custom index for benchmarking on new datasets, the csv file must contain the following fields:
```
mmcif_id,this_chain_or_ccd_id,this_interface_id,eval_type
```
where:
- `mmcif_id` is the identifier of the target, using the PDB ID or biological assembly ID is recommended
- `this_chain_or_ccd_id` is the chain ID or CCD ID of the chain or interface of interest for scoring
- `this_interface_id` is the interface ID of the interface of interest for scoring
- `eval_type` is the type of evaluation, must be one of the following values:
    - 'DNA': RNA chain
    - 'RNA': RNA chain
    - 'protein': protein chain
    - 'protein:protein': protein-protein interface
    - 'peptide:protein': peptide-protein interface
    - 'ligand:protein': ligand-protein interface
    - 'RNA:protein': RNA-protein interface
    - 'DNA:protein': DNA-protein interface
    - 'DNA:ligand': DNA-ligand interface
    - 'RNA:ligand': RNA-ligand interface

The benchmark dataset and the index file `recent_pdb_eval_set_v2_w_CASP15RNA_reduced.csv` can be obtained [here](https://zenodo.org/records/14498115).
To use a custom dataset, the dataset folder must have the following structure:
```
dataset_folder/                                                                                                                                          
├── mmcif_id_1_eval.cif
├── mmcif_id_1_eval.fasta                                                                                                                                  
├── mmcif_id_2_eval.cif
├── mmcif_id_2_eval.fasta                                                                                                                                
├── ...                                                                                                                                                  

The fasta file is structured to contain all biological sequences and ligand SMILES information. For new structure prediciton algorithms we advocate the use of `.cif` files
as the primary input model to include more flexibility and details regarding chemical modifications and branch entities. If mmCIF input is not supported for the method of
interest, the accompanying .fasta file can be used as a close approximation.

The predictions folder must have the following structure:                                                                                                

predictions_folder/                                                                                                                                      
|── mmcif_id_1/                                                                                                                                          
|   ├── conf_0/                                                                                                                                          
|   │   ├── output.cif # or output.pdb                                                                                                                   
|   ├── conf_1/                                                                                                                                          
|   │   ├── output.cif # or output.pdb                                                                                                                   
|   ├── ...                                                                                                                                              
|── mmcif_id_2/                                                                                                                                          
|   ├── conf_0/                                                                                                                                          
|   │   ├── output.cif # or output.pdb                                                                                                                   
|   ├── ...                                                                                                                                              
|── ...                                                                                                                                                  
```

The top-ranked conformers for each chain or interface of interest shall be stored in a `best_{chain_or_interface_id}` 
folder under the target name subdirectory of each method. For example:
```
method-ranked/7QR3_1
├── best_poly:C
│   └── output.cif
├── best_lig:PTR||poly:A
│   └── output.cif
└── best_poly:A||poly:C
    └── output.cif
```


## Visualizing benchmarking results

After running the benchmarking commands, the results will be saved in the csv format in folder `metrics` in the directory specified by the `--predictions` option.

To visualize the benchmarking results, we provide a simple CLI tool to generate plots. Example usage:
```bash
np-bench plot-stats \
  --method-name AF2M --scoring-df results/af2m_results/metrics/conf_1_metrics.csv \
  --method-name NP3 --scoring-df results/NPv3-base-ranked/metrics/top_ranked_metrics.csv
```
