conda activate venusrem

# protein_name need pre input
# VenusREM Part

# Make substuitutions CSV
python src/data/get_sav.py \
    --fasta_file data/$protein_dir/$protein_name.fasta \
    --output_csv data/$protein_dir/substitutions

# Search homology sequences
protein_dir="VenusREM_data"
query_protein_name=<your_protein_name>   # Protein name，such as fluorescent_protein
protein_path=data/$protein_dir/aa_seq/$query_protein_name.fasta
database=<your_path>/uniref100.fasta     # Downloaded UniRef100 protein data（FASTA ）
 
evcouplings \
    -P output/$protein_dir/$query_protein_name \
    -p $query_protein_name \
    -s $protein_path \
    -d $database \
    -b "0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9" \
    -n 5 src/single_config_monomer.txt

# Select a2m file
protein_dir="VenusREM_data"
python src/data/select_msa.py \
    --input_dir output/$protein_dir \
    --output_dir data/$protein_dir

# Download pdb at data/VenusREM_data/pdbs/

# Get structure sequence
protein_dir="VenusREM_data"
python src/data/get_struc_seq.py \
    --pdb_dir data/$protein_dir/pdbs \
    --out_dir data/$protein_dir/struc_seq
    --vocab_size 2048

# Run model 
protein_dir="VenusREM_data"
python compute_fitness.py \
    --base_dir data/$protein_dir \
    --out_scores_dir result/$protein_dir
    --aa_seq_dir my_fasta \
    --struc_seq_dir my_struc_seq \
    --mutant_dir my_substitutions \

conda diactivate
conda activate ProSST

# ProSST

python ProSST.py

conda deactivate
conda activate VespaG

# VespaG

vespag predict -i test.fasta \
               -e out/esm2_embeddings.h5 \
               -o ./out \
               --single-csv

conda deactivate
conda activate metamodel

# Metamodel

python Metamodel.py
