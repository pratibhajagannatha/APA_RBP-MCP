# APA Virtual Screen

`apa_virtual_screen.py` is a command-line tool that runs a **fine-tuned ProteinBERT** model to virtually screen proteins for **poly(A) site (PAS) activation** potential. The details of model development and fine-tuning are outlined in the study **Large-scale tethered screen of RNA-binding proteins reveals novel regulators of poly(A) site selection** by Jagannatha et al. The work builds on prior models and methods outlined in Jin et al. 2024 (HydRA) and Brandes et al. 2022 (ProteinBERT). 

---

## What this does (at a glance)

- Loads a **pickled ProteinBERT model generator** (fine-tuned for APA activator classification).
- It reads a list of proteins, loads per-protein FASTA sequences,
- applies a the fine-tuned ProteinBERT model
- Predicts a **continuous score** per protein (higher ⇒ more activator-like).
- Saves `APA_activator_prediction_scores.txt` with columns: `Protein`, `ProteinBERT_score` with **per-protein scores**.

---

## Requirements

To use this tool, your environment should be set up in accordance with ProteinBERT and HydRA's requirements. 

---

### Using the model file

To use the model file, navigate to the `model_files` directory and run the following in bash:

<pre> bash
   # Navigate into your repo 
   cd ./model_files 
   
   # Concatenate model files to create your final model files 
   cat ProteinBERT_finetuned_APA_classification_model_part_* > ProteinBERT_finetuned_APA_classification.pkl 
</pre>

## Inputs

1. **Model file** (`--model_file`)  
   Pickle file produced when saving your fine-tuned ProteinBERT “model generator”.

2. **Protein list TSV** (`--rbps`)  
   A tab-separated file with a column containing protein identifiers (default column name: `key`).  
   Example `example_input.tsv`:

   ```text
   key
   CPSF6
   RNPS1
   GRB2
   ```

3. **Sequences directory** (`--seq_dir`)  
   Folder containing **one FASTA per protein**, named **`<Protein>.fasta`**  
   (FASTA header on line 1, sequence on subsequent lines).  
   Example: `seqs/GRB2.fasta`, `seqs/CPSF6.fasta`, …

4. **Output directory** (`--out`)  
   Where the results TSV will be written.

5. **Column override** (`--protein_col`)  
   If your TSV uses a different column name than `key`, set this.

---

## Usage

```bash
python3 ./apa_virtual_screen.py \
  --model_file ./ProteinBERT_finetuned.pkl \
  --rbps ./example_input.tsv \
  --seq_dir ./seqs \
  --out ./scores_out \
  --protein_col key
```

**Output:**

```
scores_out/APA_activator_prediction_scores.txt
```

TSV schema:

| Protein | ProteinBERT_score |
|--------:|-------------------|
|  CPSF6  | 0.73              |
|  RNPS1  | 0.64              |
|  GRB2   | 0.58              |


---

## Interpreting scores

- The output is a **continuous score** in [0,1] (binary output head).  
- To convert to “activator / non-activator”, apply your study’s **chosen threshold** (e.g., a recall-oriented or F1-optimized cutoff) **after** this script.

---

---

## Citation & attribution

If you use this script or derived outputs in a publication, please cite:

- The **Large-scale tethered screen of RNA-binding proteins reveals novel regulators of poly(A) site selection** (Jagannatha *et al.*).  
- **ProteinBERT: a universal deep-learning model of protein sequence and function**: Brandes N. *et al.* (2022).  
- **HydRA: Deep-learning models for predicting RNA-binding capacity from protein interaction association context and protein sequence** (concepts referenced/adapted in comments): Jin W. *et al.* (2024).
