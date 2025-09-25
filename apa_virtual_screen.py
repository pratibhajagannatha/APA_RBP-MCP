#!/usr/bin/env python3
"""
The purpose of this code is to use the fine-tuned ProteinBERT model from Jagannatha et al. as a virtual screening tool for classifying proteins as activators or non-activators of PAS selection.
Much of this code is adapted from the HydRA publication by Jin W. et al. (2023) and ProteinBERT publication by Brandes N. et al. (2022).
This script requires an environment compatible with running ProteinBERT, as outlined in https://github.com/nadavbra/protein_bert. 

USAGE (example):

python3 ./apa_virtual_screen.py \
    --model_file ./ProteinBERT_finetuned_0.3d_0.0005lr_16b_768len_1536flen_modelfile_nosim_wheldout_revision.pkl \
    --rbps ./example_input.tsv \
    --seq_dir ./seqs/ \
    --out ./example_out \
    --protein_col key
      

Inputs:
  --model_file : Pickled ProteinBERT model generator fine-tuned for APA activator classification
  --rbps       : TSV file with one column containing RBP names (default column name: "key")
  --seq_dir    : Directory containing per-protein FASTA files named "<RBP>.fasta"
  --out        : Output directory for results
  --protein_col: Column in --rbps that contains the protein/RBP names (default: "key")

Output:
  Writes a TSV "APA_activator_prediction_scores.txt" with:
    Protein \t ProteinBERT_score
"""

import argparse
import os
import pickle

import numpy as np
import pandas as pd

from proteinbert import OutputType, OutputSpec, InputEncoder

# -----------------------------
# Global configuration constants
# -----------------------------

OUTPUT_TYPE = OutputType(False, 'binary')
UNIQUE_LABELS = [0, 1]
OUTPUT_SPEC = OutputSpec(OUTPUT_TYPE, UNIQUE_LABELS)

# Number of special tokens ProteinBERT adds
ADDED_TOKENS_PER_SEQ = 2

# Number of functional annotations for InputEncoder
n_annotations = 8943


def get_y_pred_ProteinBERT(
    model_ProteinBERT_generator,
    ProteinBERT_input_encoder,
    output_spec,
    seqs,
    raw_Y,
    protein_names,
    start_seq_len=512,
    start_batch_size=32,
    increase_factor=2,
):
    
    # Ensure no stale optimizer weights are present
    assert getattr(model_ProteinBERT_generator, "optimizer_weights", None) is None

    dataset = pd.DataFrame({'seq': seqs, 'raw_y': raw_Y, 'protein': protein_names})

    y_preds = []
    protein_names_list = []

    for len_matching_dataset, seq_len, batch_size in split_dataset_by_len2(
        dataset, start_seq_len=start_seq_len, start_batch_size=start_batch_size, increase_factor=increase_factor
    ):
        # Encode inputs for this bucket
        X, y_true, sample_weights, prot_names = encode_dataset(
            len_matching_dataset['seq'],
            len_matching_dataset['raw_y'],
            ProteinBERT_input_encoder,
            output_spec,
            seq_len=seq_len,
            needs_filtering=False,
            protein_names=len_matching_dataset['protein'],
        )

        # Skip empty buckets
        if len(X[0]) == 0 or len(X[1]) == 0:
            continue

        # Use only samples with weight==1 (mask supplied by ProteinBERT pipeline)
        mask = (sample_weights == 1)

        # Create and run the model for this sequence length
        model = model_ProteinBERT_generator.create_model(seq_len)
        y_pred = model.predict(X, batch_size=batch_size)

        # Flatten shape depending on output type
        if output_spec.output_type.is_categorical:
            y_pred = y_pred.reshape((-1, y_pred.shape[-1]))
        else:
            y_pred = y_pred.flatten()

        # Apply mask and collect
        y_preds.append(y_pred[mask])
        protein_names_list.append(prot_names[mask])

    # Concatenate across buckets
    y_pred = np.concatenate(y_preds, axis=0) if y_preds else np.array([])
    protein_names = np.concatenate(protein_names_list, axis=0) if protein_names_list else np.array([])

    return y_pred, protein_names


def split_dataset_by_len2(dataset, seq_col_name='seq', start_seq_len=512, start_batch_size=32, increase_factor=2):
    """
    Yield subsets of the dataset that fit within a maximum sequence length, increasing the allowed
    length each iteration and adjusting batch size accordingly.
    """
    seq_len = start_seq_len
    batch_size = start_batch_size

    while len(dataset) > 0:
        max_allowed_input_seq_len = seq_len - ADDED_TOKENS_PER_SEQ
        len_mask = (dataset[seq_col_name].str.len() <= max_allowed_input_seq_len)
        len_matching_dataset = dataset[len_mask]
        if len(len_matching_dataset) > 0:
            yield len_matching_dataset, seq_len, batch_size
        dataset = dataset[~len_mask]
        seq_len += start_seq_len            # alternatively: seq_len *= increase_factor
        batch_size = max(batch_size // increase_factor, 1)


def encode_dataset(
    seqs,
    raw_Y,
    ProteinBERT_input_encoder,
    output_spec,
    seq_len=512,
    needs_filtering=True,
    dataset_name='Dataset',
    verbose=True,
    protein_names=None,
):
    """
    Encode sequences (X) and labels (Y) for ProteinBERT.
    In this script we set needs_filtering=False upstream, so we do not filter by length here.
    """
    X = ProteinBERT_input_encoder.encode_X(seqs, seq_len)
    Y, sample_weights = encode_Y(pd.Series(raw_Y), output_spec, seq_len=seq_len)
    return X, Y, sample_weights, np.array(list(protein_names))


def encode_Y(raw_Y, output_spec, seq_len=512):
    """
    Encode labels based on output_spec. For binary outputs here we return flat arrays of 1s (dummy labels).
    """
    if output_spec.output_type.is_seq:
        # For sequence outputs, one would encode position-wise labels; not used in this script.
        raise ValueError("Sequence outputs not supported in this scoring script.")
    elif output_spec.output_type.is_categorical:
        # For categorical outputs, return one-hot labels; not used in this script.
        raise ValueError("Categorical outputs not supported in this scoring script.")
    elif output_spec.output_type.is_numeric or output_spec.output_type.is_binary:
        # Binary / numeric outputs: return as float vector with unit weights.
        return raw_Y.values.astype(float), np.ones(len(raw_Y))
    else:
        raise ValueError(f"Unexpected output type: {output_spec.output_type}")


def main():
    parser = argparse.ArgumentParser(description="Export ProteinBERT scores for a list of RBPs.")
    parser.add_argument("--model_file", required=True, help="Path to ProteinBERT model file (pickle).")
    parser.add_argument("--rbps", required=True, help="Path to TSV file of RBPs (column specified by --protein_col).")
    parser.add_argument("--seq_dir", required=True, help="Directory with FASTA files named <RBP>.fasta.")
    parser.add_argument("--out", required=True, help="Directory to write output TSV file.")
    parser.add_argument("--protein_col", default="key", help="Column in --rbps that contains RBP names.")

    args = parser.parse_args()

    # Load RBP names from TSV; expects a column (default 'key') with protein identifiers
    rbp_df = pd.read_csv(args.rbps, sep="\t")
    if args.protein_col not in rbp_df.columns:
        raise ValueError(f"--protein_col '{args.protein_col}' not found in {args.rbps}. "
                         f"Available columns: {list(rbp_df.columns)}")

    rbp_list = rbp_df[args.protein_col].astype(str).tolist()
    proteins = rbp_list.copy()
    print("Number of proteins to be predicted on:", len(proteins))

    # Gather sequences from per-protein FASTA files
    seq_dic_ProteinBERT = {}
    for prot in proteins:
        fasta_path = os.path.join(args.seq_dir, f"{prot}.fasta")
        if os.path.exists(fasta_path):
            with open(fasta_path) as f:
                # Skip FASTA header line, join remaining lines, strip * (stop) and whitespace
                lines = f.read().strip().splitlines()
                seq = "".join(lines[1:]).replace("*", "")
                # Keep X characters for ProteinBERT input, but also produce a version without X if needed
                seq_dic_ProteinBERT[prot] = seq
        else:
            print(f"[WARN] FASTA not found for {prot}: {fasta_path}")

    # Prepare inputs for ProteinBERT
    seq_names = list(seq_dic_ProteinBERT.keys())
    prot_seqs = [seq_dic_ProteinBERT[p] for p in seq_names]
    dummy_labels = np.ones(len(seq_names))  # dummy labels for API compatibility

    # Load model generator and input encoder
    if not os.path.exists(args.model_file):
        raise FileNotFoundError(f"Model file not found: {args.model_file}")

    with open(args.model_file, "rb") as f:
        model_generator = pickle.load(f)
    input_encoder = InputEncoder(n_annotations)

    # Run predictions
    y_pred, prot_names = get_y_pred_ProteinBERT(
        model_generator,
        input_encoder,
        OUTPUT_SPEC,
        prot_seqs,
        dummy_labels,
        seq_names,
        start_seq_len=512,
        start_batch_size=32,
    )

    # Write output TSV
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(args.out, "APA_activator_prediction_scores.txt")
    df_out = pd.DataFrame({"Protein": prot_names, "ProteinBERT_score": y_pred})
    df_out.to_csv(out_path, sep="\t", index=False)

    print(f"[OK] Wrote scores to: {out_path}")


if __name__ == "__main__":
    main()
