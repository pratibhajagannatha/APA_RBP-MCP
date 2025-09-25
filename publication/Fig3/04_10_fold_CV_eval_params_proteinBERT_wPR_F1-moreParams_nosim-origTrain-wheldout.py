import os
import pandas as pd
import numpy as np
from itertools import product
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.utils.class_weight import compute_class_weight
from proteinbert import load_pretrained_model, OutputSpec, FinetuningModelGenerator, evaluate_by_len, OutputType, finetune
from proteinbert.conv_and_global_attention_model import get_model_with_hidden_layers_as_outputs
from tensorflow import keras

# STEP 1: Hold out 25% as external test set
df = pd.read_csv('RBP_train_test_eval_bkg_corrected_nosim.tsv', sep='\t')
df['Activator_label'] = df['Activator_label'].astype(int)
train_data_dir='/tscc/nfs/home/pjaganna1/projects/apa_screen_yongsheng/hydra_2/train_test_eval_noppipia/train_eval_out_10_nosim_wheldout/'


# STEP 2: Define hyperparameter grid
param_grid = {
    'dropout': [0.2, 0.3],
    'lr': [5e-4, 1e-4],
    'batch_size': [16, 32],
    'seq_len': [512, 768],
    'final_seq_len': [1024, 1536],
}

param_combinations = list(product(*param_grid.values()))
param_names = list(param_grid.keys())

# Paths
result_file = 'proteinBERT_train_test_results_wPR_F1_moreParams_eval_wFolds_nosim_origTrain_wheldout.txt'
ProteinBERT_pretrainedBeforeFinetune_model = '/tscc/nfs/home/pjaganna1/projects/apa_screen_yongsheng/hydra_2/fine-tuning/default.pkl'
model_dir = os.path.dirname(ProteinBERT_pretrainedBeforeFinetune_model)
model_file = os.path.basename(ProteinBERT_pretrainedBeforeFinetune_model)

# Load pretrained model
pretrained_model_generator, input_encoder = load_pretrained_model(model_dir, model_file)
output_spec = OutputSpec(OutputType(False, 'binary'), [0, 1])

# Load previous results if available
if os.path.exists(result_file):
    results_df = pd.read_csv(result_file, sep='\t')
    completed_combos = [tuple(row[param] for param in param_names) for _, row in results_df.iterrows()]
else:
    results_df = pd.DataFrame()
    completed_combos = []

# Cross-validation

for count, combo in enumerate(param_combinations):
    if combo in completed_combos:
        print(f"Skipping already completed combo {combo}")
        continue

    params = dict(zip(param_names, combo))
    print(f"\n=== Testing params: {params} ===")
    fold_aucs, fold_precision, fold_recall, fold_f1 = [], [], [], []

    for fold in range(0,10):
        # Using folds used originally to benchmark data
        train_ids = pd.read_csv(os.path.join(train_data_dir, f'fold_{fold}_train.txt'),
            sep='\t', header=None)[0].tolist()
        test_ids = pd.read_csv(os.path.join(train_data_dir, f'fold_{fold}_test.txt'),
            sep='\t', header=None)[0].tolist()
        train_fold = df[df['key'].isin(train_ids)]
        test_fold = df[df['key'].isin(test_ids)]

        train_fold, val_fold = train_test_split(train_fold, test_size=0.1, stratify=train_fold['Activator_label'], random_state=42)

        class_weights = compute_class_weight(class_weight='balanced', classes=[0, 1], y=train_fold['Activator_label'].values)
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

        model_generator = FinetuningModelGenerator(
            pretrained_model_generator,
            output_spec,
            dropout_rate=params['dropout'],
            pretraining_model_manipulation_function=get_model_with_hidden_layers_as_outputs
        )

        finetune(
            model_generator,
            input_encoder,
            output_spec,
            train_fold['aa_sequence'], train_fold['Activator_label'],
            val_fold['aa_sequence'], val_fold['Activator_label'],
            seq_len=params['seq_len'],
            batch_size=params['batch_size'],
            max_epochs_per_stage=40,
            lr=params['lr'],
            begin_with_frozen_pretrained_layers=True,
            lr_with_frozen_pretrained_layers=1e-02,
            n_final_epochs=1,
            final_seq_len=params['final_seq_len'],
            final_lr=1e-5,
            callbacks=[
                keras.callbacks.ReduceLROnPlateau(patience=1, factor=0.25, min_lr=1e-05, verbose=0),
                keras.callbacks.EarlyStopping(patience=2, restore_best_weights=True)
            ],
            class_weights=class_weight_dict
        )

        results, confusion_matrix = evaluate_by_len(
            model_generator,
            input_encoder,
            output_spec,
            test_fold['aa_sequence'], test_fold['Activator_label'],
            start_seq_len=params['seq_len'],
            start_batch_size=params['batch_size']
        )

        auc = results['AUC']['All']
        fold_aucs.append(auc)

        TN = confusion_matrix.loc['0', '0']
        FP = confusion_matrix.loc['0', '1']
        FN = confusion_matrix.loc['1', '0']
        TP = confusion_matrix.loc['1', '1']

        precision = TP / (TP + FP) if (TP + FP) > 0 else 0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        fold_precision.append(precision)
        fold_recall.append(recall)
        fold_f1.append(f1)

    mean_auc = np.mean(fold_aucs)
    mean_precision = np.mean(fold_precision)
    mean_recall = np.mean(fold_recall)
    mean_f1 = np.mean(fold_f1)

    result_row = {
        **params,
        'mean_auc': mean_auc,
        'mean_precision': mean_precision,
        'mean_recall': mean_recall,
        'mean_f1': mean_f1,
        'fold_auc' :fold_aucs,
        'fold_precision': fold_precision,
        'fold_recall':fold_recall,
        'fold_f1':fold_f1
    }

    results_df = pd.concat([results_df, pd.DataFrame([result_row])], ignore_index=True)
    results_df.to_csv(result_file, sep='\t', index=False)

    print(f"\n=== Completed combo {count+1}/{len(param_combinations)} ===")
