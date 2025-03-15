import argparse
import deepspeed

# train
parser = argparse.ArgumentParser()
parser.add_argument('--world_size', type=int, default=4)
parser.add_argument('--local_rank', type=int, default=-1)
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--device', type=str, default='cuda')
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--llm_epochs', type=int, default=200)
parser.add_argument('--llm_lr', type=float, default=1e-5)
parser.add_argument('--lambda_contrast', type=float, default=1)
parser.add_argument('--decay', type=float, default=0.00005)
parser.add_argument('--eval_every_n_epochs', type=int, default=1)
parser.add_argument('--early_stop_epochs_llm', type=int, default=15)
parser.add_argument('--output_save_path', type=str, default='../checkpoints/finetune/')
parser.add_argument('--loss_temperature', type=float, default=0.1)
parser.add_argument('--save_repr', action='store_true', help="Set to True if --save_repr is specified; defaults to False")
parser.add_argument('--generate_cot', action='store_true', help="Set to True if --generate_cot is specified; defaults to False")

# llama model
parser.add_argument('--model_name', type=str, default='../llama3-8b-instruct')
parser.add_argument('--smiles_max_length', type=int, default=400)
parser.add_argument('--cot_max_length', type=int, default=800)
parser.add_argument('--final_prompt_max_length', type=int, default=400)

# lora
parser.add_argument('--use_lora', action='store_true')
parser.add_argument('--no_use_lora', action='store_false', dest='use_lora')
parser.add_argument('--lora_rank', type=int, default=4)
parser.add_argument('--lora_alpha', type=int, default=16)
parser.add_argument('--lora_dropout', type=float, default=0.05)

# gnn model
parser.add_argument('--atom_feature_size', type=int, default=9)
parser.add_argument('--bond_feature_size', type=int, default=3)
parser.add_argument('--attentionfp_input_size', type=int, default=300)
parser.add_argument('--attentionfp_hidden_size', type=int, default=300)
parser.add_argument('--attentionfp_output_size', type=int, default=300)
parser.add_argument('--atom_layers', type=int, default=4)
parser.add_argument('--mol_layers', type=int, default=2)
parser.add_argument('--gnn_dropout_ratio', type=float, default=0.2)

parser.add_argument('--cross_attn_num_heads', type=int, default=2)
parser.add_argument('--gnn_weight', type=float, default=0)
parser.add_argument('--llm_weight', type=float, default=1)

# prediction head
parser.add_argument('--mlp_input_dim', type=int, default=4096)
parser.add_argument('--mlp_hidden_dim', type=int, default=256)
parser.add_argument('--mlp_output_dim', type=int, default=1)

# dataset
parser.add_argument('--root', type=str, default='../data', help="root")
parser.add_argument('--valid_rate', type=float, default=0.1, help="valid_rate")
parser.add_argument('--test_rate', type=float, default=0.1, help="test_rate")
parser.add_argument('--split_type', type=str, default='random', help="split_type")
parser.add_argument('--num_workers', type=int, default=0, help="num_workers")
parser.add_argument('--split_seed', type=int, default=7, help="split_seed")
parser.add_argument('--use_multimodal', action='store_true')
parser.add_argument('--no_use_multimodal', action='store_false', dest='use_multimodal')
parser.add_argument('--use_cot', action='store_true', help="Set use_cot to True")
parser.add_argument('--no_use_cot', action='store_false', dest='use_cot', help="Set use_cot to False")
parser.add_argument('--encoding', type=str, default='UTF-8', help="encoding")

parser.add_argument('--datasets', nargs='+', default=['Tox21'], help="List of datasets to use")
parser.add_argument('--num_tasks', type=int, default=1, help="num_tasks")
parser.add_argument('--num_labels', type=int, default=12, help="num_labels")

# deepspeed
parser = deepspeed.add_config_arguments(parser)

args = parser.parse_args()

args.finetune_model_save_path = args.output_save_path + args.datasets[0] + '/llama_predictor'
args.finetune_MLP_save_path = args.output_save_path + 'predictor'
args.embedding_save_path = args.output_save_path + 'llm_embedding'

args.dataset_task_type = {
    'BACE': 'classification',  # 1
    'BBBP': 'classification',  # 1
    'HIV':  'classification',  # 1
    'ClinTox': 'classification',  # 2
    'Sider': 'classification',  # 27
    'Tox21': 'classification',   # 12
    'ToxCast': 'classification', # 617
    'qm8': 'regression', # 16
    'ESOL': 'regression', # 1
    'Lipo': 'regression',
    'FreeSolv': 'regression', # 1
    'QM9': 'regression',
    'QM8': 'regression',
    'QM7': 'regression',
}

args.best_valid_initial = {
    'BACE': 0,  # 1
    'BBBP': 0,  # 1
    'HIV':  0,  # 1
    'ClinTox': 0,  # 2
    'Sider': 0,  # 27
    'Tox21': 0,   # 12
    'ToxCast': 0, # 617
    'qm8': 10000, # 16
    'ESOL': 10000, # 1
    'Lipo': 10000,
    'FreeSolv': 10000, # 1
    'QM9': 10000,
    'QM8': 10000,
    'QM7': 10000,
}

args.best_valid_test = {
    'BACE': 0,  # 1
    'BBBP': 0,  # 1
    'HIV':  0,  # 1
    'ClinTox': 0,  # 2
    'Sider': 0,  # 27
    'Tox21': 0,   # 12
    'ToxCast': 0, # 617
    'qm8': 0, # 16
    'ESOL': 0, # 1
    'Lipo': 0,
    'FreeSolv': 0, # 1
    'QM9': 0,
    'QM8': 0,
    'QM7': 0,
}

args.datasets_property_prompt = {
    'BACE': 'BACE-1 Inhibit',
    'BBBP': 'Brain-blood Barrier Penetration',
    'ClinTox': 'FDA-approved and Clinically-trial-Toxic',
    'HIV': 'HIV replication Inhibit',
    'Sider': 'Side Effect',
    'Tox21': 'Toxicity',
    'ESOL': 'Estimated Solubility in Water',
    'FreeSolv': 'Solvation Free Energy',
    'Lipo': 'Lipophilicity',
    'QM9': 'Quantum Chemical Properties',
    'QM8': 'Molecular Optical Properties',
    'QM7': 'Atomization Energy',
}
