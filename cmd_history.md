python run_gbda_attack.py --data_folder data --dataset mnli --model /mnt/cloud/bairu/repos/std_text_pgd_attack/checkpoints/bert-base-uncased-mnli/ --finetune False --start_index 0 --num_samples 1000 --gumbel_samples 100 --attack_target hypothesis  --lam_sim 50

python run_gbda_attack.py --data_folder data --dataset sst --model /mnt/cloud/bairu/repos/std_text_pgd_attack/checkpoints/bert-base-uncased-sst/ --finetune False --start_index 0 --num_samples 1821 --gumbel_samples 100  --lam_sim 50
