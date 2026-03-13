sbatch ./bash_script/advPrompter/harmbench/kv_cache_none.sh
sbatch ./bash_script/advPrompter/harmbench/kv_cache_normal.sh
sbatch ./bash_script/advPrompter/harmbench/kv_cache_ours.sh

sbatch ./bash_script/advPrompter/advbench/kv_cache_none.sh
sbatch ./bash_script/advPrompter/advbench/kv_cache_normal.sh
sbatch ./bash_script/advPrompter/advbench/kv_cache_ours.sh

sbatch ./bash_script/ampleGCG/harmbench/kv_cache_none.sh
sbatch ./bash_script/ampleGCG/harmbench/kv_cache_normal.sh
sbatch ./bash_script/ampleGCG/harmbench/kv_cache_ours.sh

sbatch ./bash_script/ampleGCG/advbench/kv_cache_none.sh
sbatch ./bash_script/ampleGCG/advbench/kv_cache_normal.sh
sbatch ./bash_script/ampleGCG/advbench/kv_cache_ours.sh

squeue -u $USER

