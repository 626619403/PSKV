sbatch ./bash_script/autodan/kv_cache_ours.sh
sbatch ./bash_script/autodan/kv_cache_none.sh
sbatch ./bash_script/autodan/kv_cache_normal.sh

sbatch ./bash_script/gcg/kv_cache_ours.sh
sbatch ./bash_script/gcg/kv_cache_none.sh
sbatch ./bash_script/gcg/kv_cache_normal.sh

sbatch ./bash_script/gcq/kv_cache_ours.sh
sbatch ./bash_script/gcq/kv_cache_none.sh
sbatch ./bash_script/gcq/kv_cache_normal.sh

sbatch ./bash_script/beast/kv_cache_ours.sh
sbatch ./bash_script/beast/kv_cache_none.sh
sbatch ./bash_script/beast/kv_cache_normal.sh

sbatch ./bash_script/beast/kv_cache_ours.sh
sbatch ./bash_script/beast/kv_cache_none.sh
sbatch ./bash_script/beast/kv_cache_normal.sh

squeue -u $USER