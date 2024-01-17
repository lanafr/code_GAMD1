#!/bin/sh

#SBATCH --job-name=gamdplus
#SBATCH --output=gamdplus-%A.out  # Standard output of the script (Can be absolute or relative path). %A adds the job id to the file name so you can launch the same script multiple times and get d
#SBATCH --error=gamdplus-%A.err  # Standard error of the script
#SBATCH --time=0-24:00:00  # Limit on the total run time (format: days-hours:minutes:seconds)
#SBATCH --gres=gpu:1  # Number of GPUs if needed
#SBATCH --cpus-per-task=8  # Number of CPUs (Don't use more than 24 per GPU)
#SBATCH --mem=25G  # Memory in GB (Don't use more than 126G per GPU)

# load python module

ml python/3.8.10
#ml python/anaconda3
ml cuda/11.7

#eval "$(conda shell.bash hook)"

source GAMDnew/bin/activate

#conda deactivate
#conda activate --stack myenv

# activate corresponding environment
#conda deactivate # If you launch your script from a terminal where your environment is already loaded, conda won't activate the environment. This guards against that. Not necessary if you always $#conda activate ptl

export TF_CPP_MIN_LOG_LEVEL=0

export WANDB_API_KEY=75b2f740e6817e165e204770a9185c4c98ba649d

#set up the environment
#pip install wheel
#pip install torch==1.9.1 torchvision==0.10.1
#pip install --upgrade "jax[cuda11_pip]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
#pip install jax-md --upgrade

#pip install dgl-cu111
#pip install cupy-cuda111
#pip install pytorch-lightning==1.3.0

#pip install openmm
#pip install scikit-learn

#pip install torchmetrics==v0.6.0

#pip install torchdyn==1.0.3


#./install_gamd.sh

# run the program
#export CUDA_LAUNCH_BLOCKING=1.
export PATH="/home/guests/lana_frkin/.local/bin:$PATH"
#python train_network_lj.py # --cuda
#python train_endtoend_autoencoder_nice.py
#python generate_graphs.py
#python test_script/visualize_graph.py
#python generate_lj_data.py
#python the_sequential_network.py
python entire_network_test.py
#python entire_network.py