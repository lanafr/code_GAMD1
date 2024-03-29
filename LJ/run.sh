#!/bin/bash -l
#SBATCH --job-name=gamdplus
#SBATCH --nodes=1
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --export=ALL
#SBATCH --output="slurm-%j.out"
#SBATCH --gres=gpu:a100:1
#SBATCH --mem=20G  # Memory in GB (Don't use more than 126G per GPU)

#erlangen

ml python/pytorch-1.10py3.9
export PATH="$PATH:/home/hpc/b118bb/b118bb12/.local/bin"

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
#python lets_see.py
#python test_script/visualize_graph.py
#python generate_lj_data.py
#python the_sequential_network1.py
#python seqn_just1particle.py
#python seqn_2_decoderincluded.py
#python entire_network_test.py
#python entire_network.py
#python train_AUTOENCODER.py
python train_ENTIRE_NETWORK.py --mode train --architecture graphlatentode
#python RNN_sequential_network.py
#python entire_network_RNN.py
#python TEST_entire_network.py --cp_name EVERYTHING_20s_just_node --epoch 30 --architecture graphlatentode
#nvidia-smi
#python TEST_autoencoder.py --epoch 39 --cp_name 'AUTOENCODER_50ts(cords)_NNConv_512'
#python change_me_pls.py