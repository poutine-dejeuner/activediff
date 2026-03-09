# 1. Installer uv (plus rapide que pip/conda)
curl -LsSf https://astral.sh/uv/install.sh | sh
source $HOME/.cargo/env

# 2. Cloner ton repo
git clone https://github.com/TON_USER/reg_transfo.git
cd reg_transfo

# 3. Synchroniser l'environnement
# L'image GCP a déjà Python, on utilise l'env du système pour éviter les conflits de drivers
./install.sh

# 4. Vérifier que le GPU est bien vu par PyTorch
python -c "import torch; print(f'GPU Dispo: {torch.cuda.is_available()} - {torch.cuda.get_device_name(0)}')"
