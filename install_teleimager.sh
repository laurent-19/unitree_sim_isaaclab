# Install teleimager
cd /home/code/unitree_sim_isaaclab
git config --global --add safe.directory /home/code/unitree_sim_isaaclab

git submodule update --init --recursive

cd teleimager
sed -i 's|requires-python = ">=3.8,<3.11"|requires-python = ">=3.8,<3.12"|' pyproject.toml

pip install -e .
pip install aiortc aiohttp

cd /home/code/unitree_sim_isaaclab 
