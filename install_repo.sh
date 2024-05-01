python3 -m pip install --upgrade pip setuptools wheel
#python3 -m pip cache purge

python3 -m pip install torch

TORCH=$(python -c "import torch; print(torch.__version__)")
CUDA=$(python -c "import torch; print(torch.version.cuda)")

CUDA=${CUDA//.}

if [ $CUDA = None ]; then
    CUDA='cpu'
else
    CUDA=$(echo "cu${CUDA}")
fi

TORCH=$(echo "${TORCH}" | cut -f1 -d"+")
echo "CUDA: ${CUDA}"
echo "TORCH: ${TORCH}"

python3 -m pip install torch-geometric
python3 -m pip install --verbose --no-index torch-scatter torch-sparse torch-cluster -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html

python3 -m pip install -r requirements.txt
python3 -m pip install -e .