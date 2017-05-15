chmod 775 build.sh

echo "[BUILD] Installing tensorflow "
pip3 install tensorflow --quiet
echo "[BUILD] Tensorflow installation complete"
echo "[BUILD] Installing matplotlib"
pip3 install matplotlib --quiet
echo "[BUILD] Matplotlib installation complete"

echo "[BUILD] Making ./Data"
if [ ! -d "./Data" ]; then
    mkdir ./Data
fi

echo "[BUILD] Making ./Data/record"
if [ ! -d "./Data/record" ]; then
    mkdir ./Data/record
    mkdir ./Data/record/selfrecord
    mkdir ./Data/record/test_record
    mkdir ./Data/record/self_play
fi

