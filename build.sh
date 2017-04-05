chmod 775 build.sh

pip3 install tensorflow
pip3 install numpy
pip3 install matplotlib

if [ ! -d "./Data" ]; then
    mkdir ./Data
fi

if [ ! -d "./Data/record" ]; then
    mkdir ./Data/record
fi

if [ ! -d "./Data/Weight" ]; then
    mkdir ./Data/record
fi

if [ ! -d "./Data/Bias" ]; then
    mkdir ./Data/record
fi
