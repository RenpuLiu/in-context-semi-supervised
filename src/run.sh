UNLAB_LIST = (0 1 2 3 4 5 6 7 8 9 10)

for M in "${UNLAB_LIST[@]}"; do
  python train.py --config conf/semi_supervised.yaml
done
