python main.py --path IBRH13_num --model MLP --batch 64 --lr 0.001
python main.py --path IBRH13_num --model FC --batch 64 --lr 0.001
python main.py --path IBRH13_num --model LSTM --batch 64 --lr 0.001 --hiddensize 32 --numlayers 3
python main.py --path IBRH13_num --model RNN --batch 64 --lr 0.001 --hiddensize 32 --numlayers 3
python main.py --path IBRH13_num --model CNN --batch 64 --lr 0.001 --kernel 101 --numlayers 3
pause