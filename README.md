Based off this work: https://github.com/yukimasano/self-label

Use `video_extract.ipynb` to detect frames with objects and extract time.

Then use `data_preprocess.ipynb` to crop and resize the images, multiple images to inflate the dataset

Then use `main_custom.py` to run my version of the code

```python main_custom.py --epochs 500 --batch-size 64 --lr 0.08 --lrdrop 150 --wd -5 --dtype f64 --nopts 100 --augs 2 --lamb 10 --arch resnetv2 --archspec small --ncl 57 --hc 1 --device 0 --modeldevice 0 --exp self-label-uno-all --workers 2 --dataset-path /home/tho121/selflabel/self-label/data/uno_all --comment "self-label-uno-all" --log-intv 1 --log-iter 10```
