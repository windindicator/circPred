# circPred
Download pre-trained models from this gdrive link and place the pth files into the pretrained folder.
https://drive.google.com/drive/folders/1_JAAnswijYZYLEPUsbK-UJnmwvHTDrFr?usp=sharing


<details><summary>Citation</summary>

```bibtex
@article{chen2022interpretable,
  title={Interpretable rna foundation model from unannotated data for highly accurate rna structure and function predictions},
  author={Chen, Jiayang and Hu, Zhihang and Sun, Siqi and Tan, Qingxiong and Wang, Yixuan and Yu, Qinze and Zong, Licheng and Hong, Liang and Xiao, Jin and King, Irwin and others},
  journal={arXiv preprint arXiv:2204.00300},
  year={2022}
}
```
</details>

## Create Environment with Conda
If you want to use circPredictor, set up an RNA-FM environment. 
Please follow [RNA foundation model page](https://github.com/ml4bio/RNA-FM)

## Apply circPredictor with Existing Scripts.
```
python pred.py --data_path="./example.fasta" --task="scanning"
```

'--task' can be selected from identification, scanning, and mining.
Result file will be saved in the `result.txt`.


