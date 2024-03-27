This project uses the [Pytorch Lightning framework](https://lightning.ai/docs/pytorch/stable/starter/introduction.html).

## Training and validation
The training and validation code are defined in the `training_step` and `validation_step` methods of the `UNetR_Lightning` class.

## Submission on Tron
To change the number of node and GPUs to request, modify these slurm arguments in `submit_lightning.sh`:

```
#SBATCH --nodes=1
#SBATCH --gres=gpu:rtxa6000:2 
#SBATCH --ntasks-per-node=2  
```

The number of tasks per node must match the number GPUs. Then pass `--devices 2` to `python3 train_lightning.py`. The number of devices must match the number of GPUs requested.

## Debugging
You can use the `fast_dev_run` option of the `Trainer` class to make debugging faster. 
Change:
```
trainer = L.Trainer(min_epochs=50, max_epochs=args.epochs, accelerator="cuda", devices=args.devices, strategy="fsdp") 
```
to 
```
trainer = L.Trainer(fast_dev_run=1, accelerator="cuda", devices=args.devices, strategy="fsdp")
```
will run 1 batch of training and 1 batch of validtion. 

## Data and Transforms
The data loading process is written in `data/sevenTdata.py` and the transforms are in `transform_easy.py`. 
