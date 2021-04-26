# Simulation of Asynchronous Federated Learning

This is the repository for the work 

> Jin, Hongwei, Ning Yan, and Masood Mortazavi. "Simulating Aggregation Algorithms for Empirical Verification of Resilient and Adaptive Federated Learning." 2020 IEEE/ACM International Conference on Big Data Computing, Applications and Technologies (BDCAT). IEEE, 2020.

## Dataset generation

Data is preprocessed in the `/tmp/data` folder. More dataset will be preprocessed and store there.
Please check the folder `./data` for different dataset.
Three datasets are used: 
  * __mnist__: dataset MNIST from [OpenML](https://www.openml.org/d/554)
  * __nist__: dataset NIST Special Dataset from [NIST](https://www.nist.gov/srd/nist-special-database-19)
  * __shakespeare__: dataset Shakespeare plays original comes from [Kaggle](https://www.kaggle.com/kingburrito666/shakespeare-plays)

In each subfolder under `./data`, follow the instructions in `README.md` file to preprocess data. 
The preprocess data will be stored under `/tmp/data` folder. With the following file structure:
```s
/tmp/data
|- mnist
   |- test
   |- train
|- nist
   |- test
   |- train
|- shakespeare
   |- test
   |- train
```

## Federated Optimizer

* FedAvg: [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a/mcmahan17a.pdf), AISTATS 2017
* FedProx: [Federated Optimization for Heterogeneous Networks](https://arxiv.org/pdf/1812.06127.pdf), MLSys 2020

* FedMom: proposed method to aggregation the global model by taking its previous trained model.

## virtual environment

```s
conda create -n fedsimul python=3
conda activate fedsimul
pip3 install -r requirements.txt  
```

## Base case

* synchronous federated learning with all clients participated and refreshed
  ```s
    python main.py --optimizer fedavg -p 1 -q 1
  ```

* asynchronous FL with fixed participate and refresh rate
  ```s
    python main.py --asyn --optimizer fedavg 
  ```

* asynchronous FL with adaptive participate and adaptive refresh rate
* 
  ```s
    python main.py --asyn --adp_p --adp_q --optimizer fedavg 
  ```

* asynchronous FL with adaptive participate and adaptive refresh rate with specified parcipate rate and refresh rate
  ```s
    python main.py --asyn --adp_p --adp_q -p 0.02 -q 0.7 --optimizer fedavg
  ```

* asynchronous FL with specified window size
  ```s
    python main.py --asyn --optimizer fedavg -w 5
  ```

* For more detailed arguments, please use 
  ```s
    usage: main.py [-h] [--optimizer {fedavg,fedprox,fedmom,fedmomprox}]
               [--dataset {mnist,nist,shakespeare}] [--model MODEL]
               [--num_rounds NUM_ROUNDS] [--eval_every EVAL_EVERY]
               [--batch_size BATCH_SIZE] [--num_epochs NUM_EPOCHS]
               [--learning_rate LEARNING_RATE] [--gamma GAMMA] [--seed SEED]
               [--gpu_id GPU_ID] [--verbose] [--asyn]
               [--participate_rate PARTICIPATE_RATE]
               [--refresh_rate REFRESH_RATE] [--adp_p] [--adp_q]
               [--window_size WINDOW_SIZE] [--alpha ALPHA]

    optional arguments:
    -h, --help            show this help message and exit
    --optimizer {fedavg,fedprox,fedmom,fedmomprox}
                          name of optimizer;
    --dataset {mnist,nist,shakespeare}
                          name of dataset;
    --model MODEL         name of model;
    --num_rounds NUM_ROUNDS
                          number of rounds to simulate;
    --eval_every EVAL_EVERY
                          evaluate every ____ rounds;
    --batch_size BATCH_SIZE
                          batch size when clients train on data;
    --num_epochs NUM_EPOCHS
                          number of epochs when clients train on data;
    --learning_rate LEARNING_RATE
                          learning rate for inner solver;
    --gamma GAMMA         constant for momentum
    --seed SEED           seed for randomness;
    --gpu_id GPU_ID       gpu_id
    --verbose             toggle the verbose output
    --asyn, -a            toggle asynchronous simulation
    --participate_rate PARTICIPATE_RATE, -p PARTICIPATE_RATE
                          probability of participating
    --refresh_rate REFRESH_RATE, -q REFRESH_RATE
                          probability of refreshing
    --adp_p               toggle adaptive participate rate
    --adp_q               toggle adaptive refresh rate
    --window_size WINDOW_SIZE, -w WINDOW_SIZE
                          moving window size
  ```

## Example of running MNIST
Run the following instruction, replace `$DATASET` with a dataset of interest, specify the corresponding model to that dataset (choose from `fedsimul/models/$DATASET/$MODEL.py` and use `$MODEL` as the model name):

* with all the default arguments, one simple example is 

```s
python main.py
```

* example of running a asynchronous FL with adaptive participate rate and refresh rate.

```s
python main.py --optimizer fedavg --model mclr --dataset mnist \
    --asyn --participate_rate 0.3 --refresh_rate 0.7 \
    --eval_every=1 --batch_size=10 \
    --num_epochs=20 --num_rounds=200
```

## Setup using script

You can also setup the virtual environment by taking the setup.py script.
Simply run it as development mode:

```s
python setup.py develop
```

## Plot the results

After running the simulation, resuls will be generated under `./out/{DATASET}`. 
Each json file represent a different setting corresponding to the args in `main.py`.

Spepcify the files and metric want to be in the plot and the figure will be saved under `./out` folder.
```s
python plotfigure.py --help
usage: plotfigure.py [-h] [--metric {accuracy,flops}] [--outfile OUTFILE]
                     [--file FILE [FILE ...]] [--legend LEGEND [LEGEND ...]]

optional arguments:
  -h, --help            show this help message and exit
  --metric {accuracy,flops}
                        specify the metric
  --outfile OUTFILE     filename of saved png
  --file FILE [FILE ...]
                        files to plot
  --legend LEGEND [LEGEND ...]
                        legend of figure
```

## reference

This repository is based on the work of 

> fedprox [Federated Optimization for Heterogeneous Networks](https://arxiv.org/abs/1812.06127)
> 
> MLSys 2020

## License

The work is under [MIT License](./LICENSE)