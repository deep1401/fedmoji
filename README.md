# fedmoji

Repository for the EMNLP 2022 paper ["A Federated Approach to Predicting Emojis in Hindi Tweets"](https://arxiv.org/abs/2211.06401) by Gandhi and Mehta et al. (2022)

## Dataset

The dataset of the code is available on Zenodo at [this link](https://zenodo.org/record/5559434).
The files have a restricted access and maybe available if the following conditions are met:

- Data should be used for non-commercial research use.
- The data should not be shared outside of the research team.
- The data should not be used for user profiling.
- Requesters should have received institutional approval (e.g. the requesters have been granted IRB approval for their project).
- A research statement that details potential harms.

## Installation

Instructions to install everything before running the code.
This also includes adding a key for the [Weights & Biases API](https://wandb.ai/) being used to track experiments

```shell
pip install -r requirements.txt
wandb login <API KEY>
```

## Running Experiments

Below is an example for running the modified causalfedgsd algorithm for a client fraction of 10% and 20 local epochs. This example is assumed to be running on the 1st GPU device in a cluster and the data is assumed to be stored in a folded named `data`. The batch size is taken to be 128 and client distribution is taken to be iid. We are also including the wandb logging flags as well as locally saving the results using `--save` argument.

```shell
python main.py --data data \
                --gpu 0 \
                --rounds 100 \
                --class_weights \
                --C 0.1 \
                --E 20 \
                --batch_size 128 \
                --shared  \
                --save lstm-cls_wt-causalfedgsd-c01-e20-k100-r100 \
                --wandb  \
                --wandb_run_name LSTM_token_cls-wt_causalfedgsd_c0.1_e20_k100_r100 \

```

In order to run the same experiment under a non-iid setting, a `--non_iid` just needs to be added to this example. The `--shared` argument can be replaced with `--causalfedgsd` to run the CausalFedGSD algorithm, and none of the arguments (shared or causalfedgsd) should be included to just use the original FedProx algorithm. The `--class_weights` argument can be removed to run the same experiments using just the imbalanced dataset. The `--class_weights` argument can be replaced with `--balanced` to run the same experiments using the balanced dataset. The `--wandb` argument can be removed to not log the results to Weights & Biases.

Other values of hyperparameters can be tuned by setting the arguments `--lr` for learning rate, `--mu` for the proximal term in FedProx experiments and `--seed` for setting the same random seed across all setups. The `--K` argument can be used to set the number of clients.

Note: The models can be trained centrally by setting the `--C` parameter to 1.0 and `--K` to 1.

## Citation

```
@inproceedings{fedmoji2022
    title = "A Federated Approach to Predicting Emojis in Hindi Tweets",
    author = "Gandhi, Deep  and
      Mehta, Jash  and
      Parekh, Nirali  and
      Waghela, Karan and
      D'Mello, Lynette and
      Talat, Zeerak",
    booktitle = "Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2022",
    publisher = "Association for Computational Linguistics",
}
```
