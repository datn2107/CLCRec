import argparse
import os
import time
import numpy as np
import torch
import random
from data import (
    TrainingDataset,
    load_dataset,
    preprocess_data,
    convert_interactions_to_user_item_dict,
)
from model_CLCRec import CLCRec
from torch.utils.data import DataLoader
from train import train_epoch
from full_rank import full_ranking
from torch.utils.tensorboard import SummaryWriter

###############################248###########################################


def init():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=1, help="Seed init.")
    parser.add_argument(
        "--no-cuda", action="store_true", default=False, help="Disables CUDA training."
    )
    parser.add_argument("--data-path", default="movielens", help="Dataset path")
    parser.add_argument("--save-file", default="", help="Filename")

    parser.add_argument(
        "--path-weight-load", default=None, help="Loading weight filename."
    )
    parser.add_argument(
        "--path-weight-save", default=None, help="Writing weight filename."
    )
    parser.add_argument("--prefix", default="", help="Prefix of save_file.")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate.")
    parser.add_argument("--lr-lambda", type=float, default=1, help="Weight loss one.")
    parser.add_argument("--reg-weight", type=float, default=1e-1, help="Weight decay.")
    parser.add_argument(
        "--temp-value", type=float, default=1, help="Contrastive temp_value."
    )
    parser.add_argument("--model-name", default="SSL", help="Model Name.")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size.")
    parser.add_argument("--num-neg", type=int, default=512, help="Negative size.")
    parser.add_argument("--num-epoch", type=int, default=1000, help="Epoch number.")
    parser.add_argument("--num-workers", type=int, default=1, help="Workers number.")
    parser.add_argument("--num-sample", type=float, default=0.5, help="Workers number.")

    parser.add_argument("--dim-e", type=int, default=64, help="Embedding dimension.")
    parser.add_argument("--top-k", type=int, default=10, help="Workers number.")
    parser.add_argument("--step", type=int, default=2000, help="Workers number.")

    parser.add_argument(
        "--has-v", default=False, action="store_true", help="Has Visual Features."
    )
    parser.add_argument(
        "--has-a", default=False, action="store_true", help="Has Acoustic Features."
    )
    parser.add_argument(
        "--has-t", default=False, action="store_true", help="Has Textual Features."
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = init()

    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"

    ##########################################################################################################################################
    data_path = args.data_path
    print(data_path)
    save_file_name = args.save_file

    learning_rate = args.lr
    lr_lambda = args.lr_lambda
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_neg = args.num_neg
    num_sample = args.num_sample
    top_k = args.top_k
    prefix = args.prefix
    model_name = args.model_name
    temp_value = args.temp_value
    step = args.step

    has_v = args.has_v
    has_a = args.has_a
    has_t = args.has_t

    dim_e = args.dim_e
    writer = SummaryWriter()

    ##########################################################################################################################################
    if os.path.exists(data_path + "/result") is False:
        os.makedirs(data_path + "/result")
    with open(
        data_path + "/result/result{0}_{1}.txt".format(learning_rate, reg_weight), "w"
    ) as save_file:
        save_file.write(
            "---------------------------------lr: {0} \t reg_weight:{1} ---------------------------------\r\n".format(
                learning_rate, reg_weight
            )
        )

    ##########################################################################################################################################
    print("Data loading ...")

    dataset = load_dataset(data_path, has_v, has_a, has_t, device=device)

    n_users = dataset["n_users"]
    n_items = dataset["n_items"]

    all_data = np.concatenate((dataset["train_data"], dataset["val_data"], dataset["test_data"]), axis=0)
    user_item_all_dict = convert_interactions_to_user_item_dict(
        all_data, n_users
    )
    user_item_train_dict = convert_interactions_to_user_item_dict(
        dataset["train_data"], n_users
    )

    dataset = preprocess_data(dataset)

    train_dataset = TrainingDataset(
        n_users,
        n_items,
        user_item_all_dict,
        dataset["train_data"],
        dataset["cold_items"],
        num_neg,
        device
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size, shuffle=True, num_workers=num_workers
    )

    print("Data has been loaded.")
    ##########################################################################################################################################
    model = CLCRec(
        n_users,
        n_items,
        dataset["n_warm_items"],
        reg_weight,
        dim_e,
        dataset["v_feat"],
        dataset["a_feat"],
        dataset["t_feat"],
        temp_value,
        num_neg,
        lr_lambda,
        num_sample,
        device
    ).to(device)

    if args.path_weight_load is not None:
        model.load_state_dict(torch.load(args.path_weight_load))

    ##########################################################################################################################################
    optimizer = torch.optim.Adam(
        [{"params": model.parameters(), "lr": learning_rate}]
    )  # , 'weight_decay': reg_weight}])

    # ##########################################################################################################################################
    max_precision = 0.0
    max_recall = 0.0
    max_NDCG = 0.0
    num_decreases = 0
    max_val_result = max_val_result_warm = max_val_result_cold = list()
    max_test_result = max_test_result_warm = max_test_result_cold = list()
    for epoch in range(num_epoch):
        loss, mat = train_epoch(
            epoch,
            len(train_dataset),
            train_dataloader,
            model,
            optimizer,
            batch_size,
            writer,
            device
        )

        if torch.isnan(loss):
            print(model.result)
            with open(
                data_path + "/result/result_{0}.txt".format(save_file_name), "a"
            ) as save_file:
                save_file.write(
                    "lr:{0} \t reg_weight:{1} is Nan\r\n".format(
                        learning_rate, reg_weight
                    )
                )
            break
        torch.cuda.empty_cache()

        # train_precision, train_recall, train_ndcg = full_ranking(epoch, model, user_item_inter, user_item_inter, True, step, topK, 'Train', writer)
        val_result = full_ranking(
            epoch,
            model,
            dataset["val_data"],
            user_item_train_dict,
            None,
            False,
            step,
            top_k,
            "val/",
            writer,
        )

        val_result_warm = full_ranking(
            epoch,
            model,
            dataset["val_warm_data"],
            user_item_train_dict,
            dataset["cold_items"],
            False,
            step,
            top_k,
            "val/warm_",
            writer,
        )

        val_result_cold = full_ranking(
            epoch,
            model,
            dataset["val_cold_data"],
            user_item_train_dict,
            dataset["warm_items"],
            False,
            step,
            top_k,
            "val/cold_",
            writer,
        )

        test_result = full_ranking(
            epoch,
            model,
            dataset["test_data"],
            user_item_train_dict,
            None,
            False,
            step,
            top_k,
            "test/",
            writer,
        )

        test_result_warm = full_ranking(
            epoch,
            model,
            dataset["test_warm_data"],
            user_item_train_dict,
            dataset["cold_items"],
            False,
            step,
            top_k,
            "test/warm_",
            writer,
        )

        test_result_cold = full_ranking(
            epoch,
            model,
            dataset["test_cold_data"],
            user_item_train_dict,
            dataset["warm_items"],
            False,
            step,
            top_k,
            "test/cold_",
            writer,
        )

        if val_result[1] > max_recall:
            pre_id_embedding = model.id_embedding
            max_recall = val_result[1]
            max_val_result = val_result
            max_val_result_warm = val_result_warm
            max_val_result_cold = val_result_cold
            max_test_result = test_result
            max_test_result_warm = test_result_warm
            max_test_result_cold = test_result_cold
            num_decreases = 0

            if args.path_weight_save is not None:
                torch.save(
                    model.state_dict(),
                    data_path + "result/model_{0}.pth".format(save_file_name),
                )
        else:
            if num_decreases > 5:
                with open(
                    data_path + "result/result_{0}.txt".format(save_file_name),
                    "a",
                ) as save_file:
                    save_file.write(str(args))
                    save_file.write(
                        "\r\n-----------Val Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_val_result[0], max_val_result[1], max_val_result[2]
                        )
                    )
                    save_file.write(
                        "\r\n-----------Val Warm Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_val_result_warm[0],
                            max_val_result_warm[1],
                            max_val_result_warm[2],
                        )
                    )
                    save_file.write(
                        "\r\n-----------Val Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_val_result_cold[0],
                            max_val_result_cold[1],
                            max_val_result_cold[2],
                        )
                    )
                    save_file.write(
                        "\r\n-----------Test Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_test_result[0], max_test_result[1], max_test_result[2]
                        )
                    )
                    save_file.write(
                        "\r\n-----------Test Warm Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_test_result_warm[0],
                            max_test_result_warm[1],
                            max_test_result_warm[2],
                        )
                    )
                    save_file.write(
                        "\r\n-----------Test Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_test_result_cold[0],
                            max_test_result_cold[1],
                            max_test_result_cold[2],
                        )
                    )
                break
            else:
                num_decreases += 1
