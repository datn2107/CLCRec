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
    parser.add_argument("--result-filename", default=None, help="Filename for result")
    parser.add_argument("--model-filename", default=None, help="Filename for model")

    parser.add_argument(
        "--path-weight-load", default=None, help="Loading weight filename."
    )

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
    parser.add_argument("--early-stop", type=int, default=10, help="Early stop.")

    parser.add_argument("--dim-e", type=int, default=128, help="Embedding dimension.")
    parser.add_argument("--top-k", type=int, default=10, help="Workers number.")
    parser.add_argument("--step", type=int, default=2000, help="Evaluation Step.")

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

    learning_rate = args.lr
    lr_lambda = args.lr_lambda
    reg_weight = args.reg_weight
    batch_size = args.batch_size
    num_workers = args.num_workers
    num_epoch = args.num_epoch
    num_neg = args.num_neg
    num_sample = args.num_sample
    early_stop = args.early_stop
    top_k = args.top_k
    model_name = args.model_name
    temp_value = args.temp_value
    step = args.step

    has_v = args.has_v
    has_a = args.has_a
    has_t = args.has_t

    dim_e = args.dim_e
    writer = SummaryWriter()


    result_filename = (
        args.result_filename
        if args.result_filename is not None
        else "result_lr_{0}_lrl_{1}_reg_{2}_num_neg_{3}_tmp_{4}".format(learning_rate, lr_lambda, reg_weight, num_neg, temp_value)
    )
    model_filename = (
        args.model_filename
        if args.model_filename is not None
        else "model_lr_{0}_lrl_{1}_reg_{2}_num_neg_{3}_tmp_{4}".format(learning_rate, lr_lambda, reg_weight, num_neg, temp_value)
    )

    ##########################################################################################################################################
    if os.path.exists(data_path + "/result") is False:
        os.makedirs(data_path + "/result")

    with open(data_path + "/result/" + result_filename, "w") as save_file:
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

    dataset = preprocess_data(dataset)

    # all_data = np.concatenate(
    #     (dataset["train_data"], dataset["val_data"], dataset["test_data"]), axis=0
    # )
    # user_item_all_dict = convert_interactions_to_user_item_dict(all_data, n_users)
    # user_item_train_dict = convert_interactions_to_user_item_dict(
    #     dataset["train_data"], n_users
    # )

    all_data = np.concatenate(
        (
            dataset["train_all_warm_data"],
            dataset["val_cold_data"],
            dataset["test_cold_data"],
        ),
        axis=0,
    )
    user_item_all_dict = convert_interactions_to_user_item_dict(all_data, n_users)
    user_item_train_dict = convert_interactions_to_user_item_dict(
        dataset["train_all_warm_data"], n_users
    )

    for key in [
        "val_data",
        "val_warm_data",
        "val_cold_data",
        "test_data",
        "test_warm_data",
        "test_cold_data",
    ]:
        dataset[key + "_dict"] = convert_interactions_to_user_item_dict(
            dataset[key], n_users
        )

    train_dataset = TrainingDataset(
        n_users,
        n_items,
        user_item_all_dict,
        dataset["train_all_warm_data"],
        dataset["cold_items"],
        num_neg,
        device,
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
        device,
    ).to(device)

    if args.path_weight_load is not None:
        model.load_state_dict(torch.load(args.path_weight_load))

    ##########################################################################################################################################
    optimizer = torch.optim.Adam(
        [
            {
                "params": model.parameters(),
                "lr": learning_rate,
                "weight_decay": reg_weight,
            }
        ]
    )

    # ##########################################################################################################################################
    max_precision = 0.0
    max_recall = None
    max_NDCG = 0.0
    num_decreases = 0
    max_val_result = max_val_result_warm = max_val_result_cold = list()
    max_test_result = max_test_result_warm = max_test_result_cold = list()
    for epoch in range(num_epoch):
        print("Start training epoch {0}...".format(epoch))
        loss, mat = train_epoch(
            epoch,
            len(train_dataset),
            train_dataloader,
            model,
            optimizer,
            batch_size,
            writer,
            device,
        )

        if torch.isnan(loss):
            print(model.result)
            with open(data_path + "/result/" + result_filename, "a") as save_file:
                save_file.write(
                    "lr:{0} \t reg_weight:{1} is Nan\r\n".format(
                        learning_rate, reg_weight
                    )
                )
            break
        torch.cuda.empty_cache()

        train_precision, train_recall, train_ndcg = full_ranking(
            epoch,
            model,
            user_item_train_dict,
            user_item_train_dict,
            dataset["cold_items"],
            True,
            step,
            top_k,
            "train/",
            writer,
        )
        # val_result = full_ranking(
        #     epoch,
        #     model,
        #     dataset["val_data_dict"],
        #     user_item_train_dict,
        #     None,
        #     False,
        #     step,
        #     top_k,
        #     "val/",
        #     writer,
        # )

        # val_result_warm = full_ranking(
        #     epoch,
        #     model,
        #     dataset["val_warm_data_dict"],
        #     user_item_train_dict,
        #     dataset["cold_items"],
        #     False,
        #     step,
        #     top_k,
        #     "val/warm_",
        #     writer,
        # )

        val_result_cold = full_ranking(
            epoch,
            model,
            dataset["val_cold_data_dict"],
            user_item_train_dict,
            np.concatenate([dataset["warm_items"], dataset["test_cold_items"]]),
            False,
            step,
            top_k,
            "val/cold_",
            writer,
        )

        # test_result = full_ranking(
        #     epoch,
        #     model,
        #     dataset["test_data_dict"],
        #     user_item_train_dict,
        #     None,
        #     False,
        #     step,
        #     top_k,
        #     "test/",
        #     writer,
        # )

        # test_result_warm = full_ranking(
        #     epoch,
        #     model,
        #     dataset["test_warm_data_dict"],
        #     user_item_train_dict,
        #     dataset["cold_items"],
        #     False,
        #     step,
        #     top_k,
        #     "test/warm_",
        #     writer,
        # )

        test_result_cold = full_ranking(
            epoch,
            model,
            dataset["test_cold_data_dict"],
            user_item_train_dict,
            np.concatenate([dataset["warm_items"], dataset["val_cold_items"]]),
            False,
            step,
            top_k,
            "test/cold_",
            writer,
        )

        if max_recall is None or val_result_cold[1] > max_recall:
            pre_id_embedding = model.id_embedding
            max_recall = val_result_cold[1]
            # max_val_result = val_result
            # max_val_result_warm = val_result_warm
            max_val_result_cold = val_result_cold
            # max_test_result = test_result
            # max_test_result_warm = test_result_warm
            max_test_result_cold = test_result_cold
            num_decreases = 0

            torch.save(
                model.state_dict(),
                data_path + "/result/" + model_filename,
            )
        else:
            if num_decreases > early_stop:
                with open(
                    data_path + "/result/" + result_filename,
                    "a",
                ) as save_file:
                    save_file.write("----------------------Best Result----------------------")
                    save_file.write(str(args) + "\n")
                    save_file.write(
                        "\r\n-----------Val Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                            max_val_result_cold[0],
                            max_val_result_cold[1],
                            max_val_result_cold[2],
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

        with open(
            data_path + "/result/" + result_filename,
            "a",
        ) as save_file:
            save_file.write(str(args) + "\n")
            save_file.write("Epoch: {0}\n".format(epoch))
            save_file.write(
                "\r\n-----------Train Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                    train_precision, train_recall, train_ndcg
                )
            )
            save_file.write(
                "\r\n-----------Val Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                    val_result_cold[0],
                    val_result_cold[1],
                    val_result_cold[2],
                )
            )
            save_file.write(
                "\r\n-----------Test Cold Precition:{0:.4f} Recall:{1:.4f} NDCG:{2:.4f}-----------".format(
                    test_result_cold[0],
                    test_result_cold[1],
                    test_result_cold[2],
                )
            )
