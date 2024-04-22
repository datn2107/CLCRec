from tqdm import tqdm
import torch
import torch.nn as nn
from torch.autograd import no_grad
import numpy as np
from metric import rank, full_accuracy


def full_ranking(
    epoch,
    model,
    data,
    user_item_inter,
    mask_items,
    is_training,
    step,
    top_k,
    prefix,
    writer=None,
    need_score_matrix=False,
):
    print(prefix + " start...")
    model.eval()
    with no_grad():
        all_index_of_rank_list, all_score_matrix = rank(
            model.num_user,
            user_item_inter,
            mask_items,
            model.result,
            is_training,
            step,
            top_k,
        )
        precision, recall, ndcg_score = full_accuracy(
            data, all_index_of_rank_list, top_k
        )

        del all_index_of_rank_list
        if not need_score_matrix:
            del all_score_matrix

        print(
            "---------------------------------{0}-th Precition:{1:.4f} Recall:{2:.4f} NDCG:{3:.4f}---------------------------------".format(
                epoch, precision, recall, ndcg_score
            )
        )

        return precision, recall, ndcg_score, all_score_matrix.cpu().detach().numpy() if need_score_matrix else None
