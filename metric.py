import torch
import math
import torch.nn.functional as F

MIN_VALUE = 1e-15

def rank(num_user, user_item_inter, mask_items, result, is_training, step, topk):
    user_tensor = result[:num_user]
    item_tensor = result[num_user:]

    start_index = 0
    end_index = num_user if step == None else min(step, num_user)

    all_score_matrix = torch.FloatTensor([]).to(user_tensor.device)
    all_index_of_rank_list = torch.LongTensor([])
    while end_index <= num_user and start_index < end_index:
        temp_user_tensor = user_tensor[start_index:end_index]
        score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

        if is_training is False:
            for row, col in user_item_inter.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col)) - num_user
                    score_matrix[row][col] = MIN_VALUE
        if mask_items is not None:
            score_matrix[:, mask_items - num_user] = MIN_VALUE

        all_score_matrix = torch.cat((all_score_matrix, score_matrix), dim=0)

        _, index_of_rank_list = torch.topk(score_matrix, topk)
        all_index_of_rank_list = torch.cat(
            (all_index_of_rank_list, index_of_rank_list.cpu() + num_user), dim=0
        )

        start_index = end_index
        if end_index + step < num_user:
            end_index += step
        else:
            end_index = num_user

    return all_index_of_rank_list, all_score_matrix


def full_accuracy(data, all_index_of_rank_list, topk):
    length = 0
    precision = recall = ndcg = 0.0

    for row, col in data.items():
        user = row
        pos_items = set(col)
        num_pos = len(pos_items)
        if num_pos == 0:
            continue

        length += 1
        items_list = all_index_of_rank_list[user].tolist()
        items = set(items_list)

        num_hit = len(pos_items.intersection(items))
        precision += float(num_hit / topk)
        recall += float(num_hit / num_pos)

        ndcg_score = 0.0
        max_ndcg_score = 0.0
        for i in range(min(num_hit, topk)):
            max_ndcg_score += 1 / math.log2(i + 2)
        if max_ndcg_score == 0:
            continue

        for i, temp_item in enumerate(items_list):
            if temp_item in pos_items:
                ndcg_score += 1 / math.log2(i + 2)
        ndcg += ndcg_score / max_ndcg_score

    return precision / length, recall / length, ndcg / length
