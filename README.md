# Data Notes :

* metadata (`metadata.npy`): Is a dictionary with the following keys:
    * n_users: Number of users
    * n_items: Number of items
    * n_features: Number of features
    * n_warm_items: Number of warm items
    * n_cold_items: Number of cold items
    * n_val_cold_items: Number of cold items for the validation set,
    * n_test_cold_items: Number of cold items for the test set,
    * n_train_interactions: Number of interactions in the train set
    * n_val_interactions: Number of interactions in the validation set
    * n_test_interactions: Number of interactions in the test set
    * n_val_warm_interactions: Number of interactions in the validation set with warm items
    * n_val_cold_interactions: Number of interactions in the validation set with cold items
    * n_test_warm_interactions: Number of interactions in the test set with warm items
    * n_test_cold_interactions: Number of interactions in the test set with cold items

* train_data (`train_interactions.npy`): Is a list of tuples (User ID, Item ID) that represents the interactions between users and items.

* val_data (`val_interactions.npy`): Is a list of tuples (User ID, Item ID) that represents the interactions between users and items. Contains both warm and cold items.
* val_warm_data (`val_warm_interactions.npy`): Is similar to val_data, but contains only warm items.
* val_cold_data (`val_cold_interactions.npy`): Is similar to val_data, but contains only cold items.

* test_data (`test_interactions.npy`): Is similar to val_data, but contains only warm items.
* test_warm_data (`test_warm_interactions.npy`): Is similar to val_data, but contains only warm items.
* test_cold_data (`test_cold_interactions.npy`): Is similar to val_data, but contains only cold items.

* warm_set (`warm_items.npy`): Set of item ids that are in the train set.
* cold_set (`cold_items.npy`): Set of item ids that are not in the train set.

* t_features (`t_features.npy`): The matrix contains vectors text feature of items. The rows are the items and the columns are the features. (Optional)
* a_features (`a_features.npy`): The matrix contains vectors audio feature of users. The rows are the users and the columns are the features. (Optional)
* v_features (`v_features.npy`): The matrix contains vectors visual feature of items. The rows are the items and the columns are the features. (Optional)

# Example to Run the Codes
The instruction of commands has been clearly stated in the codes.
- Movielens dataset
`python main.py --model_name='CLCRec' --l_r=0.001 --reg_weight=0.1 --num_workers=4 --num_neg=128 --has_a=True --has_t=True --has_v=True --lr_lambda=0.5 --temp_value=2.0 --num_sample=0.5`
- Amazon dataset
`python main.py --model_name='CLCRec' --data_path=amazon --l_r=0.001 --reg_weight=0.001 --num_workers=4 --num_neg=512 --has_v=True --lr_lambda=0.9  --num_sample=0.5`

Some important arguments:

- `lr_lambda` It specifics the value of lambda to balance the U-I and R-E mutual information.

- `num_neg` This parameter indicates the number of negative sampling.

- `num_sample` This parameter indicates the probability of hybrid contrastive training.

- `temp_value` It specifics the temprature value in density ratio functions.
