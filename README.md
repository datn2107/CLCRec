# Data Notes :

## My data format

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

## Their data format
* Item ID - Start in range [n_user, n_user + n_item - 1]
* User ID - Start in range [0, n_user - 1]

* train_data: Is a list of tuples (User ID, Item ID) that represents the interactions between users and items.
* user_item_all_dict: Is a dictionary where the key is the User ID and the value is a list of Item IDs that represents the items that the user has interacted.

* val_data: Is a list of array, first element is user_id and the second to the end are the items that the user has interacted.
* val_warm_data: Similar to val_data, but the users are in the train set.
* val_cold_data: Similar to val_data, but the users are not in the train set.

* test_data, test_warm_data, test_cold_data: Similar to val_data, val_warm_data, val_cold_data, but for the test set.

* warm_set: Set of item ids that are in the train set.
* cold_set: Set of item ids that are not in the train set.

* result: The result is the matrix contains vectors feature of users and items. The first n_user rows are the users and the last n_item rows are the items. The columns are the features.
