class model_config:
    D = 30
    time_seq = 20
    h = 3 # horizon, 3, 6, 12 ,24
    batch_size = 32
    learning_rate = 1e-2

    dropout_rate = 0.1

    n_g = 32  # global_temporal_filters, 3, 5, 7    
    n_l = 32  # local_temporal_filters, 3, 5, 7
    l = 3   # length
    horozon = 3 # 3, 6, 12, 24

    n_heads = 8
    attention_stacks = 5
