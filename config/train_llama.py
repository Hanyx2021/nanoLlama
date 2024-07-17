out_dir = 'out-shakespeare-char'
eval_interval = 250 # keep frequent because we'll overfit
eval_iters = 200
log_interval = 10 # don't print too too often

# we expect to overfit on this small dataset, so only save when val improves
always_save_checkpoint = False

wandb_log = False # override via command line if you like
wandb_project = 'shakespeare-char'
wandb_run_name = 'mini-llama'

dataset = 'shakespeare_char'

dim = 128
n_layers = 8
n_heads = 8
vocab_size = 32
multiple_of = 8
norm_eps = 1e-06

batch_size = 64
block_size = 8