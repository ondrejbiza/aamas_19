F_BAR_NETWORK_NAMESPACE = "f_bar"
PROP_NETWORK_TEMPLATE = "prop{:d}"

GAMMA_NETWORK_NAMESPACE = "gamma"

OBJECT_PUCK = "disk"
OBJECT_BLOCK = "block"
OBJECTS = [OBJECT_PUCK, OBJECT_BLOCK]

OPENRAVE_BLOCK_HEIGHT = 0.499
OPENRAVE_DISK_HEIGHT = 0.245

OPT_ADAM = "adam"
OPT_SGD = "sgd"
OPT_MOMENTUM = "momentum"

REWARD_CONFIDENCE = "confidence"
REWARD_NUM_SAMPLES = "samples"

EASY = "easy"
MEDIUM = "medium"
HARD = "hard"

DECONV_BEFORE = "deconv_before"
DECONV_AFTER = "deconv_after"
UPSAMPLE_BEFORE = "upsample_before"
UPSAMPLE_AFTER = "upsample_after"

GOAL_STATE = "goal"

DEFAULT_BLOCK_SIZE = 28

TRAIN_STEP = "train_step"
TOTAL_LOSS = "total_loss"
TRANSITION_LOSS = "transition_loss"
REWARD_LOSS = "reward_loss"
REWARD_SOFTMAX_GRADS_NORM = "reward_softmax_grads_norm"
REWARD_LOGITS_GRADS_NORM = "reward_logits_grads_norm"
REWARD_MATRIX_GRAD_NORM = "reward_matrix_grad_norm"
TRANSITION_SOFTMAX_GRADS_NORM = "transition_softmax_grads_norm"
TRANSITION_LOGITS_GRADS_NORM = "transition_logits_grads_norm"
TRANSITION_MATRIX_NORMALIZED_GRAD_NORM = "transition_matrix_normalized_grad_norm"
TRANSITION_MATRIX_RAW_GRAD_NORM = "transition_matrix_raw_grad_norm"
PERPLEXITY = "perplexity"
STATE_PERPLEXITY = "state_perplexity"
ACTION_PERPLEXITY = "action_perplexity"
SUMMARY = "summary"
LOGIT_NORMS = "logit_norms"

STEP_ONE_REWARD_LOSS = "step_one_reward_loss"
STEP_TWO_REWARD_LOSS = "step_two_reward_loss"
