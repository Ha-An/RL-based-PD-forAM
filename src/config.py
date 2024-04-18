import os


N_EPISODES = 10 # Training iterations, Set to at least 2048 times
N_EVAL_EPISODES = 1 # Testing iterations
MAX_N_PARTS = 5 # Minimum number of pieces to stop performing

TRAIN = True
EXTENDED = True

# Set low and high values for the actions of the center coordination (X, Y, Z)
ACTION_SPACE_CENTER_COOR_LOW = 0  # unit: mm
ACTION_SPACE_CENTER_COOR_HIGH = 200  # unit: mm

# Set low and high values for the actions of the cutting plane angle (X, Y, Z)
ACTION_SPACE_CUT_PLANE_ANGLE_LOW = 0  # unit: degree
ACTION_SPACE_CUT_PLANE_ANGLE_HIGH = 180  # unit: degree

# Set low and high values for the observation space


# Set the initial model and import&export directory names
INPUT_MODEL = 'StanfordBunny.stl'
IMPORT_DIR = 'models'
EXPORT_DIR = 'results'
### LOG_DIR = 'logs' ###

# Import and export directories for STL files
CURRENT_DIR = os.path.abspath(os.path.dirname(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)
MESH_PATH = os.path.join(PARENT_DIR, IMPORT_DIR, INPUT_MODEL)
EXPORT_DIR = os.path.join(PARENT_DIR, EXPORT_DIR)

result_folder = os.path.join('../results', 'experiment1')

RESULTS_DIR = "../results"
LOGS_DIR = "../logs"

COST_REMOVE_SUP = 1
COST_ASSEMBLE = 1
# TensorBoard 실행:
# tensorboard --logdir="C:/tensorboard_logs/"
# http://localhost:6006/
