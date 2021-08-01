import os, argparse, time

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    elif v.isdigit():
        return int(v)
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

DATANAME = ['sd2','sd2e1','sd2e3','sd2t5','sd5','sd5e1','sd5e3','sd5t5','sd', 'sd10', 'sd15']
""" 
Dataset 1, 3: sd2 (args.did=0) with args.F=10, args.F=7
Dataset 2, 4: sd2t5 (args.did=3) with args.F=10, args.F=7
Dataset 5, 7: sd5 (args.did=4) with args.F=10, args.F=7
Dataset 6, 8: sd5t5 (args.did=7) with args.F=10, args.F=7
The others are experimental datasets.
Dataset is determined by # of machines and jobs (a.k.a. "scales"), # of families, and due-date tightness in <simul_pms.py>
Refer the papers for details: https://ieeexplore.ieee.org/document/9486959
"""
# checking arguments
def check_args(args):
    # --checkpoint_dir
    # for i in range(0, len(DatasetList)):

    folder_name = '{}_{}_{}'.format(DATANAME[args.did] if args.F==10 else DATANAME[args.did]+'f'+str(args.F), args.oopt, args.key)
    args.save_dir = args.root_dir + folder_name
    args.gantt_dir = os.path.join(args.save_dir, 'gantt')
    args.model_dir = os.path.join(args.save_dir, 'models')
    args.best_model_dir = os.path.join(args.save_dir, 'best_models')
    args.summary_dir = os.path.join(args.save_dir, 'summary')
    args.timestamp = time.strftime("%Y%m%d_%H%M%S", time.gmtime())
    if type(args.auxin_dim)==list and len(args.auxin_dim)==1: args.auxin_dim = args.auxin_dim[0]
    if 'a3c' in args.key:
        args.state_dim = args.state_dim[0]
    if 'upm' in args.oopt:
        args.action_dim = 4
        args.hid_dims = []
        args.bucket = 0
        args.auxin_dim = 0
        args.batchsize = 1
        args.warmup = 0
        if args.did == -1:
            args.max_episode = 40000
            args.save_freq = 400
            args.state_dim = [450]
        elif args.did < 4:
            args.max_episode = 20000
            args.save_freq = 200
            args.state_dim = [900]
        else:
            args.max_episode = 2500
            args.save_freq = 25
            args.state_dim = [2250]
        if args.oopt =='upm2012':
            args.state_dim = [110]
        elif args.oopt=='upm2007':
            args.GAMMA = 0.002
            args.lr = 0.001
    elif 'fab' in args.oopt:
        args.action_dim = args.F
        args.auxin_dim = 0
        args.hid_dims = [512, 128, 21]
        args.bucket = 0
        if args.did == -1:
            args.state_dim = [210*3 + args.F*(221)]
            args.max_episode = 2000
            args.save_freq = 20
        elif args.did < 4:
            args.state_dim = [420*3 + args.F*(441)]
            args.max_episode = 800
            args.save_freq = 8
        else:
            args.state_dim = [1050 * 3 + args.F * (1101)]
            args.max_episode = 50
            args.save_freq = 1

        args.ropt = 'tardiness'
        args.GAMMA = 0.9
        args.warmup = 32
        args.batchsize = 32
        args.lr= 0.000001
        args.freq_tar = 0
        args.eps_ratio = 0
    elif args.oopt == 'ours2007':
        args.action_dim = args.F*args.F
        args.auxin_dim = 0
        if args.did == -1: args.state_dim = [450]
        elif args.did < 4: args.state_dim = [900]
        else: args.state_dim = [2250]
    else:
        args.action_dim = args.F*args.F
        if args.state_type == '1D':
            args.state_dim = [args.F * (args.F*2+42) + 2]
        elif args.state_type == 'manual':
            pass
        else:
            args.state_dim = [args.F, args.F*2+42] # default
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if not os.path.exists(args.gantt_dir):
        os.mkdir(args.gantt_dir)
    if not os.path.exists(args.model_dir):
        os.mkdir(args.model_dir)
    if not os.path.exists(args.best_model_dir):
        os.mkdir(args.best_model_dir)
    if not os.path.exists(args.summary_dir):
        os.mkdir(args.summary_dir)
    # --batch_size
    assert args.batchsize >= 1, 'batch size must be larger than or equal to one'
    return args

desc = "PMS experiment"
parser = argparse.ArgumentParser(description=desc)
parser.add_argument('--env', type=str, default='pms')
parser.add_argument('--result_dir', type=str, default='results',
                    help='Directory name to save the generated images')
parser.add_argument('--key', type=str, default='default') # key for identifying experiments, results file
parser.add_argument('--did', type=int, default=0)
parser.add_argument('--eid', type=int, default=0)
parser.add_argument('--test_mode', type=str, default='logic')
parser.add_argument('--root_dir', type=str, default='./results/')
parser.add_argument('--save_freq', type=int, default=5000)
parser.add_argument('--viz', type=str2bool, default=False)
parser.add_argument('--is_load', type=str2bool, default=False)
parser.add_argument('--use_hist', type=str2bool, default=False)

# For Reinforcement Learning
parser.add_argument('--lr', type=float, default=0.0025) # learning rate
parser.add_argument('--nn', type=str, default='keep')
parser.add_argument('--is_duel', type=str2bool, default=False) # Dueling DQN option can be used
parser.add_argument('--is_noisy', type=str2bool, default=False)
parser.add_argument('--is_double', type=str2bool, default=False)
parser.add_argument('--is_train', type=str2bool, default=True)
parser.add_argument('--is_first', type=str2bool, default=True)
parser.add_argument('--eps', type=float, default=0.2) # Initial epsilon value
parser.add_argument('--eps_ratio', type=float, default=0.9) # portion of exploration episodes
parser.add_argument('--warmup', type=int, default=24000) # number of time stpes for random exploration
parser.add_argument('--GAMMA', type=float, default=1)
parser.add_argument('--batchsize', type=int, default=32)
parser.add_argument('--freq_tar', type=int, default=50) # Target network update frequency (unit: episodes)
parser.add_argument('--freq_on', type=int, default=1) # Target network update frequency (unit: time steps)
parser.add_argument('--max_episode', type=int, default=100000)
parser.add_argument('--max_buffersize', type=int, default=100000)

parser.add_argument('--policy', type=str, default='dqn_logic')
parser.add_argument('--sampling', type=str, default='td')
parser.add_argument('--F', type=int, default=10)
parser.add_argument('--action_dim', type=int, default=10) # action dim is automatically set to args.F * args.F
parser.add_argument('--auxin_dim', type=int, default=[0], nargs='+') # auxin is S_inv in the paper
parser.add_argument('--state_dim', '-s', type=int, default=[40], nargs='+') # -s 10 52 means 10 X 52 2-D state
parser.add_argument('--state_type', type=str, default='2D') # use args.state_type=manual for changing args.state_dim
parser.add_argument('--share_num', type=int, default=0)
parser.add_argument('--chg_freq', type=int, default=1)
parser.add_argument('--hid_dims', type=int, default=[64, 32, 16], nargs='+',
                    help='hidden dimensions (default: [64, 32])')

parser.add_argument('--oopt', type=str, default='default') # You can modify oopt for other baseline models
parser.add_argument('--sopt', type=str, default='default') # For advanced users
parser.add_argument('--use', type=int, default=[1,2,4,5,6,7], nargs='+') # selectively choose state features
parser.add_argument('--K', type=int, default=0) # state concatenation (obsolute)
parser.add_argument('--ropt', type=str, default='epochtard') # Option for reward generation.

parser.add_argument('--change_qty', type=str2bool, default=False)
parser.add_argument('--qty', type=str, default=None)
parser.add_argument('--cutday', type=int, default=-1)
parser.add_argument('--repeat', type=int, default=1)
parser.add_argument('--config_load', type=str, default=None)
parser.add_argument('--bucket', type=int, default=0) # Time intervals of period (T in the paper)
parser.add_argument('--equality_flag', type=bool, default=False)

args = check_args(parser.parse_args())
args.DATASET = DATANAME

