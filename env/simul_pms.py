import random, pickle, copy

from env.job_generator import *
from utils.core.timeline import *
from utils.core.Job import Job
from env.wrapper import Wrapper, Location
from utils.visualize.logger import instance_log, log_decision
import env.util_sim as util

SAVE_FLAG = False
RESERVE_ON = False
if 'target' in args.ropt: ABANDON_ON = True
else: ABANDON_ON = False

""" Change value 'True' for reproducing stochastic processing and setup time ( Table 5 in paper ) """
STOCHASTICITY = False

class PMSSim(object):

    def __init__(self, config_load=None, log : instance_log =None,
                 data_name='sd2', opt_inist=['w4'], opt_inirem='random', opt_mix='geomsort'):
        """
        :param config_load: filename of saved configuration (production plan)
        :param log: instance_log object for visualization, logging
        :param data_name: dataset name
        :param opt_inist: option for generating initial setup status of machines
        :param opt_inirem: option for generating initial remaining processing time of machines
        :param opt_mix: option for distribution of production requirements
        'geomsort' option is used in the paper
        """


        """Dataset configuration"""
        self.data_name = data_name
        self.opt_inist = opt_inist
        self.opt_inirem = opt_inirem
        self.opt_mix = opt_mix

        """Setting scales of machine configurations"""
        if data_name == 'sd':
            util.M = 10
            util.N = 210
        if 'sd2' in data_name:
            util.M = 20
            util.N = 420
            if args.F == 10: util.MIX_AVG = [146, 85, 60, 42, 28, 22, 15, 11, 7, 4]
            elif args.F == 7: util.MIX_AVG = [147, 89, 69, 50, 36, 20, 9]
        elif 'sd5' in data_name:
            util.M = 50
            util.N = 1050
            if args.F == 10: util.MIX_AVG = [361,216,141,102,74,54,42,32,19,9]
            elif args.F == 7: util.MIX_AVG = [370, 233,160, 123, 93, 47, 24]
        elif 'sd' in data_name:
            temp_scale = int(data_name[2:4])
            util.M = 10 * temp_scale
            util.N = 210 * temp_scale
        if 'e1' in data_name:
            util.eta = 1.0
        elif 'e4' in data_name:
            util.eta = 4.0
        elif 'e3' in data_name:
            util.eta = 3.0
        else:
            util.eta = 2.0
        if 't5' in data_name: util.ddt = 0.5
        else: util.ddt = 0.4
        util.F = args.F

        """Initialize variabiles for simulation"""
        # global timer
        self.wall_time = WallTime(args.bucket)
        self.observe_period = args.bucket / 4
        # uses priority queue
        self.mac_timeline = Timeline()
        # plan by products
        self.FIRST_FLAG=True
        self.first_plan=[]
        self.plan = []
        self.slices = []
        self.wip = [0] * util.M
        self.wip_type = [-1] * util.M
        self.wip_op = None
        self.due_list = list()
        self.arrival_list = list()
        self.recent_job = None
        self.record = log
        self.config_load = config_load

        """setup, proc table can be loaded from binary file"""
        file_name = 'paper'
        util.loadTable(file_name)
        # util.setSetupTable('uniform')
        util.setProcTable('manual') # processing time configurations in paper experiments are based on manual settings
        if SAVE_FLAG: self.save_file_name = 'env/config/{}_{}'.format(args.key, args.timestamp)

        """production plans also can be loaded if config_load is specified"""
        if config_load is not None:
            load_split = config_load.split('_')
            slice_name = config_load.replace('_'+load_split[-1],'')
            self.wip_op = load_split[1]
            file_path = 'env/config/{}_slices'.format(slice_name)
            f = open(file_path, 'rb')
            self.slices = pickle.load(f)
            try:
                file_path = 'env/config/{}_wip'.format(slice_name)
                f = open(file_path, 'rb')
            except FileNotFoundError:
                import string
                wip_name = 'o{}'.format(int(slice_name.strip(string.ascii_letters)) % 5 + 1)
                file_path = 'env/config/{}_wip'.format(wip_name)
                f = open(file_path, 'rb')
            self.wip = pickle.load(f)
            #FIXME
            if util.M != len(self.wip): self.wip *= int(util.M/len(self.wip))
            if max(self.wip) < 0.5 * util.PT: self.wip = [element * util.TimeUnit for element in self.wip]

            try:
                fullname = 'env/config/{}_arrival'.format(slice_name)
                f = open(fullname, 'rb')
                self.arrival_list = pickle.load(f)
            except FileNotFoundError:
                print('{} arrival file doesn\'t exist'.format(slice))

        """ If config_load is None (default), only Wrapper() class is initialized for state generation """
        self.stateMgr = Wrapper()
        self.stateMgr.SetStateParameters(sopt=args.sopt, ropt=args.ropt,
                                         action_dim=args.action_dim, state_dim=args.state_dim, bucket=args.bucket,
                                         use=args.use, auxin_dim=args.auxin_dim)

    def generate_jobs(self):
        # all_t, all_size = generate_jobs(self.num_stream_jobs)
        # for t, size in zip(all_t, all_size):
        #     self.timeline.push(t, size)
        """
        planning function
        *simple ver: equal processing time assumption
        1. N to F slices (P_n:1~N)
        2. total daily plan ~ AC*M (sum of P_nd:1~D)
        :return:
        """

        slices=list() # slices indicate the list of job quantities per each family
        criteria = 5 # the quantity of minor family P(f) can't be smaller than criteria
        perturbance = 0.05
        DAYSLICE_FLAG = False
        slice_factor = 1
        if self.config_load is not None:
            slice_factor = util.N / 210
            if type(self.slices[0]) is list:
                # slices = [row[:] for row in self.slices]
                slices = [[int(v*slice_factor) for v in row] for row in self.slices]
                if len(slices) != util.F: slices = [*zip(*slices)]
            else:
                for p in range(util.F):
                    slices.append(divide_uniform(int(self.slices[p]*slice_factor),util.Horizon,ratio=perturbance, criteria=criteria))
        else:
            if DAYSLICE_FLAG:
                temp_slices = divide_uniform(util.N, util.Horizon, ratio=perturbance)
                for d in range(util.Horizon):
                    slices.append(divide_geometric(temp_slices[d], util.F))
                slices = [*zip(*slices)]
            elif self.opt_mix == 'manual':
                slices = util.slices
            else:
                if self.opt_mix == 'geom':
                    temp_slices = divide_geometric(util.N, util.F)
                elif self.opt_mix == 'geomsort':
                    temp_slices = sorted(divide_geometric(util.N, util.F), reverse=True)
                elif 'test' in self.opt_mix:
                    test_slices = util.MIX_AVG
                    test_opt = self.opt_mix.split('_')[1]
                    if test_opt == 'fix': temp_slices = test_slices
                    else:
                        test_perturb=int(test_opt) / 100
                        for p in range(len(test_slices)):
                            test_slices[p] *= random.uniform(1-test_perturb, 1+test_perturb)
                elif self.opt_mix == 'uniform':
                    temp_slices = divide_uniform(util.N, util.F, ratio=perturbance)

                for p in range(util.F):
                    slices.append(divide_uniform(temp_slices[p],util.Horizon,ratio=perturbance, criteria=criteria))
                    # slices.append(divide_geometric(temp_slices[p], util.Horizon, criteria=3))
            self.slices = [row[:] for row in slices]

            if SAVE_FLAG:
                f = open(self.save_file_name + '_slices', 'wb')
                pickle.dump(self.slices, f)
        # print(self.slices)

        """Generate Job Objects"""
        distribution_duedates = 'step_uniform'
        if self.config_load is None or len(self.first_plan) == 0:
            jobID = 0
            self.max_due = 0
            for p in range(util.F):
                for d in range(util.Horizon):
                    quantity = slices[p][d]
                    for j in range(quantity):
                        if RESERVE_ON:
                            arrival = util.UnitTime * d
                        else:
                            arrival = 0
                        due = util.UnitTime * d
                        "determine due-date distribution"

                        if distribution_duedates == 'step':
                             due += util.UnitTime
                        elif self.config_load is not None and len(self.due_list)!=0:
                            due = self.due_list[int(jobID)]
                            if len(self.arrival_list)!=0: arrival = self.arrival_list[int(jobID)]
                        elif distribution_duedates == 'step_uniform':
                            due += get_due_list(job_num=1, tightness=util.ddt, L=util.UnitTime)
                        elif distribution_duedates == 'uniform':
                            due = get_due_list(job_num=1, tightness=util.ddt, L=2.75*util.UnitTime)
                        elif distribution_duedates == 'VNS_PAPER':
                            """ You can assume any distribution, for example: vns paper """
                            arrival, due = get_due_VNS()
                        else:
                            pass
                        if SAVE_FLAG: self.due_list.append(due)
                        self.plan.append(
                            Job(idx=jobID*slice_factor, type=p, pt=util.PT, arrival=arrival, due=due))
                            # Job(idx=jobID, type=p, pt=util.getProcessingTime(macID=0,type=p), arrival=arrival, due=arrival + util.UnitTime))

                        if slice_factor !=1 : jobID += 1/slice_factor
                        else: jobID += 1
                        if due > self.max_due: self.max_due=due
            self.first_plan = copy.deepcopy(self.plan)
            if SAVE_FLAG:
                f = open(self.save_file_name + '_due', 'wb')
                pickle.dump(self.due_list, f)
        else:
            self.plan = copy.deepcopy(self.first_plan)

    def initialize(self):
        # generate WIP distribution with remaining time
        if self.config_load is None:
            self.wip_op = random.choice(self.opt_inist)
            if 'vm' in self.data_name:
                util.M = random.randint(16,24)
                self.wip = [0] * util.M
                self.wip_type = [-1] * util.M
                self.stateMgr.machine_setup_type = [0] * util.M
            # self.wip_op = 'w4'
            self.wip_type = generate_wip_type(self.slices, self.wip_op)
        elif self.wip_type[0] == -1:
            self.wip_type = generate_wip_type(self.slices, self.wip_op)
        if self.config_load is None or sum(self.wip) == 0:
            for i in range(util.M):
                if self.opt_inirem == 'zero': self.wip[i] = 0
                elif self.opt_inirem == 'random':
                    self.wip[i] = int(random.random() * util.PT)
                elif self.opt_inirem == 'step': self.wip[i] = int(i / util.M * util.PT)
                else:
                    pass
            if SAVE_FLAG:
                f = open(self.save_file_name + '_wip', 'wb')
                pickle.dump(self.wip, f)

        # Initialize machine object with remaining time
        for i in range(util.M):
            temp_type = self.wip_type[i]
            wip = Job(idx=-(i + 1), type=temp_type, pt=0, arrival=0)
            wip.mac = i
            temp_time = self.wip[i]
            self.mac_timeline.push(temp_time, wip)
            self.stateMgr.machine_setup_type[i] = temp_type

        self.recent_job = None
        if args.bucket == 0: #DO NOT pop up the first job if bucket is specified
            new_time, job = self.mac_timeline.pop()
            self.wall_time.update(new_time)
            self.recent_job = job
        # Initialize variable for Agent learning
        # else:
            # self.mac_timeline.push(self.observe_period,None)
        self.stateMgr.set_counter(list(range(util.F)), self.plan, self.FIRST_FLAG)
        self.setup_cnt = 0
        self.decision_number = 0
        self.arrival_check = 0
        self.num_tardy = 0
        self.rnorm = 180 * util.TimeUnit * util.M / 10
        if self.FIRST_FLAG==True: self.FIRST_FLAG=False
    def get_tardiness_hour(self, total_reward):
        # mean tardiness
        return total_reward * (self.rnorm / 60 / util.TimeUnit)
    def get_mean_tardiness(self, tardiness_hour):
        return float(tardiness_hour / util.N)
    def update_arrival(self):
        for j in self.plan:
            if j.arrival <= self.wall_time.curr_time and self.arrival_check < j.arrival:
                if util.BEFORE_AS_STOCKER_FLAG:
                    self.stateMgr.update(j, Location.Before, None)
                else:
                    self.stateMgr.update(j, Location.Before, Location.Waiting)
        self.arrival_check = self.wall_time.curr_time
    def update_tardy(self):
        tardiness = 0
        for j in self.plan:
            temp_tard = self.wall_time.curr_time - j.due
            time_step = self.wall_time.bucket_size if self.wall_time.bucket_size !=0 else self.wall_time.timestep
            if temp_tard > 0 and temp_tard <= time_step:
                self.num_tardy += 1
                tardiness += temp_tard
        return tardiness

    def observe(self, opt='default'):
        mac_dict = self.mac_timeline.to_dict()
        state = []
        feasibility = []
        observe = dict()
        self.stateMgr.set_decision_time(self.wall_time.curr_time)
        if opt == 'test':
            dim = 4
            s_prod = [[0] * dim for _ in range(util.F)]
            """mac state"""
            for t, job in mac_dict.items():
                # 0 : # of mac (processing)
                p = job.type
                s_prod[p][0] += 1
            # 1 : one - hot of type
            for p in range(len(s_prod)):
                if self.recent_job.type == p:
                    s_prod[p][1] += 1
            """"lot state (plan)"""
            for job in self.plan:
                #1 : processing
                s_prod[job.type][2] += 1
            for type in range(util.F):
                # 3 : finished
                s_prod[type][3] += self.stateMgr.get_count(type, Location.Finished)
            for s in s_prod:
                state.extend(s)
        elif opt == 'default':
            key_job = self.recent_job
            if args.bucket != 0 : key_job = None
            self.stateMgr.set_from_mach(mac_dict)
            observe = self.stateMgr.getState(key_job, mac_dict, self.plan, res_gantt=self.record.res_record_dict)
        elif opt == 'upm2012':
            state = self.stateMgr.upm2012(mac_dict, from_type = self.recent_job.type if self.recent_job is not None else 0)
        elif opt == 'upm2007':
            state = self.stateMgr.upm2007(self.record.res_record_dict)
        elif opt == 'ours2007':
            self.stateMgr.set_from_mach(mac_dict)
            state = self.stateMgr.upm2007(self.record.res_record_dict)
        elif opt == 'fab2018':
            state = self.stateMgr.fab2018(mac_dict)
        elif opt == 'none':
            state = []
        else:
            pass
        if 'state' not in observe.keys(): observe.update({'state':state})
        if 'reward' not in observe.keys(): observe.update({'reward':self.stateMgr.reward_dict})
        if args.action_dim == util.F * util.F:
            if self.stateMgr.from_type is None:
                feasible_from = self.stateMgr.get_from_idlist(mac_dict)
                if len(feasible_from) == 0:
                    feasible_from = [-1]
                    print('observe: all machine is not idle in this bucket:', feasible_from)
            else: feasible_from = [self.stateMgr.from_type]
        elif args.action_dim == util.F:
            feasible_from = [0]
        else:
            if args.action_dim == 2:
                if self.stateMgr.from_type is None:
                    temp_from = self.stateMgr.get_from_type(self.plan, self.mac_timeline.to_dict())
                else:
                    temp_from = self.stateMgr.from_type
                if self.get_candidate(temp_from) is False:
                    feasibility = [1]
                else: feasibility = list(range(args.action_dim))
            else: feasibility = list(range(args.action_dim))

        if len(feasibility)==0:
            for fromid in feasible_from:
                for type in range(util.F):
                    if self.get_candidate(type) is not False: feasibility.append((fromid if fromid != -1 else type) * util.F + type)
            if len(feasibility) == 0:
                for fromid in feasible_from:
                    for type in range(util.F):
                        if self.get_candidate(type, reserveFlag=True) is not False: feasibility.append((fromid if fromid != -1 else type) * util.F + type)

        observe.update({'feasibility':feasibility})
        observe.update({'time':self.wall_time.curr_time})
        self.last_state = observe['state']
        self.last_feas = observe['feasibility']
        # print(observe['state'].shape, observe['time'])
        if args.auxin_dim != 0:
            auxin = observe['auxin']
            # print('last reward', self.last_reward)
            # if 1 in auxin[:args.action_dim]: self.last_action = auxin[:args.action_dim].index(1)
        # FIXME; check feasiblity by now mach.
        return observe

    def reset(self):
        self.wall_time.reset()
        self.mac_timeline.reset()
        self.plan.clear()
        self.generate_jobs()
        # self.max_time = generate_coin_flips(self.reset_prob)
        self.recent_job = None
        self.initialize() # initialize environment (jump to first job arrival event)
        # observe = self.observe()
        # self.last_state = observe['state']
        # self.last_action=0
        # self.last_reward=0
        # self.last_feas = observe['feasibility']

    def set_random_seed(self, seed):
        np.random.seed(seed)
        random.seed(seed)

    def getFinishedJob(self):
        tot = 0
        for prod, counter in self.stateMgr.prod_counter.items():
            tot += len(counter)
        return tot

    def check_constraints(self, j):
        check = True
        if j.arrival > self.wall_time.curr_time:
            check = False
        if ABANDON_ON and j.due <= self.wall_time.curr_time:
            check = False
        return check
    def get_candidate(self, action, reserveFlag=False):
        candidates_action = [j for j in self.plan if j.type == action]
        if len(candidates_action) == 0: return False
        candidates = [j for j in candidates_action if self.check_constraints(j)]
        if len(candidates) == 0:
            if reserveFlag: candidates = candidates_action # reservation possible (i-delay)
            else: return False
        candidates.sort(key=lambda j: (j.due, j.idx)) # remove randomness by idx sorting
        return candidates[0]

    def dispatch(self, incoming_job):
        incoming_job.mac = self.recent_job.mac
        if STOCHASTICITY:
            stoc_setup = random.Random().uniform(0.8,1.2)
            stoc_proc = random.Random().uniform(0.8,1.2)
        else: stoc_setup, stoc_proc = 1, 1

        # if incoming_job.idx == 1: print(stoc_setup, stoc_proc)
        incoming_job.st = util.getSetupTime(self.recent_job.mac, self.recent_job.type, incoming_job.type) * stoc_setup
        incoming_job.pt = util.getProcessingTime(self.recent_job.mac, incoming_job.type, factor=stoc_proc)

        start_time = max(incoming_job.arrival, self.wall_time.curr_time)
        end_time = start_time + incoming_job.pt + incoming_job.st
        if end_time is None:
            return
        self.mac_timeline.push(end_time, incoming_job)
        self.stateMgr.machine_setup_type[self.recent_job.mac] = incoming_job.type
        self.plan.remove(incoming_job)
        jobId = incoming_job.idx
        if jobId >= 0:
            if self.wall_time.curr_time < start_time: self.stateMgr.update(incoming_job, Location.Before, Location.Resource)
            else: self.stateMgr.update(incoming_job, Location.Waiting, Location.Resource)
        self.decision_number += 1
        if incoming_job.st > 0:
            self.setup_cnt += 1
            self.stateMgr.prod_counter[incoming_job.type].sn += 1
        # set to compute reward from this time point
        reward = 0
        due = incoming_job.due
        incoming_job.late = end_time - due
        if args.ropt == 'target':
            # if end_time > self.recent_job.arrival + util.UnitTime:
            #     reward = -1
            ratio = 1#util.M / (args.bucket / util.PT)
            if end_time <= due:
                reward = 1/ratio
        elif args.ropt == 'intarget':
            if self.wall_time.curr_time + incoming_job.st <= due: reward = 1
        elif args.ropt == 'tardiness':
            reward = -max(0, incoming_job.late) / (60 *util.TimeUnit) #self.rnorm #*util.M
        elif args.ropt == 'epochtard':
            if self.wall_time.bucket_size == 0:
                now_step = self.wall_time.curr_time
            else:
                now_step = self.wall_time.get_now_bucket()
            if due >= now_step:
                tard = max(0, incoming_job.late)
            else:
                tard = end_time - now_step
                self.num_tardy -= 1
            reward = - tard / self.rnorm

        elif args.ropt == 'wtard':
            reward = -util.weight[incoming_job.type] * max(0, incoming_job.late) / self.rnorm
        elif args.ropt == 'setup':
            maxst, maxpt = util.getMaxTime()
            reward = - incoming_job.st / maxst
        elif args.ropt == 'bin':
            if incoming_job.st > 0:
                reward = -1
        else:
            pass

        if self.record is not None:
            self.record.appendInfo('jobID', jobId)
            self.record.appendInfo('job_type', incoming_job.type)
            self.record.appendInfo('resId', self.recent_job.mac)
            self.record.appendInfo('res_setup', self.recent_job.type)
            self.record.appendInfo('time', self.wall_time.curr_time)
            self.record.appendInfo('time_setup', incoming_job.st)
            self.record.appendInfo('time_proc', incoming_job.pt)
            self.record.appendInfo('job_due', due)
            dec = log_decision(self.decision_number, jobId, incoming_job.type, due,
                               self.recent_job.mac, self.recent_job.type, start_time,
                               incoming_job.st, incoming_job.pt, reward, job_arrival=self.recent_job.arrival)
            # dec = DecisionObject(decisionId=self.decision_number, decisionTime=self.wall_time.curr_time
            #                      ,decisionType="pms", decision="{}_{}".format(self.recent_job.mac, jobId)
            #                      ,lotId=jobId, lotSize=incoming_job.pt, lotStatus=due
            #                      ,operationId='none', productType=util.ProdGantt[incoming_job.type]
            #                      ,reward=reward, actionvector=[incoming_job.type])
            self.record.append_history(dec)
        return reward

    def step_tabu(self, job_idx):
        if self.recent_job is not None:
            incoming_job = None
            for j in self.plan:
                if j.idx == job_idx:
                    incoming_job = j
                    break
            if incoming_job is None:
                print('No matched job idx')
                return False
            self.dispatch(incoming_job)
        # pop next job
        while True:
            new_time, job = self.mac_timeline.pop()
            if job is not None: break
        self.wall_time.update(new_time)
        self.recent_job = job
        terminal = (len(self.plan) == 0)
        return job.mac, terminal

    def step_ig(self, job_idx):
        reward = 0
        if self.recent_job is not None:
            incoming_job = None
            for j in self.plan:
                if j.idx == job_idx:
                    incoming_job = j
                    break
            if incoming_job is None:
                print('No matched job idx')
                return False
            reward = self.dispatch(incoming_job)
        # pop next job
        while True:
            new_time, job = self.mac_timeline.pop()
            if job is not None: break
        self.wall_time.update(new_time)
        self.recent_job = job
        terminal = (len(self.plan) == 0)
        return job.mac, reward, terminal

    def step_batcs(self, job_idx):
        if self.recent_job is not None:
            incoming_job = None
            for j in self.plan:
                if j.idx == job_idx:
                    incoming_job = j
                    break
            if incoming_job is None:
                print('No matched job idx', job_idx)
                return False
            self.dispatch(incoming_job)
        # pop next job
        while True:
            new_time, job = self.mac_timeline.pop()
            if job is not None: break
        self.wall_time.update(new_time)
        self.recent_job = job
        terminal = (len(self.plan) == 0)
        return job.mac, job.type, self.wall_time.curr_time, terminal

    def logic_to_type(self, action, list=False): # no intentional delay mode

        if action == 0: policy = 'wspt' #WSPT
        # else: policy = 'wmdd'
        elif action == 1: policy = 'wmdd' #WMDD
        elif action == 2:
            if args.oopt == 'upm2012': policy = 'watcs'
            else: policy = 'ratcs'
        elif action == 3: policy = 'wcovert'
        elif action == 4: policy = 'lfj-wcovert'
        else: policy = ''
        if list: policy += ' list'
        return self.logic_get_action(pol=policy)

    def step(self, action):
        # if action == -1: #infeasible case
        if args.action_dim == 2:# binary action
            if action == 1: to_type = self.logic_get_action(pol='needs')
            else: to_type = self.stateMgr.from_type
        elif args.action_dim < util.F:
            to_type = self.logic_to_type(action)
        else:
            to_type = action % util.F

        # get job of given action (Prod. Type)
        incoming_job = self.get_candidate(to_type, reserveFlag=True)
        # print(action, incoming_job)
        # push job on empty machine
        reward = 0
        if self.recent_job is not None:
            reward += self.dispatch(incoming_job)

        # pop next job
        while True:
            new_time, job = self.mac_timeline.pop()
            if job is not None: break
        self.wall_time.update(new_time)
        self.update_arrival()
        if args.ropt == 'epochtard':
            reward -= self.num_tardy * self.wall_time.timestep / self.rnorm
            reward -= self.update_tardy() / self.rnorm
            # print(reward, self.num_tardy, self.wall_time.timestep, self.wall_time.curr_time)
        if job.idx>=0: self.stateMgr.update(job, Location.Resource, Location.Finished)
        self.recent_job = job

        terminal = (len(self.plan) == 0)  # or ((len(self.mac_timeline) == 0))
        # if self.record is not None: self.record.saveInfo()
        self.stateMgr.addHistory(action, reward)

        return self.observe(args.oopt), reward, terminal

    def step_bucket(self, action):
        # FIXME: IF logic determines from which type (or which mach)
        if  args.oopt == 'upm2012':
            """action is defined as family type"""
            prod_list = self.logic_to_type(action, list=True)
            from_type = prod_list[-1]
            to_type = prod_list[0]
        else:
            """ (paper) action is defined as a tuple of job and machine family type"""
            from_type = action // util.F
            to_type = action % util.F

        actionFlag = True
        LAST_BUCKET_FLAG = False
        terminal = False
        reward_bucket = 0
        new_time = self.wall_time.curr_time
        if len(self.plan) <= util.M:
            LAST_BUCKET_FLAG = True  # last bucket flag
            if from_type == to_type: actionFlag=False
            if args.action_dim == 2: actionFlag=False
        if action == 0: actionFlag=False

        self.recent_job = None
        self.stateMgr.reward_dict.clear()
        for prod in self.stateMgr.prod_list: self.stateMgr.reward_dict[prod]=0
        """
            200218 Note : There is possibility of passing bucket as actionFlag=True (Coudn't do bucket action)
        """
        while terminal is False and (LAST_BUCKET_FLAG or self.wall_time.check_bucket(new_time)): # not finished and same bucket
            # if self.recent_job is not None and self.record is not None: self.record.saveInfo() #step save for gantt chart
            reward = 0
            new_time, job = self.mac_timeline.pop()
            if new_time is None:
                print('Infinite loop casused by non-type value in the timeline list')
                terminal=True
                break
            self.wall_time.update(new_time)
            if RESERVE_ON: self.update_arrival()
            if job is None: #observation flag
                print('Infinite loop casused by non-type Job object')
                _ = self.observe(args.oopt)
                new_time = self.wall_time.curr_time+self.observe_period
                if self.wall_time.check_bucket(new_time) is False: new_time += self.observe_period
                self.mac_timeline.push(new_time, None)
                if terminal is False: new_time, _ =self.mac_timeline.peek()
                continue
            if job.idx >= 0:
                self.stateMgr.update(job, Location.Resource, Location.Finished)
                # job.late = self.wall_time.curr_time - job.due
                # print('result', job.idx, job.late, self.wall_time.curr_time, job.due)
                if 'lag' in args.ropt: reward = 1 if job.late<=0 else 0
            self.recent_job = job
            reward_weight = 1
            # if actionFlag and self.recent_job.type == from_type:
            if actionFlag and self.recent_job.mac == self.stateMgr.from_mach[from_type][to_type] \
                    and (LAST_BUCKET_FLAG or self.stateMgr.get_count(to_type, Location.Waiting) == 1 or \
                         self.wall_time.curr_time + util.getProcessingTime(self.recent_job.mac, to_type) > self.wall_time.get_next_bucket()):
                actionFlag = False
                incoming_job = self.get_candidate(to_type, reserveFlag=True)
            else:
                # action_list = self.logic_get_action('ssu list')
                action_list = self.logic_get_action('seq needs list', force_reserve=True)

                if len(self.plan) < util.M:
                    """setup minimization control for last bucket"""
                    available_action_list = list()
                    for action_type in action_list:
                        if action_type != self.recent_job.type:
                            # when setup is required, setup only occurred for last bucket if there is no resource of to_type
                            if action_type not in list(set(self.stateMgr.machine_setup_type)):
                                # if self.stateMgr.get_from_mach(action_type, mac_dict=self.mac_timeline.to_dict())[0] == -1:
                                available_action_list.append(action_type)
                        else:
                            available_action_list.append(action_type)
                    if len(available_action_list) == 0:  # stop resource rather than setup
                        continue
                    logic_action = available_action_list[0]
                else: logic_action = action_list[0]
                # if # of action-job is 1, shd change logic_action for keeping feasible
                # FIXME: if we add actionFlag condition, tendency to same setup-type leads worse schedule.
                if actionFlag and logic_action == to_type and self.stateMgr.get_count(to_type, Location.Waiting)==1:
                    if len(action_list) == 1:
                        action_list = self.logic_get_action('ssu list', force_reserve=True)
                        if LAST_BUCKET_FLAG: continue
                        logic_action = action_list[0]
                    if len(action_list) > 1 and logic_action == to_type and self.stateMgr.get_count(to_type, Location.Waiting)==1:
                        if logic_action == action_list[1] : logic_action=action_list[0]
                        else: logic_action = action_list[1]

                incoming_job = self.get_candidate(logic_action, reserveFlag=True)
                # if incoming_job.type != self.recent_job.type: reward_weight+=1
            reward += self.dispatch(incoming_job)
            reward_bucket += reward_weight * reward
            self.stateMgr.reward_dict[incoming_job.type] += reward

            terminal = (len(self.plan) == 0) or (len(self.mac_timeline) == 0)
            if terminal is False: new_time, _ =self.mac_timeline.peek()
            else:
                if 'lag' in args.ropt: reward_bucket += 1 if incoming_job.late <=0 else 0
        # update time to next bucket
        self.wall_time.update_bucket(LAST_BUCKET_FLAG)
        self.update_arrival()
        if args.ropt == 'epochtard':
            reward_bucket -= self.num_tardy * self.wall_time.bucket_size / self.rnorm
            reward_bucket -= self.update_tardy() / self.rnorm
        # if self.wall_time.curr_time >= self.max_due: terminal=True # simulation cutting for adequate terminal state
        # print(self.wall_time.curr_bucket, 'ended with', reward_bucket)
        self.stateMgr.addHistory(action, reward_bucket)
        if self.record is not None:
            self.record.appendInfo('actionFlag', actionFlag)
            self.record.appendInfo('action_from', from_type)
        if actionFlag: action = 0 #FIXME: save action idx as 0 if fail to actual implementation

        # print('reward', reward_bucket)
        return self.observe(args.oopt), reward_bucket, terminal

    def step_logic(self, pol='ssu'):
        if self.recent_job is not None:
            job_type = self.logic_get_action(pol)
            incoming_job = self.get_candidate(job_type, reserveFlag=True)
            reward = self.dispatch(incoming_job)
        while True:
            new_time, job = self.mac_timeline.pop()
            if job is not None: break
        self.wall_time.update(new_time)
        self.update_arrival()
        if job.idx >= 0: self.stateMgr.update(job, Location.Resource, Location.Finished)
        self.recent_job = job

        terminal = (len(self.plan) == 0)  # or ((len(self.mac_timeline) == 0))

        return self.observe('none'), reward, terminal
    def _getSlack(self, type):
        incoming_job = self.get_candidate(type, reserveFlag=True)
        setup = util.getSetupTime(self.recent_job.mac, self.recent_job.type, incoming_job.type)
        end_time = self.wall_time.curr_time + incoming_job.pt + setup
        return incoming_job.due - end_time

    def logic_get_action(self, pol='ssu', force_reserve=False):
        remaining_job_type = list()
        if force_reserve is False:
            for type in range(util.F):
                if self.get_candidate(type) is not False: remaining_job_type.append(type)
        if len(remaining_job_type) == 0 or force_reserve:
            for type in range(util.F):
                if self.get_candidate(type, reserveFlag=True) is not False: remaining_job_type.append(type)
            # return False
        if 'seq' in pol:  # 연속 생산 제약
            if self.recent_job.type in remaining_job_type:
                if 'list' in pol: return [self.recent_job.type]
                else: return self.recent_job.type

        if 'ssu' in pol:
            #Shortest SetUp time
            remaining_job_type.sort(key=lambda job_type: util.getSetupTime(self.recent_job.mac, self.recent_job.type, job_type))
        elif 'needs' in pol:
            observe_range = util.ObserveRange
            observe_plan = [j for j in self.plan if j.due <= self.wall_time.curr_time + observe_range]
            remaining_job_type.sort(key=lambda job_type: util.getIndexNeeds(job_type, observe_plan, self.stateMgr.machine_setup_type, option='division'),
                                    reverse=True)
        elif 'spt' in pol:
            if 'w' in pol:
                remaining_job_type.sort(key=lambda job_type: (util.getProcessingTime(self.recent_job.mac, job_type)+
                                                              util.getSetupTime(self.recent_job.mac, self.recent_job.type, job_type)) / util.weight[job_type])
            else:
                remaining_job_type.sort(key=lambda job_type: (util.getProcessingTime(self.recent_job.mac, job_type)+
                                                              util.getSetupTime(self.recent_job.mac, self.recent_job.type, job_type)))
        elif 'lst' in pol:
            #Least slack time
            remaining_job_type.sort(key=lambda job_type: self._getSlack(job_type))
        elif pol == 'wmdd':
            remaining_job_type.sort(key=lambda job_type: max(util.getProcessingTime(self.recent_job.mac, job_type)+util.getSetupTime(self.recent_job.mac,job_type),
                                             self.get_candidate(job_type, reserveFlag=True).due-self.wall_time.curr_time) /
                                             util.weight[job_type])

        elif 'atcs' in pol:
            if util.k1==0:
                due_list = []
                for l in self.stateMgr.due_prod_list.values():
                    due_list.extend(l)
                util.setParamATCS(due_list)
            if pol == 'ratcs':
                # machine ranking index -> job selection
                RI = np.zeros((util.M, util.F))
                TIE = np.zeros((util.M, util.F))
                resid = self.recent_job.mac
                for type in remaining_job_type:
                    cand = self.get_candidate(type, reserveFlag=True)
                    pt = util.getProcessingTime(resid, type)
                    st = util.getSetupTime(resid, self.recent_job.type, type)
                    RI[0][type] = util.weight[type] * util.getIndexATCS(cand.due, pt, self.wall_time.curr_time, st)
                    TIE[0][type] = -(st + pt) / util.weight[type]
                remaining_job_type.sort(key=lambda job_type: (RI[0][job_type], TIE[0][job_type]), reverse=True)
                # print(remaining_job_type, RI[0], TIE[0])
            else:
                atcs = dict()
                for type in remaining_job_type:
                    cand = self.get_candidate(type, reserveFlag=True)
                    if 'w' in pol:
                        pt = util.getProcessingTime(self.recent_job.mac, type) / util.weight[type]
                    else:
                        pt = util.getProcessingTime(self.recent_job.mac, type)
                    atcs.update({type:util.getIndexATCS(cand.due, pt, self.wall_time.curr_time,
                                             util.getSetupTime(self.recent_job.mac, self.recent_job.type,type))})

                    remaining_job_type.sort(key=lambda job_type: (atcs[job_type], self.stateMgr.plan_prod_list[job_type]),
                                        reverse=True)
        elif 'wcovert' in pol:
            # wspt * [1-(slack+/proc)]+
            k = 2
            remaining_job_type.sort(key=lambda job_type: util.weight[job_type]/(util.getProcessingTime(self.recent_job.mac, job_type)+
                                                                                util.getSetupTime(self.recent_job.mac, self.recent_job.type,type))*
                                                         max(0,(1-max(0,self._getSlack(job_type)/(util.getProcessingTime(self.recent_job.mac, job_type)+
                                                                      util.getSetupTime(self.recent_job.mac,self.recent_job.type, type)))/k)))

        if 'list' in pol: return remaining_job_type
        return remaining_job_type[0]
