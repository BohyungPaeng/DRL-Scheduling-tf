from collections import deque, defaultdict
import sys, copy, enum
import numpy as np
import cv2

from utils.core.Job import Job
import env.util_sim as util

printFlag = True
debugFlag = False

class Location(enum.Enum):
    Before = 0
    Reenter = 4
    Waiting = 1
    Resource = 2
    Finished = 3

class ProductCounter(object):
    def __init__(self, prod, initial_map):
        self.prod = prod
        self.sn = 0
        self.an = 0
        self.info_map = dict() # key is Location, value is List of information in that location
        for name, member in Location.__members__.items():
            self.info_map[member]= list()
        for map in initial_map:
            l, v = map
            self.info_map[l].append(v)
        self.total = len(initial_map)
    def __len__(self): return self.total # Tot Job Plan per Prod, Same index with prod_list

    def count(self, loc):
        return len(self.info_map[loc])
    def info(self, loc):
        return self.info_map[loc]

    def push(self, loc, info):
        self.info_map[loc].append(info)
    def pop(self, loc, info):
        self.info_map[loc].remove(info)

class Wrapper(object):
    def __init__(self):

        '''Model statics : independent of certain product'''
        self.env_name = "SimEnvSim"
        self.prod_list = list()
        self.StateGroupList = [] # 200307 deprecated, pt will not be grouped for paper, 200428 [] means no group summation
        self.MaxProcTime = 0
        self.MaxSetupTime = 0
        self.ObserveRange = util.UnitTime

        '''Model dynamics'''
        self.plan_prod_list = list()  # Tot Job Plan per Prod, Same index with prod_list
        self.due_prod_list = defaultdict(list)
        self._prodTimeCube = []

        '''Dynamic status'''
        self.from_type = None
        self.prod_counter = defaultdict(ProductCounter)  # dict from F to cnt
        #Cumulative status
        self.max_decision = 0
        self.machine_setup_type = [0] * util.M

        '''State Utility'''
        self.action_history = []
        self.state_history = deque()
        self.reward_history = []
        self.episode = 0
        self.reward_dict = defaultdict(float)

    def SetStateParameters(self, action_dim, auxin_dim, state_dim, bucket, use=[0,1,2,3], auxin_use=[4],
                           sopt='mat', ropt='targetlag', normopt='total'):
        self.action_dim = action_dim
        self.auxin_dim = auxin_dim
        self.auxin_use = auxin_use
        if type(state_dim) is not list: self.state_dim = [state_dim]
        else: self.state_dim = state_dim
        self.bucket = bucket
        self.time_window = bucket if bucket!=0 else sys.maxsize
        # self.num_window = 5
        self.Hw = 6
        self.Hp = 5
        self.sopt = sopt
        self.ropt = ropt
        self.use = use
        self.normopt = normopt

        self.policy_mach = 'SPT'
        # self.policy_mach = 'FIFO'
        self.state_interval_time = 1800
        self.state_max_time = 7200
        self.use_scaling_factor = True

        # if self.episode % 100 == 0:
        # print(self.__dict__)
        # print(len(self.prod_list), )
    def set_counter(self, codelist:list, plan:list, printFlag=False):
        '''
        ProdCodeQtyList: [Code, Plan] value insert
        :param prodQtyList:
        :return:
        '''
        self.prod_list = codelist
        self.plan_prod_list.clear()
        self.due_prod_list.clear()
        self.prod_counter.clear()
        self.curr_time = 0
        self.episode += 1
        self.state_history.clear()
        self.action_history.clear()
        self.reward_history.clear()

        #FIXME : Comparement for Whle machine is required
        if self.MaxProcTime == 0: # Calculate once when Max processing time is not specified
            max_st, max_pt = util.getMaxTime()
            self.MaxProcTime = max_st + max_pt
            self.MaxSetupTime = max_st
            print('Max setup time is', self.MaxSetupTime, self.MaxProcTime)
        tuplelist_byprod=defaultdict(list)
        for job in plan:
            if job.type in self.prod_list:
                if job.arrival > 0:
                    tupleinfo = (Location.Before, job) #dummy before cnter
                    if util.BEFORE_AS_STOCKER_FLAG:
                        tuplelist_byprod[job.type].append((Location.Waiting, job))
                else: tupleinfo = (Location.Waiting, job)
                tuplelist_byprod[job.type].append(tupleinfo)
                self.due_prod_list[job.type].append(job.due)
        self.tot_plan_qty = 0
        for idx in range(len(self.prod_list)):
            prod = self.prod_list[idx]
            self.prod_counter[prod]=ProductCounter(prod, initial_map=tuplelist_byprod[prod])
            self.plan_prod_list.append(len(self.prod_counter[prod]))#({prod: len(self.prod_counter[prod])})
            self.tot_plan_qty += len(self.prod_counter[prod])

        self.from_type = self.get_from_type(plan)
        if self.normopt is None:
            self.normalization_factor = 1
        elif self.normopt == 'total':
            # To minimize the loss of information, constant normalization factor is the best
            self.normalization_factor = self.tot_plan_qty
        else:
            self.normalization_factor = self.plan_prod_list[:]
        self.max_plan_factor = max(self.plan_prod_list) / util.Horizon
        if printFlag:
            # print('Loading TotPlanQty of {} products'.format(len(self.prod_list)), self.prod_list)
            print('Total is {} set Remained prodCodeLPSTList as '.format(self.tot_plan_qty), self.plan_prod_list, self.due_prod_list)
            # printFlag = False

    def update(self, j, loc_from, loc_to):
        assert isinstance(j, Job)
        counter= self.prod_counter[j.type]
        assert isinstance(counter, ProductCounter)
        # counter.pop(loc_from, j.__getattribute__(self.loc_to_infostr(loc_from)))
        # counter.push(loc_to, j.__getattribute__(self.loc_to_infostr(loc_to)))
        counter.pop(loc_from, j)
        if loc_to is not None: counter.push(loc_to, j)
    def loc_to_infostr(self, loc):
        if loc == Location.Finished:
            return 'due'
        elif loc == Location.Waiting:
            return 'arrival'
        elif loc == Location.Resource:
            return 'mac'
        else:
            return 'idx'

    def get_count(self, prod, loc): return self.prod_counter[prod].count(loc)
    def get_info(self, prod, loc, attr=None):
        info_list = self.prod_counter[prod].info(loc)
        if attr is None: return info_list
        attr_list = [info_list[j].__getattribute__(attr) for j in range(len(info_list))]
        return attr_list

    def set_decision_time(self, curr_time): self.curr_time = curr_time
        # f3. remaining tim
    def getState(self, recent_job, mac_dict, plan=None, filter=None, res_gantt=None):
        if recent_job is not None:
            mac_dict.update({self.curr_time:recent_job})
            self.from_type = recent_job.type

        state = self._getFamilyBasedState(mac_dict=mac_dict, res_gantt=res_gantt, filter=filter)

        if 7 in self.use:
            # 7. setup count
            # state.extend([self.prod_counter[prod].sn/24 for prod in self.prod_list])
            # rslt = list(self._get_res_state(res_gantt).reshape(-1))
            # state.extend(rslt)
            pass
        if 8 in self.use:
            # 8. Action history ((F+1) * decision num)
            rslt = self._getHistoryVector(10)  # self.max_decision-1)
            state.extend(rslt)
            # print('history', rslt)

        if len(self.state_dim)==1:
            auxin = [1] if len(plan) <= util.M else [0]
            auxin.extend(self._get_auxin())
            state.extend(auxin)
        observation = {'state': state, 'reward': self.reward_dict}

        if self.auxin_dim != 0:
            auxin = [1] if len(plan)<=util.M else [0]
            auxin.extend(self._get_auxin())
            observation.update({'auxin':auxin})
        return observation

    def get_from_type(self, plan, mac_dict=None):
        if self.action_dim == util.F * util.F: return None
        machine_type_list = self.get_from_idlist(mac_dict)
        if len(machine_type_list)==0:
            machine_type_list = self.get_from_idlist()
            print('all machine is not idle in this bucket:', machine_type_list)
        machine_type_list.sort(key=lambda job_type: util.getIndexGap(job_type, plan, self.machine_setup_type, self.curr_time,
                                                                     fin_job=self.get_count(job_type, Location.Finished)))

        from_type = machine_type_list[0]
        self.from_type = from_type
        return from_type

    def set_from_mach(self, mac_dict=None):
        self.from_mach = -np.ones((util.F, util.F))
        for from_type in self.prod_list:
            if self.policy_mach is None or self.policy_mach == 'RANDOM' or self.policy_mach == 'FIFO':
                self.from_mach[from_type] = [self.get_from_mach(from_type, mac_dict)] * util.F
            elif self.policy_mach == 'SPT':
                self.from_mach[from_type] = self.get_from_mach(from_type, mac_dict)

    def get_from_mach(self, from_type=None, mac_dict=None):
        if self.policy_mach is None or self.policy_mach == 'RANDOM':
            mach_list = list()
            for end_time, job in mac_dict.items():
                if job.type == from_type: mach_list.append(job.mac)
            return np.random.choice(mach_list)
        elif self.policy_mach == 'FIFO':
            for end_time, job in sorted(mac_dict.items()):
                if job.type == from_type and end_time-self.curr_time<self.bucket: return job.mac
            return -1
        elif self.policy_mach == 'SPT':
            from_mach = [-1] * util.F
            mach_list = list()
            for end_time, job in mac_dict.items():
                if job.type == from_type and end_time-self.curr_time<self.bucket:
                    mach_list.append(job.mac)
            if len(mach_list)==0: return from_mach
            for to_type in range(util.F):
                mach_list.sort(key=lambda mach_idx: util.getProcessingTime(mach_idx, to_type)+util.getSetupTime(mach_idx, from_type, to_type))
                from_mach[to_type] = mach_list[0]
            return from_mach

    def get_from_idlist(self, mac_dict=None):
        if mac_dict is None:
            return list(set(self.machine_setup_type))
        else:
            machine_setup_type = list()
            prior_setup_type=list()

            for end_time, job in mac_dict.items():
                if end_time - self.curr_time < self.bucket:
                    remain_num = self.get_count(job.type, Location.Waiting)
                    # if remain_num > 3:
                    machine_setup_type.append(self.machine_setup_type[job.mac])
                    # print('feasi check', job.type, self.get_count(job.type, Location.Waiting))
                    if remain_num == 0: #setup should be occured
                        prior_setup_type.append(self.machine_setup_type[job.mac])
                    elif remain_num == 1 and end_time + job.pt - self.curr_time < self.bucket:
                        prior_setup_type.append(self.machine_setup_type[job.mac])
            if len(prior_setup_type) != 0:
                return list(set(prior_setup_type))
            return list(set(machine_setup_type))

    def _get_last_action(self, from_action=False, to_action=True, flatten=True):
        rslt_action = np.zeros((len(self.prod_list), 2))
        if len(self.action_history) != 0:
            last_action_from = self.action_history[-1] // len(self.prod_list)
            rslt_action[last_action_from,0] += 1
            last_action = self.action_history[-1] % len(self.prod_list)  # always history means to-type action
            rslt_action[last_action,1] += 1
        if from_action and to_action:
            pass
        elif from_action:
            rslt_action = rslt_action[:,0]
        elif to_action:
            rslt_action = rslt_action[:,1]
        else:
            pass
        if flatten: return list(rslt_action.reshape(-1))
        else: return rslt_action
    def _get_auxin(self):

        # 0 ~ 2 : from type related state
        if 0 in self.auxin_use:
            rslt = self._get_from_state()
        else: rslt = [[0]] * 3 #null 2D from-type state

        share_auxin = True if type(self.auxin_dim) is list and len(self.auxin_dim) == 2 else False

        # 3 : last action vector
        rslt_action = self._get_last_action(True, True, False)
        # rslt_action = self._get_last_action()
        if share_auxin: rslt.append([rslt_action] * len(self.prod_list))
        else: rslt.append(rslt_action)

        state = []
        for filter_idx in self.prod_list:
            state_by_filter = []
            for i in self.auxin_use:
                if i >= 4: continue
                elif type(rslt[i][0]) is list or type(rslt[i][0]) is np.ndarray:
                    state_by_filter.extend(rslt[i][filter_idx])
                else:
                    state_by_filter.append(rslt[i][filter_idx])

            if share_auxin:
                state.append(state_by_filter)
            else:
                state.extend(state_by_filter)

        # 4 : last reward
        if 4 in self.auxin_use:
            last_reward = 0 if len(self.reward_history) == 0 else self.reward_history[-1]
            last_reward /= 20
            if share_auxin:
                for state_by_f in state:
                    state_by_f.append(last_reward)
            else:
                state.append(last_reward)
        return state
    def _get_from_state(self, mac_dict=None): #machine or from-type related state
        rslt = []

        # 4. setup Time (S * 1)
        rslt4 = self.normalization_maxabs(self._getSetupTimeVector(), self.MaxSetupTime)

        # 5. Setup State (S * 1, one hot encoding)
        if mac_dict is not None:
            rslt5 = self._get_last_action(True, True, flatten=False)
        else:
            rslt5 = self._getSetupTypeVector()

        # 6. Processing Time (S * 1)
        rslt6 = self.normalization_maxabs(self._getProcTimeVector(), self.MaxSetupTime)

        rslt.append(rslt4)
        rslt.append(rslt5)
        rslt.append(rslt6)
        return rslt
    def _getFamilyBasedState(self, mac_dict, res_gantt=None, filter=None):
        """
        get FBS in the paper
        1. Calculate state features using various methods contained in Wrapper() class.
        2. Generate list of the state vectors named 'rslt'
        3. Stack rslt as 2-D matrix based on t he family type """

        # 0. Processd Job Count( F * 1)
        rslt0 = self._getTrackOutCnt()

        # 1. Processing Job Count (F * 5), Discretized by remaining time.
        # [[20%], [40%], [60%], [80%], [80%+],
        #  [20%], [40%], [60%], [80%], [80%+] ... ]

        if self.sopt == 'time':
            """ S_p (paper) """
            rslt1 = self._getProcessingCnt(mac_dict)
        elif self.sopt == 'proc':
            #. This is experimental states (Processing job attributes)
            rslt1 = self._get_proc_info(mac_dict)
        else:
            rslt1 = self._getProcessingCnt(mac_dict)
            rslt1 = self.normalization_maxabs(rslt1, util.M)
            rslt1 = np.concatenate((rslt1, self._get_proc_info(mac_dict)), axis=1)

        # 2. Waiting job (N_F * 2H_W)
        # rslt2 = self._getWaitingCnt() #. This is experimental states (based on observation time range)
        """ S_w (Paper) """
        rslt2 = self._getTimeCnt(Location.Waiting, job_info=True)

        # 3. Before entrance (F * 4), This is experimental (when ready time is not zero)
        rslt3 = self._getEnterCnt()

        rslt0 = self.normalization_maxabs(rslt0, self.normalization_factor)
        rslt2 = self.normalization_maxabs(rslt2, self.normalization_factor)
        rslt3 = self.normalization_maxabs(rslt3, self.normalization_factor)

        rslt = []
        rslt.append(rslt0)
        rslt.append(rslt1)
        rslt.append(rslt2)
        rslt.append(rslt3)
        rslt.extend(self._get_from_state(mac_dict=mac_dict))
        rslt7 = self._get_res_gantt(res_gantt)
        # rslt7 = self._get_res_state(res_gantt)
        rslt.append(rslt7)
        #--------------------------------------------

        return self._stackFamily(rslt, filter)

    def _stackFamily(self, rslt, filter):
        """
        stack rslt list to the 2-D state
        Two function of this method
        1. filter (list of product code or product family name)
        As default, filter function is not used.
        2. feature usage
        Developers can omit some state features by controlling self.use (specified by args.use)
        """

        state = []
        filter_idx_list = []
        # FIXME : Is it possible to generate only state vector included in filter?? (Short time, Short code)
        if filter is None:
            filter_idx_list = list(range(len(self.prod_list)))
        else:
            for code in filter:
                if code in self.prod_list:
                    filter_idx_list.append(self.prod_list.index(code))
                else:
                    filter_idx_list.append(code)
        for filter_idx in filter_idx_list:
            state_by_filter = []
            for i in self.use:
                if i >= 8: continue

                if type(rslt[i][0]) is list or type(rslt[i][0]) is np.ndarray:
                    state_by_filter.extend(rslt[i][filter_idx])
                else:
                    state_by_filter.append(rslt[i][filter_idx])

            if len(self.state_dim)==2:
                # print(filter_idx, state_by_filter)
                state.append(state_by_filter)
            else: state.extend(state_by_filter)
        return state

    def addHistory(self, last_action, last_reward):
        self.action_history.append(last_action)
        self.reward_history.append(last_reward)

    def _getTimeCnt(self, loc, job_info = False, mac_dict = None):
        """
        S_w (paper): Count jobs with respects to their due-date as in the paper
        H_w (paper) : Hyperparemeters that specify the # of time window to consider
        """
        if loc == Location.Finished: num_window=2 #if fin2 mode
        elif loc == Location.Before: num_window= self.Hw
        else: num_window=self.Hw*2
        if job_info: num_window += 4
        rslt = np.zeros((len(self.prod_list), num_window))
        #F * Loc_num * Window_num
        if mac_dict is not None:
            for end_time, job in mac_dict.items():
                slack = job.due-end_time
                rslt[job.type][self.time_to_window(slack)]+=1
        else:
            for prod in self.prod_list:
                if loc == Location.Waiting:
                    due_list = self.get_info(prod, loc, 'due')
                    pt_list = self.get_info(prod, loc, 'pt')
                    for due, pt in zip(due_list,pt_list):
                        slack = due - self.curr_time #- pt#util.PT
                        rslt[prod][self.time_to_window(slack)]+=1
                    if job_info:
                        idx = -4
                        for due in sorted(due_list):
                            slack = ((due - self.curr_time) / self.time_window) / (self.Hw) * 2
                            rslt[prod][idx] = min(1, max(0, slack + 0.5))
                            idx += 1
                            if idx == 0: break

                elif loc == Location.Finished:
                    late_list = self.get_info(prod, loc, 'late')
                    for late in late_list:
                        if num_window==2: rslt[prod][0 if late<0 else 1] += 1
                        else: rslt[prod][self.time_to_window(-late)] += 1
                elif loc == Location.Before:
                    arr_list = self.get_info(prod, loc, 'arrival')
                    for arr in arr_list:
                        time_till_arrive = arr - self.curr_time #should always positive number
                        if time_till_arrive<=0: print('ERROR! Before Location should be updated')
                        rslt[prod][self.time_to_window(time_till_arrive)-num_window] += 1

        return rslt

    def time_to_window(self, time):
        w = int(np.floor(time / self.time_window))
        if 'target' in self.ropt: w += 1
        else: w += 1 #self.num_window+1
        if w > (self.Hw-1)*2: return self.Hw*2-1 # max value
        elif w <= 0: return 0
        else: return w


    def _calTardiness(self, loc, job):
        assert isinstance(job, Job)
        if loc == Location.Resource:
            job.mac
            self.curr_time

    def _getTrackOutCnt(self):
        """ # of finished jobs, third column of S_h (paper) """
        if self.sopt == 'time': return self._getTimeCnt(Location.Finished)
        rslt = []
        for type in self.prod_list:
            rslt.append(self.get_count(type, Location.Finished))
        return rslt

    def _get_proc_info(self, mac_dict : dict):
        p = len(self.prod_list)
        n = 4
        l = 3+1 # info num: slack, rem, setup, proc
        rslt = np.zeros((p, n*l))
        prod_cnt = [0] * p
        for end_time, job in sorted(mac_dict.items()):
            temp_cnt = prod_cnt[job.type]
            if temp_cnt >= n: continue
            prod_cnt[job.type]+=1
            remain = (end_time - self.curr_time) / self.MaxProcTime
            rslt[job.type][temp_cnt*l] = remain
            if job.idx < 0: continue
            slack = ((end_time - job.due) / self.time_window) / (self.Hw)*2
            rslt[job.type][temp_cnt*l+1] = min(1,max(0,slack+0.5))
            rslt[job.type][temp_cnt*l+2] = job.st / self.MaxProcTime
            rslt[job.type][temp_cnt*l+3] = job.pt / self.MaxProcTime

        return rslt


    def _getProcessingCnt(self, mac_dict : dict):
        """
        S_p (paper) : processing job count (N_F * H_p)
        H_p (paper) : self.Hp, # of time window to consider
        """
        # rslt = [[0] * lvl for _ in range(len(self.prod_list))]
        rslt = np.zeros((len(self.prod_list), self.Hp))
        for end_time, job in mac_dict.items():
            remain = end_time - self.curr_time
            # remain_lvl = int(np.floor(lvl * remain / self.MaxProcTime))
            remain_lvl = int(np.floor(remain / self.time_window))
            if remain_lvl >= self.Hp: remain_lvl = self.Hp-1
            rslt[job.type][remain_lvl] += 1
            # if remain > job.pt: #setup going-on state
            #     rslt[job.type][lvl] = util.M #util.M is right?
        # if self.sopt == 'time':
        #     tc= self._getTimeCnt(Location.Resource, mac_dict=mac_dict)
        #     rslt = np.concatenate((rslt, tc), axis=1)
        #     # return tc
        return rslt

    def _get_res_gantt(self, res_gantt : dict):
        """ S_h (paper) : Machine history of each family """
        res_state = np.zeros((util.F, 3))
        for res_id, res_history in res_gantt.items():
            if len(res_history) > 0:
                for his in res_history:
                    res_state[his.job_type, 1] += his.time_setup
                    res_state[his.job_type, 2] += his.time_proc
                his = res_history[-1]
                til_setup = his.decision_time + his.time_setup - self.curr_time
                til_proc = til_setup + his.time_proc
                if til_setup > 0:
                    res_state[his.job_type, 0] = 1  # setup going on
                    res_state[his.job_type, 1] -= til_setup
                if til_proc: res_state[his.job_type, 2] -= til_proc

        max_time = self.curr_time * util.M if self.curr_time != 0 else 1
        res_state[:, 1] /= max_time
        res_state[:, 2] /= max_time
        return res_state
    def _get_res_state(self, res_gantt : dict):
        """ 0514. experimental features
        1. setup going-on
        2. setup history
        3. pass job history (proc)
        """

        res_state = np.zeros((util.M, 3))
        for res_id, res_history in res_gantt.items():
            proc, setup, idle = 0, 0, 0
            if len(res_history) > 0:
                for his in res_history:
                    proc += his.time_proc
                    setup += his.time_setup
                his = res_history[-1]
                til_setup = his.decision_time + his.time_setup - self.curr_time
                til_proc = til_setup + his.time_proc
                if til_setup > 0:
                    res_state[res_id, 0] = 1  # setup going on
                    setup -= til_setup
                if til_proc: proc -= til_proc

            res_state[res_id, 1] = setup / self.curr_time
            res_state[res_id, 2] = proc / self.curr_time
        return res_state


    def normalization_maxabs(self, rslt:list, factor=None):
        if debugFlag: return rslt
        is_list = (type(rslt[0]) == list) or (type(rslt[0]) == np.ndarray)
        # print(is_list, type(rslt[0]))
        if type(factor) is list: factor = factor[:]

        if factor is None: #default minmax
            factor = max(rslt)
        for prod_idx in range(len(rslt)):
            temp = rslt[prod_idx]
            temp_factor = factor
            if type(factor) is list: temp_factor = factor[prod_idx]
            if temp_factor == 0: temp_factor = max(max(temp),1) if is_list else max(temp,1) # better smoothing than 0.0001
            if is_list:
                temp_rslt = np.array(temp) / temp_factor
                rslt[prod_idx] = list(temp_rslt)
            elif type(rslt[0])==np.ndarray:
                rslt[prod_idx] = temp / temp_factor
            else:
                rslt[prod_idx] = float( temp / temp_factor )

        return rslt

    def fab2018(self, mac_dict):
        """state of TPDQN in V-C. Performance Comparisons in the paper"""
        rslt1 = np.zeros((util.M, util.F))
        # 1. res setup type M * F (one-hot)
        for _, job in mac_dict.items():
            rslt1[job.mac][job.type] = 1
        rslt2 = np.zeros((util.N, util.F))
        rslt3 = np.zeros((util.N, 3))
        # 2. job setup type N * F (one-hot)
        # 3. job location N * loc (one-hot)

        rslt4 = np.zeros(util.F)
        # 4. normalzied deviation of set due date for current operation (here, family)
        for prod in self.prod_list:
            for loc in [Location.Waiting, Location.Resource, Location.Finished]:
                job_list = self.get_info(prod, loc)
                if loc == Location.Waiting:
                    set_due = np.array(self.get_info(prod,loc,attr='due'))
                    if len(set_due)==0: rslt4[prod]=0
                    else: rslt4[prod] = np.std(set_due) / util.UnitTime
                for j in job_list:
                    if j.idx >= 0:
                        rslt2[int(j.idx)][prod] = 1
                        rslt3[int(j.idx)][loc.value-1] = 1
        state = []
        state.extend(rslt1.reshape(-1))
        state.extend(rslt2.reshape(-1))
        state.extend(rslt3.reshape(-1))
        state.extend(rslt4.reshape(-1))
        return state

    def upm2007(self, res_gantt):
        """state of LBF-Q in V-C. Performance Comparisons in the paper"""
        state = list()
        rslt1 = np.zeros((util.N, 2))
        loc_idx_dict = {Location.Waiting:1, Location.Finished:-1, Location.Resource:0}
        max_due = 0
        for prod in self.prod_list:
            for loc in [Location.Waiting, Location.Resource, Location.Finished]:
                job_list = self.get_info(prod, loc)
                for j in job_list:
                    if j.idx >= 0:
                        rslt1[int(j.idx)][0] = j.due-self.curr_time #ej
                        rslt1[int(j.idx)][1] = loc_idx_dict[loc] #qj

                        if j.due > max_due: max_due = j.due
        rslt1[:,0] /= max_due
        rslt2 = np.zeros((util.M, 3))
        b = self.curr_time #latest release time in original paper
        res_state = np.zeros((util.F, 3))
        for res_id, res_history in res_gantt.items():
            lst = 0
            recent_job = -1
            proc_job = -1
            if len(res_history) > 0:
                for i in range(1,len(res_history)+1):
                    his = res_history[-i]
                    if his.time_setup > 0:
                        lst = self.curr_time-his.decision_time
                        break
                his = res_history[-1]
                if his.decision_time + his.time_setup + his.time_proc > self.curr_time:
                    proc_job = his.job_id
                    if len(res_history) > 1:
                        recent_job = res_history[-2].job_id
                else:
                    recent_job = his.job_id
            rslt2[res_id][0] = (recent_job+1) / util.N  # T0i latest job processed
            rslt2[res_id][1] = (proc_job+1) / util.N  # Ti job processing
            if proc_job==-1: lst = 0
            rslt2[res_id][2] = lst / self.MaxProcTime  # ti time from latest setup start
        state.extend(rslt1.reshape(-1))
        state.extend(rslt2.reshape(-1))
        return state

    def upm2012(self, mac_dict, from_type):
        state = []
        proc_dict = defaultdict(list)
        mean_pt = [0] * util.F
        # f1. waiting jobs, pow(2, -1/NJ)
        for prod in self.prod_list:
            nj = self.get_count(prod, Location.Waiting)
            v = 0 if nj==0 else pow(2, -1/nj)
            state.append(v)
            for mach in range(util.M):
                proc_dict[prod].append(util.getProcessingTime(mach, prod)+util.getSetupTime(mach, from_type, prod))
            mean_pt[prod] = float(np.mean(proc_dict[prod]) / util.M)
        """
        util.M features
        f2. prod type
        f3. average proc
        f4. average slack  
        """
        temp = np.zeros((util.M, 3))
        for end_time, job in mac_dict.items():
            temp[job.mac][0] = float((job.type+1) / util.F)
            temp[job.mac][1] = float((end_time-self.curr_time)/mean_pt[job.type])
            temp[job.mac][2] = float((job.due-self.curr_time)/mean_pt[job.type])
        state.extend(temp.reshape(-1))
        """
        util.F features
        f5. min tightness
        f6. max tightness
        f7. average tightness
        f8. time interval number of tightness
        """
        temp = np.zeros((util.F, 3 + 4))
        for prod in self.prod_list:
            due_list = self.get_info(prod, Location.Waiting, 'due')
            if len(due_list) == 0: continue
            temp[prod][0] = float((np.min(due_list)-self.curr_time)/mean_pt[prod])
            temp[prod][1] = float((np.max(due_list)-self.curr_time)/mean_pt[prod])
            temp[prod][2] = float((np.average(due_list)-self.curr_time)/mean_pt[prod])
            g_cnt = [0] * 4
            for due in due_list:
                if due-self.curr_time > np.max(proc_dict[prod]): g_cnt[0] += 1
                elif due-self.curr_time > np.min(proc_dict[prod]): g_cnt[1] += 1
                elif due-self.curr_time > 0: g_cnt[2] += 1
                else: g_cnt[3] += 1
            for g in range(4):
                temp[prod][3+g] = 0 if g_cnt[g] == 0 else pow(2, -1/g_cnt[g])
        state.extend(temp.reshape(-1))

        return state

    """ From here, methods related to experimental state features"""
    def _getWaitingCnt(self):
        """
            S_w (experimental) : waiting job count ( dimension: F * 3)

        """
        rslt = [[0, 0, 0] for _ in range(len(self.prod_list))]

        for type in range(len(self.prod_list)):
            temp_lpst_list = self.get_info(type, Location.Waiting, 'due')
            for lpst in temp_lpst_list:
                if lpst <= self.curr_time: #already late job
                    rslt[type][0] += 1
                elif lpst <= self.curr_time + self.ObserveRange:
                    rslt[type][1] += 1
                else: # future job more than observ range
                    rslt[type][2] += 1
                    pass

        return rslt

    def _getEnterCnt(self):
        """Given distribution of ready time/reentrant time to State vector"""
        """relative arrival time based"""
        if 'time' in self.sopt: return self._getTimeCnt(Location.Before)

        """absolute due time based"""
        rslt = np.zeros((len(self.prod_list), 3))
        for type in range(len(self.prod_list)):
            temp_lpst_list = self.get_info(type, Location.Before, 'arrival')
            for lpst in temp_lpst_list:
                if lpst <= self.curr_time: #already late
                    rslt[type][0] += 1
                elif lpst <= self.curr_time + self.ObserveRange:
                    rslt[type][1] += 1
                else: # future job more than observ range
                    rslt[type][2] += 1
                    pass
        return rslt

    def _getProcTimeVector(self):
        state = []
        if self.from_type is None:
            for to_type in self.prod_list:
                temp_list = list()
                for from_type in self.prod_list:
                    if self.policy_mach == 'RANDOM':
                        from_mach_list = [i for i, x in enumerate(self.machine_setup_type) if x == from_type]
                        temp_tot = 0
                        temp_cnt = 0
                        for from_mach in from_mach_list:
                            temp = float(util.getProcessingTime(from_mach, to_type))
                            if temp < 0: temp = 0
                            temp_tot += temp
                            temp_cnt += 1
                        temp_list.append(0 if temp_cnt == 0 else float(temp_tot / temp_cnt))
                    else: # machine was dispatched(determined) by set_from_mach
                        from_mach = self.from_mach[from_type][to_type]
                        temp_list.append(0 if from_mach == -1 else float(util.getProcessingTime(from_mach, to_type)))
                state.append(temp_list)
            return state

        from_mach_list = self.from_mach[self.from_type] if self.from_type is not None else [0]
        for to_type in self.prod_list:
            temp_tot, temp_cnt = 0, 0
            for mach in from_mach_list:
                temp_tot += util.getProcessingTime(mach, to_type)
                temp_cnt += 1
            state.append([float(temp_tot/temp_cnt)])
        return state
    def _getSetupTimeVector(self):
        state = []

        if self.from_type is None:
            for to_type in self.prod_list:
                temp_list = list()
                for from_type in self.prod_list:
                    if self.policy_mach == 'RANDOM':
                        from_mach_list = [i for i, x in enumerate(self.machine_setup_type) if x == from_type]
                        temp_tot = 0
                        temp_cnt = 0
                        for from_mach in from_mach_list:
                            temp = float(util.getSetupTime(from_mach, from_type, to_type))
                            if temp < 0: temp = 0
                            temp_tot += temp
                            temp_cnt += 1
                        temp_list.append(self.MaxSetupTime if temp_cnt == 0 else float(temp_tot / temp_cnt))
                    else: # machine was dispatched(determined) by set_from_mach
                        from_mach = self.from_mach[from_type][to_type]
                        temp_list.append(self.MaxSetupTime if from_mach == -1 else float(util.getSetupTime(from_mach, from_type, to_type)))
                        # temp_list.append(0 if from_mach == -1 else float(util.getSetupTime(from_mach, from_type, to_type)))
                state.append(temp_list)
            return state
        state = []

        for to_type in self.prod_list:
            temp_tot = 0
            temp_cnt = 0
            for from_type in [self.from_type]:
                for from_mach in self.from_mach[from_type] :
                    temp = float(util.getSetupTime(from_mach, from_type, to_type))
                    if temp < 0: temp = 0
                    temp_tot += temp
                    temp_cnt += 1
            state.append([float(temp_tot / temp_cnt)])
        return state
    def _getSetupTypeVector(self):
        if self.from_type is None:  # from_type is decided by agent
            rslt_macst = np.diag([1]*len(self.prod_list))
        else:
            # 4. Setup State (S * 1, one hot encoding)
            mac_setup = self.from_type
            rslt_macst = [0] * len(self.prod_list)
            rslt_macst[mac_setup] += 1
        return self.normalization_maxabs(rslt_macst, 1)

    def _getHistoryVector(self, window_size=1):
        # vector_size = 1+len(self.prod_list)
        vector_size = 1+self.action_dim
        his_vector = [0] * (window_size * vector_size)
        his = self.action_history
        for i in range(window_size):
            if i>=len(his):
                his_vector[vector_size * i + vector_size - 1] += 1
                continue
            codeIdx = his[-(i+1)]
            his_vector[vector_size*i+codeIdx] += 1
        return his_vector