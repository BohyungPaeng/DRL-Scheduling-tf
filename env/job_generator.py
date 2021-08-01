import numpy as np
from config import *
from random import randint, uniform, shuffle, Random, sample, choice

def generate_jobs(num_stream_jobs):

    # time and job size
    all_t = []
    all_size = []

    # generate streaming sequence
    t = 0
    for _ in range(num_stream_jobs):
        if args.job_distribution == 'uniform':
            size = int(np.random.uniform(
                args.job_size_min, args.job_size_max))
        elif args.job_distribution == 'pareto':
            size = int((np.random.pareto(
                args.job_size_pareto_shape) + 1) * \
                args.job_size_pareto_scale)
        else:
            print('Job distribution', args.job_distribution, 'does not exist')

        if args.cap_job_size:
            size = min(int(args.job_size_max), size)

        t += int(np.random.exponential(args.job_interval))

        all_t.append(t)
        all_size.append(size)

    return all_t, all_size

def divide_uniform(number, parts_number, ratio=0, criteria=0):
    if criteria >= number:
        d = randint(0, parts_number - 1)
        parts = [0]*parts_number
        parts[d] = number
        return parts
    if parts_number > number:
        raise ValueError("Number of parts can't be higher than the number");

    parts = []
    number_rest = number
    average = number / parts_number

    for i in range(1, parts_number + 1):
        if (i == parts_number):
            parts.append(number_rest)
            break
        else:
            new_number = int(uniform(1-ratio, 1+ratio) * average)
        number_rest -= new_number
        parts.append(new_number)

    shuffle(parts)
    return parts

def divide_geometric(number, parts_number, allow_zero=False, criteria=0):
    """
    Divide a given number into bunch of digits.
    Better way)
    1.Push N zeroes in it.
    2.Push K-1 ones in it.
    3.Shuffle the array.
    4.#s of Zero are bucket size.
    """
    if criteria >= number:
        d = randint(0, parts_number - 1)
        parts = [0]*parts_number
        parts[d] = number
        return parts
    if parts_number > number:
        raise ValueError("Number of parts can't be higher than the number");

    parts = []
    number_rest = number
    average = number / parts_number

    for i in range(1, parts_number + 1):
        if (i == parts_number):
            parts.append(number_rest)
            break
        else:
            new_number = np.random.randint(0, number_rest) if allow_zero else np.random.randint(1, (number_rest - (parts_number - i)) // 2)

        number_rest -= new_number
        parts.append(new_number)

    return parts


import env.util_sim as util
import sys

def generate_wip_type(slices, wip_op='wall', swapFlag=False):
    wip_type = [-1] * util.M
    type_order = list()
    for i in range(len(slices)):
        type_order.append(i)
    type_order.sort(key=lambda t: slices[t], reverse=True)


    for i in range(util.M):
        if wip_op == 'wall':
            wip_type[i] = i % util.F
        elif wip_op == 'w1':
            wip_type[i] = type_order[0]
        elif wip_op == 'w2':
            if i < util.M * 0.2:
                wip_type[i] = 1#type_order[1]
            else:
                wip_type[i] = 0#type_order[0]
        elif wip_op == 'w4':
            if i < util.M * 0.5:
                wip_type[i] = type_order[0]
            elif i < util.M * 0.8:
                wip_type[i] = type_order[1]
            elif i < util.M * 0.9:
                wip_type[i] = type_order[2]
            else:
                wip_type[i] = type_order[3]
    if swapFlag:
        n = randint(1, int(round(util.M * 0.2)))
        selected_resource_list = sample(range(util.M), n)
        for i in selected_resource_list:
            wip_type[i] = choice(type_order)
        print("swap initial setup status of machines: ", selected_resource_list)
    return wip_type

estL = None
def get_cmax_estimated(policy=None, slices=None, p_mj_list=None):
    global estL
    if estL is None:
        # computing L
        if policy == 'uniform':
            estL = 0
            for i in range(util.F):
                min_p = min([util.getProcessingTime(j, i) for j in range(util.M)])
                estL += min_p
            min_s = 1000000000
            for i in range(util.M):
                for j in range(util.F):
                    for k in range(util.F):
                        st = util.getSetupTime(i, j, k)
                        if st != 0 and st < min_s:
                            min_s = st
            print(min_p, min_s, estL)
            estL += min_s * util.N
            estL /= util.M  # tabu setting
            print('uniform style L', estL)
        elif policy == 'TABU':
            '''
            job의 type을 정하는 logic을 따라해서 각 job의 min processing time과 min setup time을 계산 (type은 밑에서 정하므로)
            '''
            estL=0
            for p in range(util.F):
                for d in range(util.Horizon):
                    quantity = slices[p][d]
                    for j in range(quantity):
                        min_p = sys.maxsize
                        min_s = sys.maxsize
                        for m in range(util.M):
                            pt = util.getProcessingTime(m, p)
                            if pt < min_p:
                                min_p = pt
                            for p_ in range(util.F):
                                st = util.getSetupTime(m, p, p_)
                                if st != 0 and st < min_s:
                                    min_s = st
                        estL += min_p + min_s
            estL /= util.M
            print('Tabu Style L', estL)
        elif policy == 'Chen':
            jobID_type = list()
            for p in range(util.F):
                for d in range(util.Horizon):
                    quantity = slices[p][d]
                    for j in range(quantity):
                        jobID_type.append(p)
            C_max = 0
            # for j in range(util.N):
            #     for m in range(util.M):
            #         C_max += util.PTable[m][jobID_type[j]] * p_mj_list[m][j] / (util.M * util.M)
            #         for i in range(util.N):
            #             C_max += util.STable[m][jobID_type[i]][jobID_type[j]] / (util.N * util.M * util.M)
            for j in range(util.N):
                prod = jobID_type[j]
                for m in range(util.M):
                    C_max += util.getProcessingTime(m, prod) / (util.M * util.M)
                    for f in range(util.F):
                        C_max += util.getSetupTime(m, f, prod) / (util.F * util.M * util.M)
            print('Chen Style L', C_max)
            estL = C_max

def get_due_list(job_num=1, tightness=0.4, R=0.8, L=None, slices=None, info=None):
    R = 0.4  # tabu setting
    T = tightness  # tabu setting (loose)
    # T = 0.5 # tabu setting (tight)
    if L is None:
        # get_cmax_estimated('uniform', slices, info)
        get_cmax_estimated('Chen', slices, info)
        L = estL

    if job_num == 1: return uniform(L*(1-T-R), L*(1-T+R)) // util.TimeUnit * util.TimeUnit
    due_list = list()
    for i in range(job_num):
        due_list.append( int(uniform(L*(1-T-R), L*(1-T+R))) )
    return due_list

total_proc_time = 0
def get_due_VNS(B=6, alpha=0.25, beta=0.25):
    '''
    B : max batch size [3,6]
    alpha : arrival factor [0,25, 0.5, 0.750
    beta : due date factor
    '''
    global total_proc_time
    if total_proc_time == 0:
        print('total_proc_time calculation')
        '''
        VNS 논문에서는 기계별로 job type에 따라 processing time이 다르지 않음
        '''
        # # identical parallel machine
        # for p in range(util.F):
        #     total_proc_time += util.PTable[0][p] * (8 * util.M)
        # total_proc_time = total_proc_time / (util.M * B)
        '''
        기계별로 job type에 따라 processing time이 다를 경우
        '''
        for m in range(util.M):
            for p in range(util.F):
                total_proc_time += util.getProcessingTime(macID=m, type=p) * (8 * util.M)
        total_proc_time = total_proc_time / (util.M * util.M * B)
    arrival = uniform(0, alpha * total_proc_time)// util.TimeUnit * util.TimeUnit
    due = arrival + uniform(0, beta * total_proc_time)// util.TimeUnit * util.TimeUnit
    return arrival, due
