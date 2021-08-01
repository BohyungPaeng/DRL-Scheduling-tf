import numpy as np
from config import *
from collections import defaultdict
import pickle, random, sys, math

STable = defaultdict(list)
PTable = defaultdict(list)
TimeUnit = 60
Horizon = 3
UnitTime = 24 * 60 * TimeUnit
# UnitTime = 180 * TimeUnit * 1.53
# UnitTime = 60 * TimeUnit * 29.47
"""RI setting"""
# UnitTime = 60 * TimeUnit * 23
# Horizon = 4
ObserveRange = 2880* TimeUnit
ST = 360* TimeUnit  # setup time
PT = 180* TimeUnit  # processing time
PTMODE = 'default'
# PTMODE = 'identical'
# PTMODE = 'constant'
M = 20  # of machine
'''new'''
N = 420  # of job
'''old'''
# N = 150  # of job
F = 10 # number of product type (a.k.a. family) args.action_dim
eta = 2.0
ddt = 0.4
AC = UnitTime / PT  # average capacity(job #) for one day
weight = [1.1, 0.9, 1.3, 1.25, 1.22, 0.77, 0.65, 0.80, 0.99, 1.0]

MIX_AVG = [67, 43, 29, 21, 16, 12, 9, 7, 4, 2]

BEFORE_AS_STOCKER_FLAG = False
""" Env Static parameters """
dailyMaxSetup = 0
UTILCUTTIME = 86400
avgLotSize = 1000

"""Decision related Env"""
# Simulation Day로 끊을것인지 결정
timeCut = True
datCutDay = 1

# M, F, N이 config에 따라 바뀌므로 바꿔주는 세팅
def setM(m):
    global M
    M = m
    # VNS 논문의 세팅
    # global M, N
    # M = m
    # N = 8 * M * F

def setP(p):
    global F
    F = p
    # VNS 논문의 세팅
    # global F, N
    # F = p
    # N = 8 * M * F

def setN(n):
    global N
    N = n

def saveTable(file_name):
    f = open(file_name+'_setupcfg', 'wb')
    pickle.dump(STable, f)
    f = open(file_name+'_proccfg', 'wb')
    pickle.dump(PTable, f)
    print(STable)
    print(PTable)

def loadTable(file_name):
    fullname = 'env/config/{}_setupcfg'.format(file_name)
    print('load STable, PTable : ', fullname)
    f = open(fullname, 'rb')
    global STable, PTable
    STable = pickle.load(f)
    fullname = 'env/config/{}_proccfg'.format(file_name)
    f = open(fullname, 'rb')
    PTable = pickle.load(f)
    # print(STable)
    # for m in range(M):
    #     print(m, np.mean(np.array(STable[m])), np.std(np.array(STable[m])))
    # exit()

def setSetupTable(opt='constant'):
    for m in range(M):
        table = np.zeros((F, F))
        for f in range(F):
            for t in range(F):
                st = 0
                if f!=t:
                    if opt=='constant': st = ST
                    elif opt=='uniform': st = ST*random.uniform(0.7,1.2)
                    elif opt=='VNS': st = 0
                    elif opt=='Chen': st = TimeUnit*random.uniform(10, 100)
                    table[f][t]=int(st)
        STable[m]=table

def setProcTable(opt='constant'):
    tt = [[1.1, 0.9, 1.3, 1.25, 1.22, 0.77, 0.65, 0.80, 0.99, 1.0],
          [1.0, 1.0, 1.25, 1.3, 1.22, 0.77, 0.65, 0.99, 0.80, 1.0],
          [1.0, 1.0, 1.25, 1.3, 1.22, 0.77, 0.65, 0.99, 0.80, 1.0],
          [1.0, 1.0, 1.25, 1.3, 1.22, 0.77, 0.65, 0.99, 0.80, 1.0],
          [1.1, 0.9, 1.3, 1.25, 1.22, 0.77, 0.65, 0.80, 0.99, 1.0],
          [1.05, 0.95, 1.3, 1.25, 0.77, 1.22, 0.65, 0.80, 0.99, 1.0],
          [0.95, 1.05, 1.3, 1.25, 0.77, 1.22, 0.65, 0.80, 0.99, 1.0],
          [1.1, 0.9, 1.3, 1.25, 1.22, 0.77, 0.65, 0.80, 0.99, 1.0],
          [1.1, 0.9, 1.3, 1.25, 1.22, 0.77, 0.65, 0.80, 0.99, 1.0],
          [0.9, 1.1, 1.3, 1.25, 0.77, 1.22, 0.65, 1.0, 0.99, 0.8]]
    for m in range(M):
        if opt == 'VNS' and m > 0:
            PTable[m] = table
            # VNS paper has identical processing time for each machine (IPMS problem)
            continue
        else:
            table = np.zeros(F)
        for p in range(F):
            if opt == 'constant':
                pt = PT
            elif opt == 'uniform':
                pt = PT*random.uniform(0.9,1.1)
            elif opt == 'manual':
                pt = PT*tt[m%10][p]
            elif opt == 'VNS':
                temp = random.Random().random()
                if temp < 0.2:
                    pt = 0.2 * PT
                elif temp < 0.4:
                    pt = 0.4 * PT
                elif temp < 0.7:
                    pt = 1 * PT
                elif temp < 0.9:
                    pt = 1.6 * PT
                else:
                    pt = 2.0 * PT
            else:
                pt = 0
            table[p] = int(pt)
        PTable[m]=table
    # print('SETTING NEW PROC', PTable)

def getSetupTime(mach=0, from_type=0, to_type=0):
    mach=0
    st = STable[mach % F][from_type][to_type]
    if st < ST*0.05: st *= TimeUnit
    return st * (eta/2)

def getProcessingTime(macID=0, type=0, factor=1):
    pt = PTable[macID % F][type]
    # Chen setting에서는 TimeUnit이 기준이라 주석 처리
    if pt < PT * 0.05:
        pt *= TimeUnit
    if PTMODE == 'default':
        return pt * factor
    if PTMODE == 'constant':
        return PT
    if PTMODE == 'identical':
        pt = weight
        return pt[type]*PT


def getMaxTime():
    maxst = max([np.max(STable[x]) for x in STable.keys()])
    maxpt = max([np.max(PTable[x]) for x in PTable.keys()])
    if maxst < ST*0.05: maxst *= TimeUnit
    if maxpt < PT*0.05: maxpt *= TimeUnit
    return maxst, maxpt

def getIndexGap(job_type, plan, machine_setup_type, curr_time, fin_job=None):
    """
    EQP GAP: BEST_EQP - NOW_EQP
    BEST_EQP: target plan * avg_proc / rolling time

    Progress rate gap in PKG: (best fin - fin) / plan for rolling time(1-day)
    """
    avg_proc = 0
    num_mac = 0
    plan_type = [j for j in plan if j.type == job_type]
    num_job = len(plan_type)
    if num_job == 0: return -num_mac

    if fin_job is None:
        for mach, setup in enumerate(machine_setup_type):
            if setup == job_type: avg_proc += PTable[mach][job_type]
            num_mac += 1
        avg_proc /= num_mac
        rolling_time = max(j.due for j in plan_type) - curr_time
        UPED = rolling_time / avg_proc
        BEST_EQP = num_job / UPED
        return BEST_EQP - num_mac
    else:
        best_fin = (num_job + fin_job) / max(j.due for j in plan_type) * curr_time
        return (best_fin - fin_job) / num_job

def getIndexNeeds(job_type, plan, machine_setup_type, option='division'):
    num_job = 0
    for j in plan:
        if j.type == job_type:
            num_job += 1
    num_mac = 0
    for t in machine_setup_type:
        if t == job_type:
            num_mac += 1

    if option == 'division':
        # if num_job != 0 and num_job<=3: num_job = 3
        if num_mac == 0: return num_job * 2
        return num_job / num_mac
    else:
        return num_job - num_mac

def getReqEQP():
    """
    BEST_EQP_REQ = (목표치 – 실제치) / (차수 * target_per_day)
    :return:
    """


k1 = 0
k2 = 0
s_bar = ST
def setParamATCS(due_list):
    eta = ST / PT
    beta = eta * 30 / N # 30 is expected number of setup
    cmax = N * PT * (1+beta) / M #4860 for default

    R_due = (np.max(due_list)-np.min(due_list))/cmax
    tau = 1 - (np.mean(due_list) / cmax)
    global k1, k2, s_bar
    if R_due<0.5: k1 = 4.5+R_due
    else: k1 = 6-2*R_due
    k2 = tau/2/np.sqrt(eta)
    s_bar = np.mean(np.array(list(STable.values()))) * TimeUnit
    # k2 = 1
    print(k1, k2, s_bar)
    return k1, k2

def getIndexATCS(due, proc, curr, setup):
    WSPT = 1 / proc
    slackness = max(due-proc-curr, 0)
    MSR = math.exp(-slackness/(k1*PT))
    SU = math.exp(-setup/(k2*s_bar))

    # print(WSPT, MSR, SU)

    return WSPT*MSR*SU

# VNS 논문에서처럼 k1, k2 값을 찾기위해 만듬
def getIndexATCS_(due, proc, curr, setup, max_arrival, k_1, k_2):
    WSPT = 1 / proc
    slackness = max(due - proc - curr + max(max_arrival - curr, 0), 0)
    MSR = math.exp(-slackness / (k_1 * PT))
    SU = math.exp(-setup / (k_2 * s_bar))

    return WSPT * MSR * SU

def getIndexBATCS(dues, proc, curr, setup, max_arrival, k_1, k_2):
    BATCS = 0
    for i in range(len(dues)):
        WSPT = 1 / proc
        slackness = max(dues[i] - proc - curr + max(max_arrival - curr, 0), 0)
        MSR = math.exp(-slackness/(k_1*PT))
        SU = math.exp(-setup/(k_2*s_bar))
        BATCS += WSPT*MSR*SU

    return BATCS * len(dues)

# Calculate k1 as in VNS paper
def getIndexATC(due, proc, curr, max_arrival, k):
    WSPT = 1 / proc
    slackness = max(due - proc - curr + max(max_arrival - curr, 0), 0)
    MSR = math.exp(-slackness/(k*PT))
    return WSPT*MSR

def getIndexBATC(dues, proc, curr, max_arrival, k):
    BATC = 0
    for i in range(len(dues)):
        WSPT = 1 / proc
        slackness = max(dues[i] - proc - curr + max(max_arrival - curr, 0), 0)
        MSR = math.exp(-slackness/(k*PT))
        BATC += WSPT*MSR
    return BATC * len(dues)
