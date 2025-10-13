"""
Configuration module containing all global variables and paths used in the project.
"""
from datetime import date
import numpy as np

# Data paths - kept exactly as in original code
MacroFactor = ['gdpgr','pconsump','cpi','Unemploy','m3tb','tb10y','aa10y','gpdinv']
VARModel = 'input/var.csv'
Mapping = 'input/mapping.csv'
Mortality = 'input/mortality.csv'
Census = 'input/census.csv'
RecFun = [77.96,66.31,0.34,-1.23,-25.34,-51.33,-3.4,-9.86,-8.45,-32.19,4.26,0.12,-17.02]
histMF = 'input/histMF.csv'
histAR = 'input/histAR.csv'
cholMF = 'input/var1chol.csv'
cholNormal = 'input/normalcholf.csv'
cholRecession = 'input/recessioncholf.csv'
termmix = 'input/termmix.csv'
migration = 'input/migration.csv'

# Model parameters
batch_size = 1000
gamma = 0.99
eps_start = 0.75
eps_end = 0.05
eps_decay = max(1000, batch_size)  # Will be set to max(1000, batch_size)
static_update = 10
neg_multiple = 1.0

# Dynamic liability projection assumptions
fMI = 0.01  # female mortality improvement
mMI = 0.01  # male mortality improvement
salaryGr = 0.02  # salary growth rate
salaryAvgPeriod = 5  # salary averaging period
benefitRate = 0.01  # benefit rate as percentage of averaging salary
COLA = 0.8  # portion of cost of living adjustment based on CPI
maxCOLA = 0.05  # maximum annual COLA rate
lumpSumProb = 0.1  # probability that the benefits are paid as a lump sum
lumpSumDR = 0.04  # discount rate used to calculate lump sum
valuationDate = date(2016,12,31)  # valuation date
planliab = 10000000.  # initial plan liability

# Experiment parameters
n_sim = 1  # number of simulations
n_dyn = 60  # number of dynamic periods in quarter
max_period = 40  # maximum number of periods for projection
plan_asset = 10000000.0 * 1.0  # initial plan asset
saa = [0.5, 0.5]  # sample asset allocation (AA-rated bond, large scale public equity)
rebalance = 0.25
bond_freq = 0.25  # bond coupon frequency in bond fund calculation
expected_rate = 0.04  # expected long term interest rate
bs = True  # rebalance or not

# Sample plan participant information
dateOfBirth = date(1981,6,30)  # date of birth for a sample plan participant
startDate = date(2005,1,31)  # start date of employment
salary = 80000.  # current salary
retireDate = date(2045,6,30)  # expected retirement date
weight = 0.08  # weight of the sample plan participant as in the entire pension plan
occupation = 4  # occupation type
salaryMultiple = np.array([0.9,1.0,1.1,1.2,1.5])  # salary multiple for each occupation type
gender = "F"  # gender of the sample plan participant

# Other parameters
dyn = 10  # dynamic projection time point in quarter
inter = "linear"  # yield curve interpolation method
extro = "tar"  # yield curve extrapolation method
target = 0.04  # target long term interest rate