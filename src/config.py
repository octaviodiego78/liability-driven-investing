"""
Configuration module containing all global variables and paths used in the project.
"""
from datetime import date
import numpy as np
import pandas as pd
from pathlib import Path

# Get the project root directory (parent of src)
_PROJECT_ROOT = Path(__file__).parent.parent

# Data paths - kept exactly as in original code
MacroFactor = ['gdpgr','pconsump','cpi','Unemploy','m3tb','tb10y','aa10y','gpdinv']
VARModel = str(_PROJECT_ROOT / 'input/var.csv')
Mapping = str(_PROJECT_ROOT / 'input/mapping.csv')
Mortality = str(_PROJECT_ROOT / 'input/mortality.csv')
Census = str(_PROJECT_ROOT / 'input/census.csv')
RecFun = [77.96,66.31,0.34,-1.23,-25.34,-51.33,-3.4,-9.86,-8.45,-32.19,4.26,0.12,-17.02]
histMF = str(_PROJECT_ROOT / 'input/histMF.csv')
histAR = str(_PROJECT_ROOT / 'input/histAR.csv')
cholMF = str(_PROJECT_ROOT / 'input/var1chol.csv')
cholNormal = str(_PROJECT_ROOT / 'input/normalcholf.csv')
cholRecession = str(_PROJECT_ROOT / 'input/recessioncholf.csv')
termmix = str(_PROJECT_ROOT / 'input/termmix.csv')
migration = str(_PROJECT_ROOT / 'input/migration.csv')

# Model parameters
batch_size = 1000
gamma = 0.99
eps_start = 0.75
eps_end = 0.05
eps_decay = max(1000, batch_size)  # Will be set to max(1000, batch_size)
static_update = 10
neg_multiple = 1.0
scale = 1.0  # Scaling of volatility. If set to 1.0, historical calibration is used.

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
n_sim = 1000  # number of simulations
n_dyn = 39  # number of dynamic periods in quarter
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

# Load data from CSV files
var = pd.read_csv(VARModel)
fundmap = pd.read_csv(Mapping)
employees = pd.read_csv(Census)
tmpArrays = np.array(pd.read_csv(str(_PROJECT_ROOT / "input/test_states.csv"), header=None))
liabAll = np.array(pd.read_csv(str(_PROJECT_ROOT / "input/liab.csv"), header=None))

# Compute derived values (using late import to avoid circular dependency)
def _initialize_multiple():
    """Initialize the multiple value using functions from utils."""
    from utils import sampleMF, liabMultiple
    
    # Load historical data
    hmf = pd.read_csv(histMF)
    har = pd.read_csv(histAR)
    var1chol = pd.read_csv(cholMF)
    normalChol = pd.read_csv(cholNormal)
    recessionChol = pd.read_csv(cholRecession)
    
    # Generate check value
    check = sampleMF(var, fundmap, hmf, har, var1chol, normalChol, recessionChol, 60, 60, stochastic=False)
    
    # Calculate multiple
    return liabMultiple(employees, check, planliab, inter, extro, target)

# Initialize multiple (will be computed when first accessed)
multiple = _initialize_multiple()