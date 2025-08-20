######################################################################
# Version 6 of RSCM (RSCM_v6)
# Input : climate data, observed LAI or VI data, crop parameters
# Output: simulated LAI, crop growth, and yield
# Coded by: Chi Tim Ng and Jonghan Ko
# Date: 01 June 2025
######################################################################

import os
import glob
from ctypes import CDLL, POINTER, c_int, c_double
import numpy.ctypeslib as npct
import pandas
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
import math
from math import exp, pi, cos, acos, sin, tan, log
from scipy.ndimage import gaussian_filter1d
import cartopy.crs as ccrs
from cartopy.io.shapereader import Reader
from cartopy.feature import ShapelyFeature


##########################################################################################################
# User instruction: Please choose the library file according to your computer system.
##########################################################################################################

fPath = os.path.abspath(os.getcwd()) + '//'  ###*** current working directory ***###
cfun = CDLL(fPath + 'CodeC\RSCM_v2_1.dll')  # Windocws ; *** For Linux, use the below line
#cfun = CDLL(fPath + 'CodeC/RSCM_v2_1.so') # Linux ; *** For Windows, use the above line


##########################################################################################################
# Contents: This py file contains four parts
# Part I: Declaring the functions in the C library RSCM_v2_1.dll (for Windows) or RSCM_v2_1.so (for Linux)
# Part II: Python routines that call the functions in RSCM_v2_1.dll (for Windows) or RSCM_v2_1.so (for Linux)
# Part III: Python routines that simulate a crop growth process.
# Part IV: Python routines that display and save the calculation results.
# Part V: Python routines that read the data from the files.
##########################################################################################################


##########################################################################################################
# Part I: Declaring the functions in the C library RSCM_v2_1.dll (for Windows) or RSCM_v2_1.so (for Linux)
# The library contains three functions.
# RSCM_Multiple:
#   Estimate the parameters (a,b,c,L0,rGDD) in the Remote Sensing and Crop Model (RSCM)
#   from a data of LAI or data of VIs
# GetCoef:
#   To obtain regression coefficients from a dataset containing both LAI and VIs.
# GetLAI:
#   Imply LAI from VIs given the regression coefficients.
##########################################################################################################

arr_3d_double = npct.ndpointer(dtype=np.float64, ndim=3, flags='C_CONTIGUOUS')
arr_2d_double = npct.ndpointer(dtype=np.float64, ndim=2, flags='C_CONTIGUOUS')
arr_1d_double = npct.ndpointer(dtype=np.float64, ndim=1, flags='C_CONTIGUOUS')
arr_1d_int    = npct.ndpointer(dtype=np.int32, ndim=1, flags='C_CONTIGUOUS')

cfun.GetLAI.argtypes = [POINTER(c_double),POINTER(c_int),POINTER(c_int),POINTER(c_int),
    arr_2d_double,arr_2d_double,arr_2d_double,arr_1d_double]

cfun.GetCoef.argtypes = [arr_2d_double,POINTER(c_int),POINTER(c_int),POINTER(c_int),
        arr_2d_double,arr_2d_double]

cfun.RSCM_Multiple.argtypes = [POINTER(c_double),POINTER(c_double),POINTER(c_double),
        POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double),POINTER(c_double),
        POINTER(c_int),POINTER(c_int),POINTER(c_int),POINTER(c_int),POINTER(c_int),POINTER(c_int),
        POINTER(c_int),POINTER(c_int),POINTER(c_int),POINTER(c_int),POINTER(c_int),POINTER(c_int),
        arr_1d_double, arr_2d_double, arr_1d_double, arr_2d_double, arr_2d_double, arr_3d_double, arr_3d_double,
        arr_1d_int, arr_1d_int, arr_2d_double, arr_2d_double]


##########################################################################################################
# Part II: Python routines that call the functions in RSCM_v5.dll (for Windows) or RSCM_v5.so (for Linux)
# Optim_RSCM_LAI and Optim_RSCM_VI:
#   Estimate the parameters (a,b,c,L0,rGDD) in the Remote Sensing and Crop Model (RSCM)
#   Call RSCM from the library. Default values are given to some inputted arguments.
#   Optim_RSCM_LAI can be used when the data of LAI is supplied.
#   Optim_RSCM_VI can be used when the data of VIs is supplied.
# get_regr_coef_VI_LAI:
#   To obtain regression coefficients linking VIs to LAI
#   Call GetCoef from the library.
#   Input: A dataset containing observed values of both LAI and VIs.
# get_LAI_from_VI
#   Call GetLAI from the library.
#   Input: The regression coefficients from get_regr_coef_VI_LAI
#   Input: A dataset containing VIs only
#   LAI is implied from the inputted VIs based on the inputted regression coefficients.
##########################################################################################################

def Optim_RSCM_LAI(cpara,fmLAI,nrecords,
                    wx_data,obs_d,sub_paddyFields,sub_plantingDate,para0,
                    bayOpt,prior_cov=0,prior_mean_tran=0):
    oOpt = bayOpt
    plantingDateOpt = 1
    impliedLAIOpt = 0
    regressionOpt = 0
    start = 0
    n_unknown = len(para0)
    nObs = obs_d.shape[1]
    n_VI = 1
    ndoy = wx_data.shape[1]
    n_wcol = wx_data.shape[2]
    npixels = wx_data.shape[0]
    
    if bayOpt == 0:
        prior_mean_tran = para0         # Set a default value though it is not actually used in the calculation.
        prior_cov = np.diag(1/para0)    # Set a default value though it is not actually used in the calculation.
    
    reginvSigma = np.empty((n_VI,n_VI))
    coef = np.empty((n_VI,2))
    prior_inv_cov = inv(prior_cov)
    
    paraout = np.zeros((npixels, n_unknown))
    LAIobs = np.zeros((npixels, nObs))
    ODOY = obs_d[:,:,0]

    Tbase = cpara['Tbase']  # base temperature (ex, 15.6 for cotton)
    k     = cpara['k']      # Light extinction coefficient (ex, 0.9 for cotton)
    RUE   = cpara['RUE']    # radiation use efficiency (ex, 2.3 for cotton)
    SLA   = cpara['SLA']    # specific leaf area (ex, 0.01)
    beta1 = cpara['beta1']  # ratio of SR to PAR (ex, 0.418)
    eGDD  = cpara['eGDD']   # GDD at plant emergence
    pd    = cpara['pd']     # parameter d (ex, 10)
    
    cfun.RSCM_Multiple(c_double(np.array(Tbase,dtype=c_double)),
        c_double(np.array(k,dtype=c_double)),c_double(np.array(RUE,dtype=c_double)),
        c_double(np.array(SLA,dtype=c_double)),c_double(np.array(beta1,dtype=c_double)),
        c_double(np.array(eGDD,dtype=c_double)),
        c_double(np.array(pd,dtype=c_double)),c_double(np.array(fmLAI,dtype=c_double)),
        c_int(np.array(oOpt, dtype=c_int)),
        c_int(np.array(n_unknown,dtype=c_int)),c_int(np.array(n_VI,dtype=c_int)),
        c_int(np.array(n_wcol,dtype=c_int)),c_int(np.array(plantingDateOpt,dtype=c_int)),
        c_int(np.array(regressionOpt,dtype=c_int)),
        c_int(np.array(impliedLAIOpt,dtype=c_int)),c_int(np.array(start,dtype=c_int)),
        c_int(np.array(ndoy,dtype=c_int)),c_int(np.array(nrecords,dtype=c_int)),
        c_int(np.array(nObs,dtype=c_int)),c_int(np.array(npixels,dtype=c_int)),
        para0, prior_inv_cov, prior_mean_tran, coef, reginvSigma, wx_data, obs_d, sub_paddyFields, sub_plantingDate, paraout, LAIobs)

    ODOYLAI = np.full((npixels, nObs, 2), np.nan)
    ODOYLAI[:,:,0] = ODOY
    ODOYLAI[:,:,1] = LAIobs

    return paraout,ODOYLAI

def Optim_RSCM_VI(cpara,fmLAI,nrecords,
                    wx_data,obs_d,sub_paddyFields,sub_plantingDate,para0,
                    bayOpt,regressionOpt,impliedLAIOpt,
                    coef,Sigma,prior_cov=0,prior_mean_tran=0):
    oOpt = 2+bayOpt
    plantingDateOpt = 1
    start = 0
    n_unknown = len(para0)
    nObs = obs_d.shape[1]
    n_VI = obs_d.shape[2]-1
    ndoy = wx_data.shape[1]
    n_wcol = wx_data.shape[2]
    npixels = wx_data.shape[0]

    if bayOpt == 0:
        prior_mean_tran = para0         # Set a default value though it is not actually used in the calculation.
        prior_cov = np.diag(1/para0)    # Set a default value though it is not actually used in the calculation.
    
    reginvSigma = inv(Sigma)
    prior_inv_cov = inv(prior_cov)
    
    paraout = np.zeros((npixels, n_unknown))
    LAIobs = np.zeros((npixels, nObs))
    ODOY = obs_d[:,:,0]

    Tbase = cpara['Tbase']  # base temperature (ex, 15.6 for cotton)
    k     = cpara['k']      # Light extinction coefficient (ex, 0.9 for cotton)
    RUE   = cpara['RUE']    # radiation use efficiency (ex, 2.3 for cotton)
    SLA   = cpara['SLA']    # specific leaf area (ex, 0.01)
    beta1 = cpara['beta1']  # ratio of SR to PAR (ex, 0.418)
    eGDD  = cpara['eGDD']   # GDD at plant emergence
    pd    = cpara['pd']     # parameter d (ex, 10)

    cfun.RSCM_Multiple(c_double(np.array(Tbase,dtype=c_double)),
        c_double(np.array(k,dtype=c_double)),c_double(np.array(RUE,dtype=c_double)),
        c_double(np.array(SLA,dtype=c_double)),c_double(np.array(beta1,dtype=c_double)),
        c_double(np.array(eGDD,dtype=c_double)),
        c_double(np.array(pd,dtype=c_double)),c_double(np.array(fmLAI,dtype=c_double)),
        c_int(np.array(oOpt, dtype=c_int)),
        c_int(np.array(n_unknown,dtype=c_int)),c_int(np.array(n_VI,dtype=c_int)),
        c_int(np.array(n_wcol,dtype=c_int)),c_int(np.array(plantingDateOpt,dtype=c_int)),
        c_int(np.array(regressionOpt,dtype=c_int)),
        c_int(np.array(impliedLAIOpt,dtype=c_int)),c_int(np.array(start,dtype=c_int)),
        c_int(np.array(ndoy,dtype=c_int)),c_int(np.array(nrecords,dtype=c_int)),
        c_int(np.array(nObs,dtype=c_int)),c_int(np.array(npixels,dtype=c_int)),
        para0, prior_inv_cov, prior_mean_tran, coef, reginvSigma, wx_data, obs_d, sub_paddyFields, sub_plantingDate, paraout, LAIobs)

    ODOYLAI = np.full((npixels, nObs, 2), np.nan)
    ODOYLAI[:,:,0] = ODOY
    ODOYLAI[:,:,1] = LAIobs

    return paraout,ODOYLAI

def get_regr_coef_VI_LAI(wobs_data,regressionOpt):
    n_wobs = wobs_data.shape[0]
    n_VI = wobs_data.shape[1]-2

    coef = np.zeros((n_VI,2))
    Sigma = np.zeros((n_VI,n_VI))
    
    cfun.GetCoef(np.array(wobs_data, dtype=np.double),c_int(np.array(n_wobs,dtype=c_int)),
        c_int(np.array(n_VI,dtype=c_int)),c_int(np.array(regressionOpt,dtype=c_int)),
        coef,Sigma)
    return coef,Sigma

def get_LAI_from_VI(fmLAI,regressionOpt,coef,Sigma,obs_d):
    nObs = obs_d.shape[0]
    n_VI = obs_d.shape[1]-1
    LAI = np.empty((nObs))

    reginvSigma = inv(Sigma)
    ODOY = np.array(obs_d[:,0], dtype=np.int64)

    cfun.GetLAI(c_double(np.array(fmLAI,dtype=c_double)),c_int(np.array(n_VI,dtype=c_int)),
        c_int(np.array(regressionOpt,dtype=c_int)),c_int(np.array(nObs,dtype=c_int)),
        coef,reginvSigma,obs_d,LAI)
    return pandas.DataFrame({'ODOY':ODOY,'OLAI':LAI})


##########################################################################################################
# Part III: Python routines that simulate a crop growth process.
# The main function is sim_cG
# Inputs:
#   The estimated parameter. Can be obtained from Optim_RSCM_LAI and Optim_RSCM_VI
#   ODOYLAI = days of year that LAIs are observed. The simulated LAIs can then be compared to those observed.
#   ET_para = ET-associated parameters
#   ppara = productivity-associated parameters
#   lpara = location parameters
#   cpara = crop related parameters
#   start = start planting date
#   wx_data = weather data, including daily solar radiation, highest temperature, lowest temperature.
# The functions ET0_HS (Hargreaves-Samani ET0) and ETc (Crop ET) are called in sim_cG
##########################################################################################################

# *** Hargreaves-Samani ET0 (Harvreaves & Samani, 1985)********************
# INPUT: Tmin & Tmax = min & max temp, doy = day of year, lat = latitude
#      k_Rs  = rad. adjustment coef. (default=0.17, inland=0.16, coast=0.19)
# Return: ET0
# *************************************************************************
def ET0_HS(Tmax,Tmin,doy,lat,k_Rs):
  solarConstant = 0.082  # solar constant (0.0820 M2/m2/min)
  d_r = 1 + 0.033*cos((2*pi/365)*doy) # inverse relative distance Earth-Sun
  lat_rad = (pi/180)*lat # latitude in radian
  sDeclination = 0.409*sin((2*pi/365)*doy - 1.39) # solar declination (rad)
  w_s = acos((-tan(lat_rad)*tan(sDeclination))) # sunset hour angle (rad)

  # extraterrestrial radiation
  R_a = (24*60/pi)*solarConstant*d_r*((w_s*sin(lat_rad)*sin(sDeclination))+(cos(lat_rad)*sin(w_s)))

  Tmean = (Tmax + Tmin)/2
  R_ag = 0.408 * R_a  # extraterrestrial radiation in equivalent of ET (mm)
  evapTrans0 = 0.0135*k_Rs * (Tmean + 17.78) * (Tmax - Tmin)**0.5 * R_ag
  return evapTrans0

# ***** Crop ET ****************************************************************************
# INPUT: Tmin & Tmax = min & max temp, lat = latitude
#   k_Rs  = rad. adjustment coef. (default=0.17, inland=0.16, coast=0.19)
#   VI = vegetation index, et_ref = reference ET, cday = cumulative day
#   d_para1 = damping parameter (corr. between LAI & T/ET0), rice = 0.49 (Nay-Htoon et al, 2018)
#   d_para2 = damping parameter (corr. between NDVI & LAI), rice = 0.8 (Nay-Htoon et al, 2018)
#   VI_max = max VI, VI_min = min VI, kc_max = max value of crop coefficient (Kc)
#   ke_opt = option to calculate soil E parameter (0 = using VI, 1 = using soil info)
#   kr_opt = option to calculate E red. coeff. (1 = flooded crops ; 2 = dryland or irrigated crops)
#   fC_soil = field capacity of the soil water (0.07 ~ 0.40), silt loam = 0.22-0.36
#   wiltPoint = wilting point of the soil water (0.02 ~ 0.24), silt loam = 0.09-0.21
#   REW = readily evaporated water (2 ~ 12 mm), silt loam = 8-11
# Return: crop ET
#*****************************************************************************************
def ETc(Tmax,Tmin,lat,k_Rs,VI,et_ref,cday,VI_max,VI_min,kc_max,ke_opt,kr_opt,
    k_cb, fC_soil, wiltPoint, REW):
    if ke_opt == 1:  k_e = 0.25 * (1. - (1.18*(VI-VI_min)))  # soil E coef. using VI
    if ke_opt == 0:  ed_ssoil = 0.5*wiltPoint    # effective depth of the surface soil subject to drying to 0.5*wiltingPoint
    TEW = 1000*(fC_soil - 0.5*wiltPoint)*ed_ssoil  # max depth of water able to be evaporated
    dep_ss = TEW - TEW/cday  # cumulative depletion from the soil surface at the end of day j-1
    if (kr_opt == 1): k_r = 1.0  # E reduction coef. for flooded crops (e.g., paddy rice)
    if (kr_opt == 2): k_r = (TEW - dep_ss)/(TEW - REW)  # E reduc. coef. for dryland & irrigated crops
    if (k_r <= 0.0): k_r = 0.000001
    if (k_r >= 1.0): k_r = 1.0
    k_e = k_r*(kc_max - k_cb)  # soil E coef.
    if k_e >= kc_max: k_e = kc_max

    cropET = (k_cb + k_e) * et_ref
    return cropET

def sim_cG(para0,ODOYLAI,ET_para,ppara,lpara,cpara,sub_plantingDate,wx_data):
    # Get ET parameter values
    a_ndvi = ET_para['a_ndvi']   # LAI-NDVI conversion coef. a, rice = 0.60 (Ko et al., 2015)
    b_ndvi = ET_para['b_ndvi']   # LAI-NDVI conversion coef. b, rice = 0.35 (Ko et al., 2015)
    d_para1 = ET_para['d_para1'] # damping parameter (LAI vs T/ET0), rice=0.49 (Nay-Htoon et al, 2018)
    d_para2 = ET_para['d_para2'] # damping parameter (NDVI vs LAI), rice=0.8 (Nay-Htoon et al, 2018)
    VI_max = ET_para['VI_max']   # max VI or LAI
    VI_min = ET_para['VI_min']   # min VI or LAI
    kc_max = ET_para['kc_max']   # max value of crop coefficient (Kc), rice = 1.2
    ini_D = ET_para['ini_D']     # init., initial crop dev. period : rice = 30 (FAO, Allen et al., 1998)
    dev_D = ET_para['dev_D']     # Dev., development period : rice = 30
    mid_D = ET_para['mid_D']     # mid., mid development period : rice = 80
    late_D = ET_para['late_D']   # late, late development period : rice 40
    k_ini = ET_para['k_ini']     # initial Kc : rice = 1.05
    k_mid = ET_para['k_mid']     # mid Kc : rice = 1.20
    k_end = ET_para['k_end']     # end Kc : rice = 0.90-60
    fC_soil = ET_para['fC_soil'] # field capacity of the soil water (0.07 ~ 0.40), silt loam = 0.22-0.36
    wiltPoint = ET_para['wiltPoint'] # wilting point of the soil water (0.02 ~ 0.24), silt loam = 0.09-0.21
    REW = ET_para['REW']         # readily evaporated water (2 ~ 12 mm), silt loam = 8-11
    kc_opt = ET_para['kc_opt']   # basal crop coeff. cal. option: 0 = FAO Kc, 1 =  LAI based
    ke_opt = ET_para['ke_opt']   # soil E parameter cal. option: 0 = using soil info, 1 = using VI
    kr_opt = ET_para['kr_opt']   # E red. coeff. cal. opt.: 1 = flooded crops ; 2 >= dryland or irrigated crops
    CGP2 = ini_D + dev_D + mid_D + late_D   # crop growing period for ET calculation
    nrecords = int(CGP2)    # CGP from the ET parameters

    Tbase = cpara['Tbase']  # base temperature (ex, 15.6 for cotton)
    k     = cpara['k']      # Light extinction coefficient (ex, 0.9 for cotton)
    RUE   = cpara['RUE']    # radiation use efficiency (ex, 2.3 for cotton)
    SLA   = cpara['SLA']    # specific leaf area (ex, 0.01)
    beta1 = cpara['beta1']  # ratio of SR to PAR (ex, 0.418)
    eGDD  = cpara['eGDD']   # GDD at plant emergence
    pd    = cpara['pd']     # parameter d (ex, 10)

    lat  = lpara['lat']         # latitude of the site
    elev = lpara['elev']        # Elevation of the site
    k_Rs = lpara['k_Rs']        # Rad adj. coef. (default=0.17,inland=0.16,coast=0.19,Gwangju=0.09)

    pmGDD = ppara['pmGDD']      # GDD at plant maturity
    #RTM   = ppara['RTM']        # GDD from reproduction to maturity (RTM)                     
    #fMat  = ppara['fMat']       # fMat = mat. factor
    #fGC   = ppara['fGC']        # fGC = yield conv. factor
    fg1   = ppara['fg1']        # factor of grain partitioning 1 (fg1), rice = 10             
    fg2   = ppara['fg2']        # factor of grain partitioning 2 (fg2), rice = 12             
    a1    = ppara['a1']         # coeff of LUE (LUE_cint = a1 * LAI + b1, Xue et al., 2017)   
    b1    = ppara['b1']         # coeff of LUE (LUE_cint = a1 * LAI + b1, Xue et al., 2017)   
    a2    = ppara['a2']         # coeff of max GPP (GPP_max = a2 * LAI + b2, Xue et al., 2017)
    b2    = ppara['b2']         # coeff of max GPP (GPP_max = a2 * LAI + b2, Xue et al., 2017)  

    npixels = para0.shape[0]
    RSCMProcess = np.full((npixels, nrecords, 13), np.nan)
    cyield = np.full(npixels, np.nan)

    for current in range(npixels):
        start = sub_plantingDate[current]
        if np.sum(para0[current,:]==0) == 0:
            pa = para0[current,0]
            pb = para0[current,1]
            pc = para0[current,2]
            LAI0 = para0[current,3]    # initialize LAI
            rGDD = para0[current,4]

            DOY = np.array(wx_data[current,:,0], dtype=np.int64)     # day of year
            RAD = np.array(wx_data[current,:,1])  # solar radiation
            Tmax = np.array(wx_data[current,:,2]) # max temperature
            Tmin = np.array(wx_data[current,:,3]) # min temperature

            LAI = np.zeros((nrecords))
            OLAI = np.empty((nrecords))
            AGDM = np.zeros((nrecords))
            sDOY = np.zeros((nrecords)) # seasonal DOY
            GDD = np.zeros((nrecords))
            PGP = np.zeros((nrecords))
            NPP = np.zeros((nrecords))
            GPP = np.zeros((nrecords))
            ETcrop = np.zeros((nrecords))

            for i in range(nrecords-1):
                dGDD = (Tmax[i+start]+Tmin[i+start])/2. - Tbase # daily GDD
                if dGDD < 0: dGDD = 0
                GDD[i+1] = GDD[i] + dGDD
            mGDD = GDD[nrecords-1]     # maximum GDD

            DAR = 0              # days after reproduction
            LAI[0] = LAI0        # initialize LAI
            for i in range(nrecords-1):
                tRatio = exp(-k*LAI[i])  # transmission ratio of SR in the canopy
                APAR = beta1*RAD[i+start]*(1.0-tRatio) # absorption of SR based on Beer's law
                dM = RUE * APAR    # biomass conversion based on Monteith's law
                NPP[i] = dM / 2.592  # g CO2/s -> 0.00003 CH2O/s * 86,400 = 2.592 g/day
                #exp_LPF = pa*exp(pb*(GDD[i+1]-eGDD))
                x = pb * (GDD[i+1] - eGDD)
                # clamp to avoid math.exp overflow ### 25.05.24
                x_safe = max(min(x, 700), -700)
                exp_LPF = pa * math.exp(x_safe)
                exp_LPF = exp_LPF / (1.0 + exp_LPF)  
                LPF = 1.0 - exp_LPF   # leaf partitioning function
                dLAI = SLA * LPF * dM

                LSOpt = 0 # LSOpt, 0=parameter-based, 1=respiration-based senescent function
                if (LSOpt==0) and ((LPF==0.0) or (GDD[i+1]>=rGDD)):
                    #fac = 1.0 - exp(-pa*pc*(GDD[i+1] - rGDD))  ### LAI smmoothing factor at the rGDD initiation: 24.12.29 ###
                    LSF = pc*AGDM[i]/pd     # leaf senescence function
                    dLAI = -dLAI*LSF
                if (LSOpt==1) and ((LPF==0.0) or (GDD[i+1]>=rGDD)):
                    MResp = 0.015*AGDM[i]  # maintenace resp (1.5% of AGDM)
                    LSF = pc*MResp         # leaf senescence function
                    dLAI = -dLAI*LSF

                YOpt1 = 0  # YOpt1, 0 = default
                boll_to_lint = 0.05
                temp_mLAI = 1.0
                if (GDD[i] > rGDD-5) and (GDD[i] < rGDD+10): temp_mLAI = LAI[i]
                if (temp_mLAI < 2.):
                    #boll_to_lint = 0.05  # max partitioning for cotton
                    temp_rGDD = rGDD
                    pf0 = 1.0
                elif (temp_mLAI >= 2.) and (temp_mLAI < 5.):
                    #boll_to_lint = 0.025    # reduced partitioning
                    temp_rGDD = rGDD + 200  # delay partitioning
                    pf0 = 0.5
                elif (temp_mLAI >= 5.):
                    #boll_to_lint = 0.015    # least partitioning under higher veg. growth
                    remp_rGDD = rGDD + 250  # delay partitioning
                    pf0 = 0.3

                if (YOpt1 == 0) and (GDD[i+1] >= temp_rGDD) or (i >= (nrecords-30)):
                    # 1) base envelope (as you originally had)
                    fYPF = pmGDD - (fg1 * mGDD) / ((mGDD - temp_rGDD) / fg2)
                    x_safe = max(min(pb * (fYPF - GDD[i+1]), 700), -700)
                    base = 1.0 - pa * math.exp(x_safe)

                    # 2) normalize position in grain‐fill [0,1]
                    stage = (GDD[i+1] - temp_rGDD) / max((mGDD - temp_rGDD), 1e-6)
                    stage = max(0.0, min(1.0, stage))

                    # 3) Beta‐style window to gate the envelope:
                    #    (stage^fg1) * ((1-stage)^fg2) peaks in between
                    window = (stage ** fg1) * ((1.0 - stage) ** fg2)

                    # 4) combine, clip, and accumulate
                    YPF = max(0.0, base * window)
                    PGP[i+1] = PGP[i] + dM * YPF * pf0

                    DAR += 1                    # days after reproduction
                    if (DAR == 1): RDOY = i     # day of year at reproduction

                LAI[i+1] = LAI[i] + dLAI
                if LAI[i+1] < 0.0: LAI[i+1] = 0.0

                LUE_cint = a1*LAI[i+1]+b1    # light use efficiency (Xue et al., 2017)
                GPP_max = a2*LAI[i+1]+b2     # max GPP (Xue et al., 2017)
                # DL_hr = dayLength(n) # default day length = 15.
                PAR_mol = RAD[i+1]*beta1*4.55 *1000000./(3600.*15.) #  PAR in umol/s (J/s -> umol = 4.55)
                GPP_t = ((LUE_cint*GPP_max*PAR_mol)/(LUE_cint*PAR_mol+GPP_max)) # GPP (Xue et al., 2017)
                GPP[i+1] = GPP_t * 12. * (15.*3600.)/1000000.  #  umol -> g

                et_ref = ET0_HS(Tmax[i],Tmin[i],DOY[i],lat,k_Rs) # Hargreaves-Samani ET0

                # Determine FAO basal crop coefficient (Kc)
                if (kc_opt == 0):
                    if (DOY[i+1] <= (DOY[1]+ini_D)): k_cb = k_ini    # initial Kc
                    if (DOY[i+1] > (DOY[1]+ini_D)) and (DOY[i] <= (DOY[1]+ini_D+dev_D)):
                        k_cb = k_ini + (k_mid-k_ini)/dev_D           # development period Kc
                    if (DOY[i+1] > (DOY[1]+ini_D+dev_D)) and (DOY[i+1] <= (DOY[1]+ini_D+dev_D+mid_D)):
                        k_cb = k_mid                                 # mid Kc
                    if (DOY[i+1] > (DOY[1]+ini_D+dev_D+mid_D)) and (DOY[i+1] <= DOY[1]+nrecords):
                        k_cb = k_mid - (k_mid-k_end)/late_D          # late period Kc
                # Determine VI-based Kc
                NDVI = a_ndvi * exp(b_ndvi)
                exponent = d_para1/d_para2 # d_para1 = damping para in LAI vs. T/ET0, d_para2 = damping para in NDVI vs. LAI
                ratio = (VI_max-LAI[i+1])/(VI_max-VI_min)
                if(ratio<0.0001): ratio = 0.0001
                if(ratio>0.9999): ratio = 0.9999
                if (kc_opt != 0): k_cb = 1.0-ratio**exponent # Kcb (Nay-Htoon et al, 2018)

                ETcrop[i+1] = ETc(Tmax[i+1],Tmin[i+1],lat,k_Rs,NDVI,et_ref,CGP2,
                    VI_max,VI_min,kc_max,ke_opt,kr_opt,k_cb,fC_soil,wiltPoint,REW)

                sDOY[i] = DOY[i+start]
                AGDM[i+1] = AGDM[i] + dM

            sDOY[i+1] = sDOY[i]+1
            cyield[current] = PGP[i]
            LAI = gaussian_filter1d(LAI, 5)

            # Align LAIobs with LAI and sDOY so that all have length = nrecords
            # Aligned LAIobs are stored in OLAI
            ODOY = ODOYLAI[current,:,0]
            LAIobs = ODOYLAI[current,:,1]
            OLAI[:] = np.nan #np.NaN
            for i in range(nrecords):
                counter = 0
                for j in ODOY:
                    if (int(j) == int(sDOY[i])) and (LAIobs[counter] > 0.01):
                        OLAI[i] = LAIobs[counter]
                    counter += 1

            RSCMProcessOne = pandas.DataFrame({'DOY':sDOY,'DAP':range(nrecords),
                                                'RAD':RAD[start-1:start+nrecords-1],'Tmax':Tmax[start-1:start+nrecords-1],'Tmin':Tmin[start-1:start+nrecords-1],
                                                'GDD':GDD,'OLAI':OLAI,'LAI':LAI,'AGDM':AGDM,'NPP':NPP,'GPP':GPP,'ETc':ETcrop,'PGP':PGP})
            RSCMProcess[current,:,:] = RSCMProcessOne.to_numpy()
           
    return cyield,RSCMProcess


##########################################################################################################
# Part IV: Python routines that display and save the calculation results.
# plot_cG (for one pixel):
#   Plot both observed and simulated crop growth process (LAI and AGDM) of one pixel in a diagram.
#   Input: RSCMProcess, to be obtained from sim_CG.
# plot_pixgro_out (for multiple pixel):
#   Plot both observed and confidence band of simulated crop growth process (LAI and AGDM) in a diagram.
#   Input: RSCMProcess, to be obtained from sim_CG.
# mapping_yield (for multiple pixel):
#   Plot the predicted yield on the map
# out_file_cG (for one pixel):
#   Print the information of estimated parameters, predicted crop yield, and the crop growth process to a file.
# save_out_pixgro (for multiple pixel):
#   Save the results to output files. These contain the arrays of
#   (i) estimated parameters (a,b,c,Lo,rGDD),
#   (ii) simulated crop growth process (RSCMProcess), and
#   (iii) predicted yields
# save_out_pixgro_summary (for multiple pixel):
#   Similar to save_out_pixgro, but save the summary statistics only.
##########################################################################################################

def plot_cG(RSCMProcess):
    sDOY = RSCMProcess['DOY']
    OLAI = RSCMProcess['OLAI']
    LAI = RSCMProcess['LAI']
    AGDM = RSCMProcess['AGDM']
    
    fig, ax1 = plt.subplots()
    color = 'tab:green'
    ax1.set_xlabel('Day of year')
    ax1.set_ylabel('LAI (m$^2$'+' m$^-$'+'$^2$)', color=color)
    OLAI = np.array(OLAI)
    if np.nanmax(LAI) >= np.nanmax(OLAI):
        maxL = np.nanmax(LAI)
    if np.nanmax(OLAI) >= np.nanmax(LAI):
        maxL = np.nanmax(OLAI)
    ax1.set_ylim((0, maxL*1.2))
    ax1.tick_params(axis='y', labelcolor=color)
    y1 = ax1.plot(sDOY, LAI, '--', color=color, label = 'SLAI')
    y2 = ax1.plot(sDOY, OLAI, color=color, marker='o', label = 'OLAI')

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:blue'
    ax2.set_ylabel('AGDM (g m$^-$'+'$^2)$', color=color)
    ax2.set_ylim((0, np.nanmax(AGDM)*1.1))
    ax2.tick_params(axis='y', labelcolor=color)
    y3 = ax2.plot(sDOY, AGDM, color=color, label='SAGDM')
    fig.tight_layout()  # otherwise the right y-label is slightly clipped

    y_all = y1 + y2 + y3
    labs = [l.get_label() for l in y_all]
    plt.legend(y_all, labs, loc='upper left', frameon=False) # show legends
    plt.show()

# -- plot SLAI and OLAI vs. DOY --
def plot_pixgro_out(RSCMProcess, plot_file_path = None):
    reshaped_RSCMProcess = RSCMProcess.reshape(-1, 13)
    df = pandas.DataFrame(reshaped_RSCMProcess)
    df.columns = ['DOY','DAP','Tmax','Tmin','Rad','GDD','OLAI','SLAI','AGDM','NPP','GPP','ETc','PGP'] ### 24.08.11 ###
    
    # Calculating mean values for OLAI and SLAI grouped by DOY1
    mean_values = df.groupby('DOY')[['OLAI', 'SLAI']].mean().reset_index()
    
    # Calculating standard deviation values for OLAI and SLAI grouped by DOY1
    std_dev_values = df.groupby('DOY')[['OLAI', 'SLAI']].std().reset_index()
    
    # Merge these two DataFrames for a comprehensive view
    mean_std_dev = pandas.merge(mean_values, std_dev_values, on='DOY', suffixes=('_mean', '_std_dev'))
    
    plt.figure(figsize=(5, 3))
    plt.rcParams['font.size'] = '13'
    
    # Plotting OLAI mean values
    plt.plot(mean_std_dev['DOY'], mean_std_dev['OLAI_mean'], marker='o', label='OLAI Mean', color='blue')
    
    # Adding shaded area for OLAI standard deviation
    plt.fill_between(mean_std_dev['DOY'], 
                     mean_std_dev['OLAI_mean'] - mean_std_dev['OLAI_std_dev'], 
                     mean_std_dev['OLAI_mean'] + mean_std_dev['OLAI_std_dev'], 
                     color='blue', alpha=1.0)
    
    # Plotting SLAI mean values
    plt.plot(mean_std_dev['DOY'], mean_std_dev['SLAI_mean'], label='SLAI Mean', color='green')
    
    # Adding shaded area for SLAI standard deviation
    plt.fill_between(mean_std_dev['DOY'], 
                     mean_std_dev['SLAI_mean'] - mean_std_dev['SLAI_std_dev'], 
                     mean_std_dev['SLAI_mean'] + mean_std_dev['SLAI_std_dev'], 
                     color='green', alpha=0.2)
    
    #plt.xticks(np.arange(120, 230, 20))
    #plt.yticks(np.arange(0, 5.5, 1))
    plt.xlabel('Day of year')
    plt.ylabel('LAI (m$^2$ m$^{-2}$)')
    plt.legend()

    if plot_file_path:
        directory = os.path.dirname(plot_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(plot_file_path)
        print(f"Plot saved to {plot_file_path}")
    plt.show()

#-- Function to map yield data --
def mapping_yield(map_dir, cyield, rows, columns, map_extent, image_extent, map_file_path = None):  
    fig, ax = plt.subplots(subplot_kw={'projection': ccrs.PlateCarree()})
    ax.set_extent(map_extent, crs=ccrs.PlateCarree())  # extent as [lon_min, lon_max, lat_min, lat_max]

    # Add continents or shape files from a directory
    shape_feature = ShapelyFeature(Reader(map_dir).geometries(),
                                   ccrs.PlateCarree(), edgecolor='black')
    ax.add_feature(shape_feature, facecolor='none')

    # Hide the default map frame (spines)
    try:
        ax.spines['geo'].set_visible(False)  # Newer versions
    except AttributeError:
        ax.outline_patch.set_visible(False)  # Older versions
      
    # Plot data with geographical extent
    if cyield.size == rows * columns:
        yield_img = ax.imshow(cyield.reshape((columns, rows)), origin='upper',
                            extent=image_extent, transform=ccrs.PlateCarree(),
                            interpolation='nearest', cmap='viridis')
        cbar = plt.colorbar(yield_img, ax=ax, orientation='vertical',
                       fraction=0.046, pad=0.04, aspect=20, shrink=0.8)
        cbar.set_label('Yield (ton $ha^{-1}$)', fontsize=12, labelpad=5) # Set the title of the color bar
    else:
        raise ValueError("yield_img size does not match the product of rows and cols")
    plt.title('Yield Mapping')       
   
    if map_file_path:
        directory = os.path.dirname(map_file_path)
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        plt.savefig(map_file_path)
        print(f"Map saved to {map_file_path}")
    plt.show()
 
def out_file_cG(outf,para0,cyield,RSCMProcess):
    sDOY = RSCMProcess['DOY']
    GDD  = RSCMProcess['GDD']
    OLAI = RSCMProcess['OLAI']
    LAI  = RSCMProcess['LAI']
    AGDM = RSCMProcess['AGDM']
    NPP  = RSCMProcess['NPP']
    GPP  = RSCMProcess['GPP']
    ET   = RSCMProcess['ETc']
    PGP  = RSCMProcess['PGP']
    nrecords = RSCMProcess.shape[0]
    
    outf.write('The parameter values used or converged in the simulated result:\n')
    outf.write('{0:>12} {1:9.5}\n'.format('Initial LAI  =',para0[3])) # L0
    outf.write('{0:>13} {1:9.5}\n'.format('Parameter a =',para0[0])) # para a
    outf.write('{0:>13} {1:9.5}\n'.format('Parameter b =',para0[1])) # para b
    outf.write('{0:>13} {1:9.5}\n'.format('Parameter c =',para0[2])) # para c
    outf.write('\n')
        
    cyield = cyield * 10 # Convert to g/m2 to kg/ha
    outf.write('Estimated grain yield (kg/ha) :\n')
    outf.write('{0:7.1f}\n'.format(cyield))
    outf.write('\n')
    outf.write('{0:>3} {1:>4} {2:>5} {3:>6} {4:>6} {5:>7} {6:>6} {7:>6} {8:>6} {9:>6}\n'.
    format('DOY','DAP','GDD','OLAI','LAI','AGDM','NPP','GPP','ETc','PGP'))
    for i in range(nrecords):
        pfs = '{0:>3.0f}{1:>4d}{2:>7.1f}{3:>7.2f}{4:>7.2f}{5:>8.1f}{6:>7.2f}{7:>7.2f}{8:>7.2f}{9:>7.1f}\n'
        outf.write(pfs.format(sDOY[i],i,GDD[i],OLAI[i],LAI[i],AGDM[i],NPP[i],GPP[i],ET[i],PGP[i]))

def save_out_pixgro(output_dir,RSCMProcess,cyield,paraout):
    output_path = f"{output_dir}/"
    os.makedirs(output_path, exist_ok=True)  # Directory to save the files
    
    # Save arrays as binary files
    RSCMProcess.tofile(os.path.join(output_path, 'RSCMProcess.bin'))
    cyield.tofile(os.path.join(output_path, 'cyield.bin'))
    paraout.tofile(os.path.join(output_path, 'paraout.bin'))
    
    # Save arrays as npy files
    np.save(os.path.join(output_path, 'RSCMProcess.npy'), RSCMProcess)
    np.save(os.path.join(output_path, 'cyield.npy'), cyield)
    np.save(os.path.join(output_path, 'paraout.npy'), paraout)

def save_out_pixgro_summary(outf, RSCMProcess, cyield):
    reshaped_RSCMProcess = RSCMProcess.reshape(-1, 13)
    df = pandas.DataFrame(reshaped_RSCMProcess)
    df.columns = ['DOY','DAP','Tmax','Tmin','Rad','GDD','OLAI','SLAI','AGDM','NPP','GPP','ETc','PGP'] ### 24.08.11 ###

    # Calculating mean values for relevant columns grouped by DOY
    mean_values = df.groupby('DOY')[['OLAI', 'SLAI', 'AGDM', 'NPP', 'GPP', 'ETc', 'PGP']].mean().reset_index()
    
    # Calculating standard deviation values for OLAI, SLAI, and AGDM grouped by DOY
    std_dev_values = df.groupby('DOY')[['OLAI', 'SLAI', 'AGDM']].std().reset_index()
    
    meanYield = np.nanmean(cyield) #* 10 # Convert to kg/ha
    stdYield = np.nanstd(cyield) #* 10 # Convert to kg/ha
    maxYield = np.nanmax(cyield) #* 10 # Convert to kg/ha
    minYield = np.nanmin(cyield) #* 10 # Convert to kg/ha
    
    pfs = 'Mean yield (ton/ha):{0:7.3f}  std:{1:5.3f}\n'
    outf.write(pfs.format(meanYield, stdYield))
    outf.write(' Max yield (ton/ha):{0:7.3f}\n'.format(maxYield))
    outf.write(' Min yield (ton/ha):{0:7.3f}\n'.format(minYield))
        
    outf.write('\n')
    outf.write('{0:>3}{1:>6}{2:>10}{3:>6}{4:>10}{5:>9}{6:>10}{7:>6}{8:>6}{9:>6}{10:>6}\n'.
               format('DOY','OLAI','Std_OLAI','SLAI','Std_SLAI','AGDM','Std_AGDM','NPP','GPP','ETc','PGP'))

    for i in range(len(mean_values)):
        pfs = '{0:>3.0f}{1:>6.1f}{2:10.2f}{3:>6.1f}{4:>10.2f}{5:>9.1f}{6:>10.2f}{7:>6.1f}{8:>6.1f}{9:>6.1f}{10:>6.1f}\n'
        outf.write(pfs.format(mean_values.loc[i, 'DOY'], 
                              mean_values.loc[i, 'OLAI'], std_dev_values.loc[i, 'OLAI'],
                              mean_values.loc[i, 'SLAI'], std_dev_values.loc[i, 'SLAI'],
                              mean_values.loc[i, 'AGDM'], std_dev_values.loc[i, 'AGDM'],
                              mean_values.loc[i, 'NPP'], mean_values.loc[i, 'GPP'],
                              mean_values.loc[i, 'ETc'], mean_values.loc[i, 'PGP']))


##########################################################################################################
# Part V: Python routines that read the data from the files.
# read_binary_data:
#   Read a map containing the values of a variable pixel by pixel from a binary data file.
# read_binary_data_multiple:
#   Read maps containing the values of a variable pixel by pixel from a binary data files. One file for one date.
# retrieve_RSCMProcessOne:
#   Inputs: paraout,cyield,RSCMProcess for multiple pixels
#   Inputs: current, the pixel that the user would like to retrieve
# create_3D_data_from_one_pixel:
#   The inputs into RSCM_LAI and RSCM_VI must include the dimension "pixel".
#   If the data of only one pixel is inputted, the users are required to create the data with extra dimension using create_3D_data_from_one_pixel
#   Input: wx_data_2d,obs_d_2d,start, start for start planting date
##########################################################################################################

def read_binary_data(file_path, data_type, npixels, data_start):
    data_size = np.dtype(data_type).itemsize
    try:
        data = np.fromfile(file_path, dtype=data_type, count=npixels, offset=data_start * data_size)
    except Exception as e:
        raise IOError(f"An error occurred while reading the file: {e}")
    if data.size != npixels:
        raise ValueError(f"Expected to read {npixels} elements, but got {data.size}.")
    return data

def read_binary_data_multiple(file_path_1, file_path_2, data_type, npixels, data_start):
    fnames = glob.glob(file_path_1 + '*' + file_path_2)
    nfiles = len(fnames)

    index = np.zeros(nfiles)
    index_arr = np.zeros((npixels, nfiles))
    data_arr = np.full((npixels, nfiles), np.nan)

    for i in range(nfiles):
        fn = fnames[i]
        begin = len(file_path_1)
        end = fn.index(file_path_2)
        index[i] = int(fn[begin:end])
        index_arr[:,i] = int(fn[begin:end])
        data_arr[:,i] = read_binary_data(fn, data_type, npixels, data_start)

    ind = np.argsort(index)
    return nfiles, index_arr[:,ind], data_arr[:,ind]

def retrieve_RSCMProcessOne(paraout,cyield,RSCMProcess,current):
    RSCMProcessOne = pandas.DataFrame({'DOY':RSCMProcess[current,:,0],
                                        'DAP':RSCMProcess[current,:,1],
                                        'RAD':RSCMProcess[current,:,2],
                                        'Tmax':RSCMProcess[current,:,3],
                                        'Tmin':RSCMProcess[current,:,4],
                                        'GDD':RSCMProcess[current,:,5],
                                        'OLAI':RSCMProcess[current,:,6],
                                        'LAI':RSCMProcess[current,:,7],
                                        'AGDM':RSCMProcess[current,:,8],
                                        'NPP':RSCMProcess[current,:,9],
                                        'GPP':RSCMProcess[current,:,10],
                                        'ETc':RSCMProcess[current,:,11],
                                        'PGP':RSCMProcess[current,:,12]})
    paraoutOne = paraout[current,:]
    cyieldOne = cyield[current]
    return paraoutOne,cyieldOne,RSCMProcessOne

def create_3D_data_from_one_pixel(wx_data_2d,obs_d_2d,start):
    # Convert to 3d array with the first dimension = 1
    obs_d = np.zeros((1,)+obs_d_2d.shape)
    obs_d[0,:,:] = obs_d_2d
    wx_data = np.zeros((1,)+wx_data_2d.shape)
    wx_data[0,:,:] = wx_data_2d
    sub_paddyFields = np.zeros(1).astype(np.int32)
    sub_plantingDate = np.zeros(1).astype(np.int32)
    sub_paddyFields[0] = 1
    sub_plantingDate[0] = start
    return wx_data,obs_d,sub_paddyFields,sub_plantingDate
