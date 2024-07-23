import numpy as np
from scipy.integrate import odeint

###################################################
# Model of whole body fasted catabolism
###################################################


# FIXED RATIOS (Energy per molecule rel to glucose)
nG = 1/30
nL = 14/30
nF = 108/30
nK = 22.5/30

# For Protein and Amino acids contribution 
nP = 22.1/30 
cP = 5.08

# Time constants
# From volume of distribution mesrurements
# TODO Put in correct values for the realative circulatory fluxes 
tauG = 14.0  * 0.05
tauL = 5.0   * 0.15
tauF = 10.0  * 0.7*3/16/2*3
tauK = 5.0 * 0.05

tauI = 1.0 # Insulin is fast

parameter_names = ["vE", "k", "Imax", "C", "KI_lipo", "KA_glut4", "KI_GY", "KIL", "KIK", "KF", "KG","omega", 
                   "lam", "gamma", "beta", "kappa", "alpha", "VR", "VFK", "KFK", "VLG", "KL",
                   "v0", "rho", "v_in_I", "v_in_L", "v_in_G", "v_in_F", "v_in_K", 
                   "with_LI", "with_SI", "with_GI", "with_LL", "with_LK"]


def change_parameters(p,e=[1.0,],ix=["vE",]):
    p_c = p.copy()
    for this_e, this_ix in zip(e,ix):
        i = parameter_names.index(this_ix)
        p_c[i] = this_e
        
    return p_c


def fluxes(x,A,p):
    L,G,F,K,I = x
    
    vE, k, Imax, C, KI_lipo, KA_glut4, KI_GY, KIL, KIK, KF, KG, omega, \
    lam, gamma, beta, kappa, alpha, VR, VFK, KFK, VLG, KL, \
    v0, rho, v_in_I, v_in_L, v_in_G, v_in_F, v_in_K, \
    with_LI, with_SI, with_GI, with_LL, with_LK = p
    
    # Insulin
    vI = ( Imax * abs(G)**k / (abs(G)**k + C**k) - I)/ tauI + v_in_I
    I0 = abs(1.0)**k / (abs(1.0)**k + C**k)

    # Isulin action 
    if with_LI:
        LI = 1.0 - rho * (KI_lipo + 1.0) * I / (I + KI_lipo)
    else:
        LI = 1.0 - rho * (KI_lipo + 1.0) * I0 / (I0 + KI_lipo)
        
    if with_SI:
        SI =  I / (I + KA_glut4) / ( I0 / (I0 + KA_glut4))
    else:
        SI = 1.0
        
    if with_GI:
        GI = (1.0 - (KI_GY + 1.0) * I / (I + KI_GY)) \
             / (1.0 - (KI_GY + 1.0) * I0 / (I0 + KI_GY))
    else:
        GI = 1.0
            
   
    # Lactate action
    if with_LL:
        LL = 1.0/(L/KIL+1.0) * (1.0/KIL+1.0)
    else:
        LL = 1.0
        
    # Ketone action
    if with_LK:
        LK = 1.0/(K/KIK+1.0) * 2.0
    else:
        LK = 1.0

    M = vE/(nG*omega*G*SI + nL*lam*L + gamma*G*SI + nF*beta*F + nK*kappa*K)
        
    vGL = omega*G*SI*M
    vG = gamma*G*SI*M
    vL = lam*L*M
    vF = beta*F*M
    vK = kappa*K*M
    
    vA = alpha*A * LI * LL * LK 
    vR = VR * (G/KG * F/KF)/ (1 + G/KG + F/KF)
    
    vFK = VFK * F/(F+KFK)
    vLG = VLG * L/(L+KL)
    
    v0 = v0 * GI
    vLG = vLG * GI
    
    return np.array([vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR,
                     v_in_L, v_in_G, v_in_F, v_in_K, vI])
    

def equation(x,A,p):
    vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, \
    v_in_L, v_in_G, v_in_F, v_in_K, vI = fluxes(x,A,p) 

    dLdt = 2.0*vGL - 2.0*vLG - vL + v_in_L
    dGdt = v0 + 1/2*(vA -vR) + vLG - vGL - vG + v_in_G
    dFdt = 3.0*(vA-vR) - vF - vFK + v_in_F
    dKdt = 4.0*vFK - vK + v_in_K
    dIdt = vI
    # Scale the dynamic equation to relative to glucose 
    # concentration
    return [dLdt/tauL, dGdt/tauG, dFdt/tauF, dKdt/tauK, dIdt]
    

###################################################
# Parameter constraitns
###################################################


def ref_parameters():
    # Parameters 
    vE = 1.0

    # Insulin secretion
    k = 3.4
    C = 2.3
    
    #Ref. insulin
    Imax = 1.0
    I0 = abs(1.0)**k / (abs(1.0)**k + C**k)

    rho = 1.0
    KI_lipo = I0
    LI = 1.0 - rho * (KI_lipo + 1.0) * I0 / (I0 + KI_lipo)
                   
    KA_glut4 = I0 
    SI = 1.0
    
    KIL = 1.0
    KIK = 1.0
    KI_GY = I0

    # Reference fluxes
    vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR, vP, vCO2 = reference_fluxes()
                             
    # Calculate parmeters 
    omega = vGL/SI
    lam = vL
    gamma = vG/SI
    beta = vF
    kappa = vK
    alpha = vA/LI
    
    # Resterification
    KF = 0.1
    KG = 0.1
    VR = vR / (1/KF * 1/KG / (1 + 1/KF + 1/KG ))
    
    # Ketogenesis
    KFK = 0.1
    VFK = vFK /(1/(1+KFK))

    # Gluconeogenesis
    KL = 0.1
    VLG = vLG /(1/(1+KL))
    
    # Parameters to manipulate the model
    v_in_I = 0.0 #Insulin infusion
    v_in_L = 0.0 # lactate infusion
    v_in_G = 0.0 # glucose infusion
    v_in_F = 0.0 # fatty-accid infusion
    v_in_K = 0.0 # Ketone infusion   
    
    with_LI = True
    with_SI = True
    with_GI = False
    with_LL = True
    with_LK = True

    return [vE, k, Imax, C, KI_lipo, KA_glut4, KI_GY, KIL, KIK, KF, KG, omega, 
            lam, gamma, beta, kappa, alpha, VR, VFK, KFK, VLG, KL,
            v0, rho, v_in_I, v_in_L, v_in_G, v_in_F, v_in_K, 
            with_LI, with_SI, with_GI, with_LL, with_LK]


def reference_fluxes(direct_contributions = [0.04, 0.24, 0.5, 0.05], gng_ratio = 1.0, vE=1.0):
    # direct_contributions = [G,L,F,K]
    DG,DL,DF,DK = direct_contributions

    # Stoichiometry of the reactions

    # constraints N*v = 0 
    # fluxes =   [vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, vP, vCO2]
    g = np.array([ 0, -1,  0,  0, -1,   0,  1,   1, 0.5, -0.5,  0, 0]) # 0
    l = np.array([-1,  0,  0,  0,  2,   0, -2,   0,   0,    0,  0, 0]) # 0 
    f = np.array([0 ,  0, -1,  0,  0,  -1,  0,   0,   3,   -3,  0, 0]) # 0 
    k = np.array([0 ,  0,  0, -1,  0,   4,  0,   0,   0,    0,  0, 0]) # 0
    e = np.array([nL,  1, nF, nK, nG,   0,  0,   0,   0,    0, nP, 0]) # 1
    c = np.array([0.5, 1,16/6,4/6, 0,   0,  0,   0,   0,    0, cP,-1]) # 0

    # Direct contribution constraints
    dg = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, -DG])          #0
    dl = np.array([0.5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -DL])        #0
    df = np.array([0, 0, 16/6, 0, 0, 0, 0, 0, 0, 0, 0, -DF])       #0 
    dk = np.array([0, 0, 0, 4/6, 0, 0, 0, 0, 0, 0, 0, -DK])        #0

    # Reseterficication constraint
    r = np.array([0 ,  0, 0,  0,  0,  0,  0,   0,  2/3,  -1,  0, 0]) # 0 

    # Glucose / glycolysis constraint vGL = v0
    gl = np.array([ 0, 0,  0,  0, 0,   0,  gng_ratio,   -1, 0, 0,  0, 0]) # 0
 
    # Stack the constraints
    A = np.vstack([g,l,f,k,e,c,dg,dl,df,dk,r,gl])

    # Solve the linear system
    b = np.zeros(12)
    b[4] = vE

    x = np.linalg.solve(A,b)

    [vL, vG, vF, vK, vGL, vFK, vLG, v0, vA, vR, vP, vCO2] = x

    # Steady state Fluxes see Mathematica
    # vGL=0.32805
    # vLG=0.165296
    # v0=0.165296 
    # vL=0.325506
    # vG=0.0406883
    # vF=0.183097
    # vK=0.183097
    # vFK=0.0457743
    # vA=0.228872 
    # vR=0.152581

    return vL, vG, vF, vK, vGL, vFK,  vLG, v0, vA, vR, vP, vCO2