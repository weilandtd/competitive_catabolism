from skimpy.analysis.oracle.load_pytfa_solution import load_fluxes, load_concentrations,\
    load_equilibrium_constants
from skimpy.core.parameters import ParameterValuePopulation, load_parameter_population
from pytfa.io.json import load_json_model
from skimpy.io.yaml import load_yaml_model

# Compute Nv 
from skimpy.analysis.ode.utils import make_flux_fun
from skimpy.utils.namespace import QSSA
from skimpy.utils.general import get_stoichiometry

# Parameter sampling
from skimpy.sampling.simple_parameter_sampler import SimpleParameterSampler

import pandas as pd 
import numpy as np
import os

from tqdm import tqdm

if __name__ == '__main__':

    # Load the tfa model 
    model_file = 'reduced_model_ETC_core_20240816-155234_continuous.json'
    tmodel = load_json_model(model_file)
    #sol = tmodel.optimize()

    # Reload and prepare the model
    kmodel = load_yaml_model(model_file.replace("_continuous.json", "_kinetic_curated.yml"))

    faraday_const = 23.061 # kcal / mol / mV
    RT = tmodel.RT # kcal /mol
    delta_psi_scaled = 150/1000 * faraday_const / RT # mV * F / RT 

    # These are common for the sodium gneralize the assingemnt 
    # Delta ion deltaH = 5.309573444801986e-05

    for parameter in kmodel.parameters.values(): 
        if 'charge_ion_MPM_na1_m_na1_c' in str(parameter.symbol):
            parameter.value = 1
            print(parameter.symbol , parameter.value)
        if 'delta_psi_scaled_MPM_na1_m_na1_c' in str(parameter.symbol):
            parameter.value = delta_psi_scaled
            print(parameter.symbol , parameter.value)
        if 'delta_ion_concentration_MPM_na1_m_na1_c' in str(parameter.symbol):
            parameter.value = -2
            print(parameter.symbol , parameter.value)


    # Parametrize the membrane potential modifiers
    # Charge export from mitochondria
    # Pos
    kmodel.parameters.charge_transport_MPM_na1_m_na1_c_NADH2_u10mi.value = -4 # 4 H+ to the outside (Complex I)
    kmodel.parameters.charge_transport_MPM_na1_m_na1_c_CYOOm2i.value = -4 # 4 H+ to the outside (Complex IV)
    kmodel.parameters.charge_transport_MPM_na1_m_na1_c_CYOR_u10mi.value = -4 # 4 H+ to the outside (Complex III)
    # Neg
    kmodel.parameters.charge_transport_MPM_na1_m_na1_c_ATPtm.value = 1 # -1 to the outside 
    
    # Charge import into mitochondria
    # Pos
    kmodel.parameters.charge_transport_MPM_na1_m_na1_c_ASPGLUm.value = 1 # 1 H+ to the inside
    kmodel.parameters.charge_transport_MPM_na1_m_na1_c_ATPS4mi.value = 3 # 3 H+ to the inside

    # Compile the jacobian expressions
    NCPU = 8
    kmodel.repair()
    kmodel.prepare()
    kmodel.compile_jacobian(ncpu=NCPU)


    # Initiate a parameter sampler
    params = SimpleParameterSampler.Parameters(n_samples = 10)
    sampler = SimpleParameterSampler(params)


    # Load TFA samples 
    tfa_sample_file = 'reduced_model_ETC_core_20240816-155234_tfa_sampling.csv'
    tfa_samples = pd.read_csv(tfa_sample_file)

    # Scaling parameters
    CONCENTRATION_SCALING = 1e3 # 1 mol to 1 mmol
    TIME_SCALING = 1.0 # 1min
    DENSITY = 1200 # g/L 
    GDW_GWW_RATIO = 1.0 # Fluxes are in gWW

    # To test how close to zero the dxdt is
    flux_scaling_factor = 1e-6 / (GDW_GWW_RATIO / DENSITY) \
                          * CONCENTRATION_SCALING \
                          / TIME_SCALING
    
    # Add additional fluxes and concentration to model insulin turnover
    # Turnover of inusnlin is about 5 min 
    
    tfa_samples['Insulin_secretion'] = 1/5 / flux_scaling_factor
    tfa_samples['Insulin_degradation'] = 1/5 / flux_scaling_factor
    tfa_samples['insulin_e'] = 1e-3 
    

    additional_fluxes = ['Insulin_secretion', 'Insulin_degradation']
    additional_concentrations = ['insulin_e']


    
    S = get_stoichiometry(kmodel, kmodel.reactants).todense()

    lambda_max_all = []
    lambda_min_all = []

    # Make a new directory for the output
    os.makedirs(tfa_sample_file.replace(".csv",''), exist_ok=True)

    path_for_output = './'+tfa_sample_file.replace(".csv",'')+'/paramter_pop_{}.h5'

    flux_fun = make_flux_fun(kmodel, QSSA)

    # Fix insulin modifer activation constants
    kmodel.reactions.GLCt1r.parameters.k_activation_HAMI_insulin_e_GLCt1r.bounds = (5, 5.001)
    kmodel.reactions.PFK.parameters.k_activation_HAMI_insulin_e_PFK.bounds = (5, 5.001)

    # Integrate brenda enyzme data into the model
    # PGI
    PGI_f6p_c_KM = 50e-3 # mM
    PGI_g6p_c_KM = 0.5 # mM
    kmodel.reactions.PGI.parameters.km_product.bounds = (PGI_f6p_c_KM * 0.8 , PGI_f6p_c_KM * 1.2)
    kmodel.reactions.PGI.parameters.km_substrate.bounds = (PGI_g6p_c_KM * 0.8 , PGI_g6p_c_KM * 1.2)

    # LDH_L 
    LDH_L_pyruvate_c_KM = 0.05 # mM
    LDH_L_lac_L_c_KM = 5 # mM
    kmodel.reactions.LDH_L.parameters.km_product2.bounds = (LDH_L_pyruvate_c_KM * 0.8 , LDH_L_pyruvate_c_KM * 1.2)
    kmodel.reactions.LDH_L.parameters.km_substrate2.bounds = (LDH_L_lac_L_c_KM * 0.8 , LDH_L_lac_L_c_KM * 1.2)

    # NDPK1m 
    NDPK1m_atp_c_KM = 1.0 # mM
    NDPK1m_gdp_c_KM = 0.1 # mM
    NDPK1m_adp_c_KM = 0.1 # mM
    NDPK1m_gtp_c_KM = 1.0 # mM
    kmodel.reactions.NDPK1m.parameters.km_product1.bounds = (NDPK1m_gtp_c_KM * 0.8 , NDPK1m_gtp_c_KM * 1.2)
    kmodel.reactions.NDPK1m.parameters.km_substrate1.bounds = (NDPK1m_gdp_c_KM * 0.8 , NDPK1m_gdp_c_KM * 1.2)
    kmodel.reactions.NDPK1m.parameters.km_product2.bounds = (NDPK1m_adp_c_KM * 0.8 , NDPK1m_adp_c_KM * 1.2)
    kmodel.reactions.NDPK1m.parameters.km_substrate2.bounds = (NDPK1m_atp_c_KM * 0.8 , NDPK1m_atp_c_KM * 1.2)

    # GLUT4 KM
    kmodel.reactions.GLCt1r.parameters.km_substrate.bounds = (4.9, 5.1)
    kmodel.reactions.GLCt1r.parameters.km_product.bounds = (4.9, 5.1)

    # Hexokinase HK1 
    # https://www.brenda-enzymes.org/enzyme.php?ecno=2.7.1.1#KM%20VALUE%20[mM]
    kmodel.reactions.HEX1.parameters.km_substrate1.bounds = (0.1, 0.4) # ATP Brenda
    kmodel.reactions.HEX1.parameters.km_substrate2.bounds = (0.1, 1.0) # Glucose

    # PFK 
    # HAS actual hill kinetics
    # https://onlinelibrary.wiley.com/doi/10.1002/jcb.24039
    # https://febs.onlinelibrary.wiley.com/doi/10.1016/j.febslet.2007.05.059
    kmodel.reactions.PFK.parameters.km_substrate1.bounds = (0.03, 0.04) # ATP Brenda
    kmodel.reactions.PFK.parameters.km_substrate2.bounds = (0.07, 0.09) # F6P


    # LDH pmt-coa inhibition
    # https://www.science.org/doi/10.1126/science.abm3452?url_ver=Z39.88-2003&rfr_id=ori:rid:crossref.org&rfr_dat=cr_pub%20%200pubmed#sec-3
    # Arround 1 uM
    kmodel.reactions.LDH_L.parameters.k_inhibition_IM_pmtcoa_c_LDH_L.bounds = (0.8e-3 , 1.2e-3)

    # # BDHm 
    # # https://www.brenda-enzymes.org/all_enzymes.php?ecno=1.1.1.30&table=KM_Value#TAB
    # kmodel.reactions.BDHm.parameters.km_substrate1.bounds = (0.05, 0.07) # NAD
    # kmodel.reactions.BDHm.parameters.km_substrate2.bounds = (1.0, 5.0) # 3BHb -> GUESS
    # kmodel.reactions.BDHm.parameters.km_product1.bounds = (0.02, 0.03) # NADH
    # kmodel.reactions.BDHm.parameters.km_product2.bounds = (0.25, 0.35) # AcAc

    # # OCOAT1m/ SCOT1
    # # https://www.brenda-enzymes.org/enzyme.php?ecno=2.8.3.5
    # kmodel.reactions.OCOAT1m.parameters.km_substrate1.bounds = (0.1, 0.3) # SucCoa
    # kmodel.reactions.OCOAT1m.parameters.km_substrate2.bounds = (0.05, 0.1) # ACAC 0.07
    # kmodel.reactions.OCOAT1m.parameters.km_product1.bounds = (0.02, 0.03) # Succ
    # kmodel.reactions.OCOAT1m.parameters.km_product2.bounds = (0.04, 0.06) # AcAcCoa

    # #Acetyl CoA C acetyltransferase mitochondrial
    # # https://www.brenda-enzymes.org/enzyme.php?ecno=2.3.1.9#KM%20VALUE%20[mM]
    # kmodel.reactions.ACACT1rm.parameters.km_substrate1.bounds = (0.03, 0.031) # AcCoa
    # kmodel.reactions.ACACT1rm.parameters.km_substrate2.bounds = (0.03, 0.031) # AcCoa
    # kmodel.reactions.ACACT1rm.parameters.km_product1.bounds = (0.01, 0.03) # CoA
    # kmodel.reactions.ACACT1rm.parameters.km_product2.bounds = (4e-3, 5e-3) # AcAcCoa

    # # # # G3PD1 - Glycerol 3-phosphate dehydrogenase (NAD+)
    # # # # https://www.brenda-enzymes.org/enzyme.php?ecno=1.1.1.8#KM%20VALUE%20[mM]
    # kmodel.reactions.G3PD1.parameters.km_product1.bounds = (0.002, 0.010) # NADH
    # kmodel.reactions.G3PD1.parameters.km_product2.bounds = (0.02, 0.03) # DAHP
    # kmodel.reactions.G3PD1.parameters.km_substrate1.bounds = (0.01, 0.04) # NAD+
    # kmodel.reactions.G3PD1.parameters.km_substrate2.bounds = (1.0,2.0) # Glycerol-3P

    # # # r0205 - GPD2 - Glycerol-3-phosphate dehydrogenase (Quinone)
    # # # Brenda 
    # kmodel.reactions.r0205.parameters.km_substrate2.bounds = (0.5, 10) # Glycerol-3P

    # TPI - le
    # https://www.brenda-enzymes.org/enzyme.php?ecno=5.3.1.1#KM%20VALUE%20[mM]
    # kmodel.reactions.TPI.parameters.km_substrate.bounds = (1.0, 2.0) #DHAP
    # kmodel.reactions.TPI.parameters.km_product.bounds = (0.25, 1.0) # GAP


    # NOTE DW -> TRANSPORTERS SHOULD HAVE SAME KM for substrate and product pairs
    
    for i, sample in tqdm(tfa_samples.iterrows()):
        # Load fluxes and concentrations
        fluxes = load_fluxes(sample, tmodel, kmodel,
                                density=DENSITY,
                                ratio_gdw_gww=GDW_GWW_RATIO,
                                concentration_scaling=CONCENTRATION_SCALING,
                                time_scaling=TIME_SCALING,
                                xmol_in_flux=1e-6,
                                additional_fluxes=additional_fluxes)
        
        concentrations = load_concentrations(sample, tmodel, kmodel,
                                             concentration_scaling=CONCENTRATION_SCALING,
                                             additional_concentrations=additional_concentrations)
        
        # ATP dissipation should be saturated KM << [atp_c]
        atp_c = concentrations['atp_c']
        kmodel.reactions.cyt_atp2adp.parameters.km_substrate1.bounds = (atp_c*1e-4, atp_c*1e-3)

        # FATP1t - Fatty acid transport unsaturated
        hdca_e = concentrations['hdca_e']
        kmodel.reactions.FATP1t.parameters.km_substrate1.bounds = (hdca_e*4.9, hdca_e*5.0)
        kmodel.reactions.FATP1t.parameters.km_product1.bounds = (hdca_e*4.9, hdca_e*5.0)

        # # ASPGLUm unsaturated - This is a commited step in the malate aspartate shuttle
        # # Substrate control of asp and glu 
        asp_L_c = concentrations['asp_L_c'] 
        glu_L_c = concentrations['glu_L_c']
        kmodel.reactions.ASPGLUm.parameters.km_substrate1.bounds = (glu_L_c*4.9, glu_L_c*5)
        kmodel.reactions.ASPGLUm.parameters.km_substrate2.bounds = (asp_L_c*4.9, asp_L_c*5)
        kmodel.reactions.ASPGLUm.parameters.km_product1.bounds = (glu_L_c*4.9, glu_L_c*5)
        kmodel.reactions.ASPGLUm.parameters.km_product2.bounds = (asp_L_c*4.9, asp_L_c*5)

        # BDH and OCOAT1m unsaturated in their ketones effects (mass action effect)
        # AcAc and 3HB
        acac_m = concentrations['acac_m']
        bhb_m = concentrations['bhb_m']
        kmodel.reactions.BDHm.parameters.km_substrate2.bounds = (bhb_m*5, bhb_m*10)
        kmodel.reactions.OCOAT1m.parameters.km_substrate2.bounds = (acac_m*5, acac_m*10)

        # FACOALm - Fatty acyl-CoA ligase 
        hdca_c = concentrations['hdca_c']
        kmodel.reactions.FACOAL160i.parameters.km_substrate1.bounds = (hdca_c*10, hdca_c*100)

        # Fetch equilibrium constants
        load_equilibrium_constants(sample, tmodel, kmodel,
                                concentration_scaling=CONCENTRATION_SCALING,
                                in_place=True)
        
        # Generate sampels and fetch slowest and fastest eigenvalues
        params, lamda_max, lamda_min = sampler.sample(kmodel, fluxes, concentrations,
                                                        only_stable=False,
                                                        min_max_eigenvalues=True,
                                                        seed=i+100)
        
        # Test Nv = 0 
        params_population = ParameterValuePopulation(params, kmodel)

        # Test if the resulting sets are NV=0
        fluxes_1 = flux_fun(concentrations, parameters=params_population['0'])
        fluxes_1 = pd.Series(fluxes_1)
        dxdt = S.dot(fluxes_1[kmodel.reactions])
        if np.any(abs(dxdt) > 1e-9*flux_scaling_factor):
            raise RuntimeError('dxdt for idx {} not equal to 0'
                               .format(np.where(abs(dxdt) > 1e-9*flux_scaling_factor)))
        

        lambda_max_all.append(pd.DataFrame(lamda_max))
        lambda_min_all.append(pd.DataFrame(lamda_min))

        params_population.save(path_for_output.format(i))

    # Process df and save dataframe
    lambda_max_all = pd.concat(lambda_max_all, axis=1)
    lambda_min_all = pd.concat(lambda_min_all, axis=1)

    # Save the eigenvalue distributino
    lambda_max_all.to_csv(tfa_sample_file.replace(".csv","_lambda_max.csv"))
    lambda_min_all.to_csv(tfa_sample_file.replace(".csv","_lambda_min.csv"))

    """
    Prune parameters based on the time scales
    """

    MAX_EIGENVALUES = -1/60 # Faster than an hour response time 
    
    # Prune parameter based on eigenvalues
    is_selected = (lambda_max_all < MAX_EIGENVALUES )
    is_selected.columns = range(lambda_max_all.shape[1])

    fast_parameters = []
    fast_index = []

    for i, row in is_selected.T.iterrows():
        if any(row):
            fast_models = np.where(np.array(row))[0]
            # Load the respective solutions
            parameter_population = load_parameter_population(path_for_output.format(i))
            fast_parameters.extend([parameter_population._data[k] for k in fast_models])
            fast_index.extend(["{},{}".format(i,k) for k in fast_models])

    # Generate a parameter population file
    parameter_population = ParameterValuePopulation(fast_parameters,
                                               kmodel=kmodel,
                                               index=fast_index)
    
    parameter_population.save( tfa_sample_file.replace(".csv",'_pruned_parameters.hdf5'))



    # Modal analysis
    from skimpy.analysis.modal import modal_matrix
    from skimpy.viz.modal import plot_modal_matrix
    import random

    # Pic a random parameter set and plot the modal matrix
    index = random.choice(list(parameter_population._index.keys()))
    # Print the index
    print(f"Will perform modal analysis on index: {index}")

    sample = tfa_samples.iloc[int(index.split(',')[0])]
    concentrations = load_concentrations(sample, tmodel, kmodel,
                                                concentration_scaling=CONCENTRATION_SCALING,
                                                additional_concentrations=additional_concentrations)
    parameter_values = parameter_population[index]

    kmodel.prepare()
    kmodel.compile_jacobian(sim_type=QSSA,ncpu=8)
    M = modal_matrix(kmodel,concentrations,parameter_values)

    plot_modal_matrix(M,filename='modal_matrix.html',
                      width=800, height=600,
                      clustered=True,
                      backend='svg',
                      )
    
    # Make a histogram of the slow eigenvalues
    import matplotlib.pyplot as plt
    bins = np.linspace(0, 240, 100)
    plt.hist( -1/np.real(lambda_max_all.values.flatten()), bins=bins)
    plt.show()
