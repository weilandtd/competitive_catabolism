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


if __name__ == '__main__':

    # Load the tfa model 
    model_file = 'reduced_model_ETC_core_20240710-100629_continuous.json'
    tmodel = load_json_model(model_file)
    sol = tmodel.optimize()

    # Reload and prepare the model
    kmodel = load_yaml_model(model_file.replace("_continuous.json", "_kinetic_curated.yml"))

    # Compile the jacobian expressions
    NCPU = 12
    kmodel.prepare()
    kmodel.compile_jacobian(ncpu=NCPU)


    # Initiate a parameter sampler
    params = SimpleParameterSampler.Parameters(n_samples = 10)
    sampler = SimpleParameterSampler(params)


    # Load TFA samples 
    tfa_sample_file = 'reduced_model_ETC_core_20240710-100629_tfa_sampling.csv'
    tfa_samples = pd.read_csv(tfa_sample_file)


    # Scaling parameters
    CONCENTRATION_SCALING = 1e3 # 1 mol to 1 mmol
    TIME_SCALING = 1 # 1min
    DENSITY = 1200 # g/L 
    GDW_GWW_RATIO = 1.0 # Fluxes are in gWW

    # To test how close to zero the dxdt is
    flux_scaling_factor = 1e-3 / (GDW_GWW_RATIO / DENSITY) \
                          * CONCENTRATION_SCALING \
                          / TIME_SCALING
    
    S = get_stoichiometry(kmodel, kmodel.reactants).todense()

    lambda_max_all = []
    lambda_min_all = []

    # Make a new directory for the output
    os.makedirs(tfa_sample_file.replace(".csv",''), exist_ok=True)

    path_for_output = './'+tfa_sample_file.replace(".csv",'')+'/paramter_pop_{}.h5'

    flux_fun = make_flux_fun(kmodel, QSSA)


    for i, sample in tfa_samples.iterrows():
        # Load fluxes and concentrations
        fluxes = load_fluxes(sample, tmodel, kmodel,
                                density=DENSITY,
                                ratio_gdw_gww=GDW_GWW_RATIO,
                                concentration_scaling=CONCENTRATION_SCALING,
                                time_scaling=TIME_SCALING)
        concentrations = load_concentrations(sample, tmodel, kmodel,
                                                concentration_scaling=CONCENTRATION_SCALING)
        
        # ATP dissipation should be saturated KM << [atp_c]
        atp_c = concentrations['atp_c']
        kmodel.reactions.cyt_atp2adp.parameters.km_substrate1.bounds = (atp_c*1e-4, atp_c*1e-3)
        
        # Fetch equilibrium constants
        load_equilibrium_constants(sample, tmodel, kmodel,
                                concentration_scaling=CONCENTRATION_SCALING,
                                in_place=True)
        
        # Generate sampels and fetch slowest and fastest eigenvalues
        params, lamda_max, lamda_min = sampler.sample(kmodel, fluxes, concentrations,
                                                        only_stable=True,
                                                        min_max_eigenvalues=True)
        
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

    MAX_EIGENVALUES = -1.0    # 1/hr
    #MIN_EIGENVALUES = -1e13   

    # Prune parameter based on eigenvalues
    is_selected = (lambda_max_all < MAX_EIGENVALUES ) #& (lambda_min_all > MIN_EIGENVALUES )
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
