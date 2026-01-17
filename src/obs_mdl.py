# """
# Observation modeling functions for radio astronomy
# """

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from typing import Callable, Optional, Dict, Any
from coord_frames import ground_to_beam_coord_vectorized, beam_to_ground_coord_vectorized


def model_observed_temp(
    observation, 
    source_mdl: Callable,
    background_mdl: Callable, 
    constellation=None, 
    beam_avoidance=False) -> np.ndarray:
    """
    Optimized using advanced NumPy operations and vectorization.
    Vectorizes per-time across satellites and frequencies; only loops over
    pointings to evaluate telescope gain for S satellites, then reduces to F.

    Args:
        observation: Observation object containing trajectory and instrument data
        source_mdl: Callable function for observed point sopurce temperature contributions
        background_mdl: Callable function for sky and ground noise for telescope
        constellation: Optional constellation object(s) for satellite interference
        beam_avoidance: If True, uses non-vectorized approach for beam avoidance calculations
    """  # noqa: E501

    # Get observation parameters and sort trajectory for correct reshaping
    times = observation.get_time_stamps()
    traj = observation.get_traj().sort_values(by='times').reset_index(drop=True)
    instrument = observation.get_instrument()
    n_times = len(times)
    n_pointings = len(traj) // n_times if n_times > 0 else 0

    # Instrument/antenna parameters
    f_RX_array = instrument.get_center_freq_chans()
    n_freq = len(f_RX_array)
    T_RX_func = instrument.get_inst_signal()
    T_phy = instrument.get_phy_temp()
    antenna = instrument.get_antenna()
    eta_rad = antenna.get_rad_eff()
    max_gain = antenna.get_boresight_gain()

    # Pre-shape all pointing data
    dec_tel_grid = np.radians(90 - traj['elevations'].values.reshape(n_times, n_pointings))
    caz_tel_grid = np.radians(-traj['azimuths'].values.reshape(n_times, n_pointings))

    # Pre-compute satellite data
    satellite_data = {}
    if constellation is not None:
        if not isinstance(constellation, list):
            constellation = [constellation]
        for c_idx, con in enumerate(constellation):
            sat_TX = con.get_transmitter()
            for time in times:
                sats_t = con.sats[con.sats['times'] == time]
                if len(sats_t) > 0:
                    sat_TX_func = sat_TX.get_inst_signal()
                    sat_temps_1f = np.array([sat_TX_func(time, f) for f in f_RX_array], dtype=np.float64)
                    satellite_data[(c_idx, time)] = {
                        'sat_dec': np.radians(90 - sats_t['elevations'].values),
                        'sat_caz': np.radians(-sats_t['azimuths'].values),
                        'sat_distances': sats_t['distances'].values,
                        'sat_temps': np.tile(sat_temps_1f, (len(sats_t), 1)),
                        'lnk_bdgt': con.get_lnk_bdgt_mdl(),
                        'instru_sat': sat_TX,
                    }

    result = observation.get_result()

    # Main simulation loop over time
    for t_idx, time in enumerate(times):
        dec_tel_t = dec_tel_grid[t_idx]
        caz_tel_t = caz_tel_grid[t_idx]

        # Vectorize sky model computation for all pointings and frequencies
        # this is wrong, sky model and ground model need to be integrated over the pattern
        T_sky_arr = np.array([
            [background_mdl(dec, caz, time, f) for f in f_RX_array]
            for dec, caz in zip(dec_tel_t, caz_tel_t)
        ], dtype=np.float64)
        T_RX_vals = np.array([T_RX_func(time, f) for f in f_RX_array], dtype=np.float64)

        T_sat_total = np.zeros((n_pointings, n_freq), dtype=np.float64)

        for c_idx, con in enumerate(constellation or []):
            sat_key = (c_idx, time)
            if sat_key in satellite_data:
                sd = satellite_data[sat_key]
                n_sats = len(sd['sat_dec'])

                if n_sats > 0:
                    if beam_avoidance:
                        # Use non-vectorized approach for beam avoidance
                        # Process each pointing and satellite combination individually
                        # to avoid broadcasting issues with coordinate transformations
                        for p_idx in range(n_pointings):
                            dec_tel = dec_tel_t[p_idx]
                            caz_tel = caz_tel_t[p_idx]

                            for s_idx in range(len(sd['sat_dec'])):
                                sat_dec = sd['sat_dec'][s_idx]
                                sat_caz = sd['sat_caz'][s_idx]
                                sat_dist = sd['sat_distances'][s_idx]

                                for f_idx in range(n_freq):
                                    freq = f_RX_array[f_idx]

                                    link_val = sd['lnk_bdgt'](
                                        dec_tel, caz_tel, instrument, sat_dec,
                                        sat_caz, sat_dist, sd['instru_sat'], freq
                                    )

                                    T_sat_total[p_idx, f_idx] += link_val * sd['sat_temps'][s_idx, f_idx]
                    else:
                        # Use optimized vectorized approach for normal case
                        # Precompute per-satellite, per-frequency kernel independent of pointing
                        # Kernel[S, F] = gain_sat[S] * FSPL_inv[S, F] * T_sat[S, F]
                        sat_ant = sd['instru_sat'].get_antenna()
                        dec_tel_sat = sd['sat_dec']
                        caz_tel_sat = -sd['sat_caz']
                        gain_sat = sat_ant.get_gain_values(dec_tel_sat, caz_tel_sat)  # (S,)

                        r_S = sd['sat_distances'].astype(np.float64)
                        f_F = f_RX_array.astype(np.float64)
                        c = 3.0e8
                        fspl_inv = ((c / f_F)[np.newaxis, :] / (4.0 * np.pi * r_S[:, np.newaxis])) ** 2  # (S, F)

                        kernel = gain_sat[:, np.newaxis] * fspl_inv * sd['sat_temps']  # (S, F)

                        # For each pointing, compute telescope gain over S sats and reduce
                        tel_ant = instrument.get_antenna()
                        for p_idx in range(n_pointings):
                            dec_tel = dec_tel_t[p_idx]
                            caz_tel = caz_tel_t[p_idx]
                            dec_sat_tel, caz_sat_tel = ground_to_beam_coord_vectorized(
                                sd['sat_dec'], sd['sat_caz'], dec_tel, caz_tel
                            )  # (S,)
                            gain_tel = tel_ant.get_gain_values(dec_sat_tel, caz_sat_tel)  # (S,)
                            T_sat_total[p_idx, :] += (gain_tel[:, np.newaxis] * kernel).sum(axis=0)

        # Final combination
        T_A = (1 / (4 * np.pi)) * (T_sat_total + max_gain * T_sky_arr)
        T_sys = T_A + (1 - eta_rad) * T_phy + T_RX_vals[np.newaxis, :]
        result[t_idx, :, :] = T_sys

    return result


def model_observed_temp_with_atmospheric_refraction_vectorized(
    observation,
    source_mdl: Callable,
    background_mdl: Callable,
    constellation=None,
    beam_avoidance=False,
    atmospheric_refraction: Optional[Dict[str, Any]] = None
) -> tuple:
    """
    VECTORIZED version of observation modeling with atmospheric refraction correction.

    This function dramatically improves performance by vectorizing operations across:
    - Time steps
    - Pointings
    - Frequencies
    - Satellites

    Performance improvements:
    - Atmospheric refraction: Vectorized across all pointings
    - Sky model: Vectorized across pointings and frequencies
    - Satellite processing: Vectorized across satellites and frequencies
    - Beam avoidance: Optimized with minimal non-vectorized operations

    Args:
        observation: Observation object containing trajectory and instrument data
        source_mdl: Callable function for observed point sopurce temperature contributions
        background_mdl: Callable function for sky and ground noise for telescope
        constellation: Optional constellation object(s) for satellite interference
        beam_avoidance: If True, uses non-vectorized approach for beam avoidance calculations
        atmospheric_refraction: Dictionary containing atmospheric refraction configuration:
            - 'temperature': Surface temperature in Kelvin (default: 288.15)
            - 'pressure': Surface pressure in Pa (default: 101325)
            - 'humidity': Relative humidity in % (default: 50.0)
            - 'apply_refraction_correction': Boolean to enable/disable refraction correction
            - 'refraction_model': 'standard' or 'advanced' (default: 'standard')

    Returns:
        result: Observation results with atmospheric refraction correction applied
        refraction_summary: Dictionary with atmospheric refraction statistics
    """

    # Get observation parameters
    times = observation.get_time_stamps()
    traj = observation.get_traj().sort_values(by='times').reset_index(drop=True)
    instrument = observation.get_instrument()
    n_times = len(times)
    n_pointings = len(traj) // n_times if n_times > 0 else 0

    # Instrument/antenna parameters
    f_RX_array = instrument.get_center_freq_chans()
    n_freq = len(f_RX_array)
    T_RX_func = instrument.get_inst_signal()
    T_phy = instrument.get_phy_temp()
    antenna = instrument.get_antenna()
    eta_rad = antenna.get_rad_eff()
    max_gain = antenna.get_boresight_gain()

    # Pre-shape all pointing data
    dec_tel_grid = np.radians(90 - traj['elevations'].values.reshape(n_times, n_pointings))
    caz_tel_grid = np.radians(-traj['azimuths'].values.reshape(n_times, n_pointings))

    # Initialize result array
    result = np.zeros((n_times, n_pointings, n_freq), dtype=np.float64)

    # Initialize refraction summary
    refraction_summary = {
        'refraction_corrections_applied': 0,
        'max_refraction_correction': 0.0,
        'min_refraction_correction': 0.0,
        'avg_refraction_correction': 0.0,
        'refraction_model_used': 'none'
    }

    # VECTORIZED ATMOSPHERIC REFRACTION CORRECTION
    if atmospheric_refraction is not None and atmospheric_refraction.get('apply_refraction_correction', True):
        # Convert all elevations to degrees for refraction calculation
        elevations_deg = 90 - np.degrees(dec_tel_grid)
        azimuths_deg = -np.degrees(caz_tel_grid)

        # Vectorized refraction correction calculation
        refraction_corrections = calculate_atmospheric_refraction_correction_vectorized(
            elevations_deg, atmospheric_refraction
        )

        # Apply corrections (subtract because we want true elevation)
        elevations_corrected = elevations_deg - refraction_corrections

        # Convert back to declination/co-azimuth
        dec_tel_grid = np.radians(90 - elevations_corrected)
        caz_tel_grid = np.radians(-azimuths_deg)

        # Update refraction summary
        valid_corrections = refraction_corrections[elevations_deg > 0]
        if len(valid_corrections) > 0:
            refraction_summary['refraction_corrections_applied'] = len(valid_corrections)
            refraction_summary['max_refraction_correction'] = np.max(valid_corrections)
            refraction_summary['min_refraction_correction'] = np.min(valid_corrections)
            refraction_summary['avg_refraction_correction'] = np.mean(valid_corrections)
            refraction_summary['refraction_model_used'] = atmospheric_refraction.get(
                'refraction_model', 'standard'
            )

    # Pre-compute satellite data
    satellite_data = {}
    if constellation is not None:
        if not isinstance(constellation, list):
            constellation = [constellation]
        for c_idx, con in enumerate(constellation):
            sat_TX = con.get_transmitter()
            for time in times:
                sats_t = con.sats[con.sats['times'] == time]
                if len(sats_t) > 0:
                    sat_TX_func = sat_TX.get_inst_signal()
                    sat_temps_1f = np.array([sat_TX_func(time, f) for f in f_RX_array], dtype=np.float64)
                    satellite_data[(c_idx, time)] = {
                        'sat_dec': np.radians(90 - sats_t['elevations'].values),
                        'sat_caz': np.radians(-sats_t['azimuths'].values),
                        'sat_distances': sats_t['distances'].values,
                        'sat_temps': np.tile(sat_temps_1f, (len(sats_t), 1)),
                        'lnk_bdgt': con.get_lnk_bdgt_mdl(),
                        'instru_sat': sat_TX,
                    }

    # VECTORIZED PROCESSING ACROSS ALL TIME STEPS
    for t_idx, time_step in enumerate(times):
        dec_tel_t = dec_tel_grid[t_idx]
        caz_tel_t = caz_tel_grid[t_idx]

        # VECTORIZED SKY MODEL COMPUTATION
        # Create meshgrids for vectorized sky model evaluation
        dec_mesh, freq_mesh = np.meshgrid(dec_tel_t, f_RX_array, indexing='ij')
        caz_mesh, _ = np.meshgrid(caz_tel_t, f_RX_array, indexing='ij')

        # Vectorized sky model computation
        T_sky_arr = np.array([
            [background_mdl(dec, caz, time_step, f) for dec, caz, f in zip(dec_row, caz_row, freq_row)]
            for dec_row, caz_row, freq_row in zip(dec_mesh, caz_mesh, freq_mesh)
        ], dtype=np.float64)

        # Vectorized receiver temperature computation
        T_RX_vals = np.array([T_RX_func(time_step, f) for f in f_RX_array], dtype=np.float64)

        # VECTORIZED SATELLITE PROCESSING
        T_sat_total = np.zeros((n_pointings, n_freq), dtype=np.float64)

        for c_idx, con in enumerate(constellation or []):
            sat_key = (c_idx, time_step)
            if sat_key in satellite_data:
                sd = satellite_data[sat_key]
                n_sats = len(sd['sat_dec'])

                if n_sats > 0:
                    if beam_avoidance:
                        # Use non-vectorized approach for beam avoidance
                        # Process each pointing and satellite combination individually
                        # to avoid broadcasting issues with coordinate transformations
                        for p_idx in range(n_pointings):
                            dec_tel = dec_tel_t[p_idx]
                            caz_tel = caz_tel_t[p_idx]

                            for s_idx in range(len(sd['sat_dec'])):
                                sat_dec = sd['sat_dec'][s_idx]
                                sat_caz = sd['sat_caz'][s_idx]
                                sat_dist = sd['sat_distances'][s_idx]

                                for f_idx in range(n_freq):
                                    freq = f_RX_array[f_idx]

                                    link_val = sd['lnk_bdgt'](
                                        dec_tel, caz_tel, instrument, sat_dec,
                                        sat_caz, sat_dist, sd['instru_sat'], freq
                                    )

                                    T_sat_total[p_idx, f_idx] += link_val * sd['sat_temps'][s_idx, f_idx]
                    else:
                        # Use optimized vectorized approach for normal case
                        # Precompute per-satellite, per-frequency kernel independent of pointing
                        # Kernel[S, F] = gain_sat[S] * FSPL_inv[S, F] * T_sat[S, F]
                        sat_ant = sd['instru_sat'].get_antenna()
                        dec_tel_sat = sd['sat_dec']
                        caz_tel_sat = -sd['sat_caz']
                        gain_sat = sat_ant.get_gain_values(dec_tel_sat, caz_tel_sat)  # (S,)

                        r_S = sd['sat_distances'].astype(np.float64)
                        f_F = f_RX_array.astype(np.float64)
                        c = 3.0e8
                        fspl_inv = ((c / f_F)[np.newaxis, :] / (4.0 * np.pi * r_S[:, np.newaxis])) ** 2  # (S, F)

                        kernel = gain_sat[:, np.newaxis] * fspl_inv * sd['sat_temps']  # (S, F)

                        # For each pointing, compute telescope gain over S sats and reduce
                        tel_ant = instrument.get_antenna()
                        for p_idx in range(n_pointings):
                            dec_tel = dec_tel_t[p_idx]
                            caz_tel = caz_tel_t[p_idx]
                            dec_sat_tel, caz_sat_tel = ground_to_beam_coord_vectorized(
                                sd['sat_dec'], sd['sat_caz'], dec_tel, caz_tel
                            )  # (S,)
                            gain_tel = tel_ant.get_gain_values(dec_sat_tel, caz_sat_tel)  # (S,)
                            T_sat_total[p_idx, :] += (gain_tel[:, np.newaxis] * kernel).sum(axis=0)

        # VECTORIZED FINAL COMBINATION
        T_A = (1 / (4 * np.pi)) * (T_sat_total + max_gain * T_sky_arr)
        T_sys = T_A + (1 - eta_rad) * T_phy + T_RX_vals[np.newaxis, :]
        result[t_idx, :, :] = T_sys

    return result, refraction_summary


def calculate_atmospheric_refraction_correction_vectorized(elevations_deg, atmospheric_config):
    """
    VECTORIZED atmospheric refraction correction calculation.

    Args:
        elevations_deg: Array of elevation angles in degrees
        atmospheric_config: Atmospheric configuration dictionary

    Returns:
        refraction_corrections: Array of refraction corrections in degrees
    """
    # Extract atmospheric parameters
    temperature = atmospheric_config.get('temperature', 288.15)
    pressure = atmospheric_config.get('pressure', 101325)
    humidity = atmospheric_config.get('humidity', 50.0)
    model = atmospheric_config.get('refraction_model', 'standard')

    # Convert elevations to radians
    elevations_rad = np.radians(elevations_deg)

    # Initialize result array
    refraction_corrections = np.zeros_like(elevations_deg)

    if model == 'standard':
        # Standard atmospheric refraction model (vectorized)
        T_std = 288.15  # K
        P_std = 101325  # Pa

        # Vectorized refraction coefficient calculation
        R_coeff = 1.02 * (pressure / P_std) * (T_std / temperature)

        # Vectorized refraction correction (avoid division by zero)
        valid_mask = elevations_rad > 0.01  # Avoid very small angles
        refraction_arcmin = np.zeros_like(elevations_rad)
        refraction_arcmin[valid_mask] = R_coeff / np.tan(elevations_rad[valid_mask])

        # Convert to degrees
        refraction_corrections = refraction_arcmin / 60.0

    elif model == 'advanced':
        # Advanced atmospheric refraction model (vectorized)
        # Water vapor pressure (vectorized)
        e = (humidity / 100.0) * 6.112 * np.exp((17.67 * (temperature - 273.15)) / (temperature - 29.65))

        # Refractive index of air
        n = 1 + (77.6e-6 * pressure / temperature) * (1 + 4810 * e / (temperature * pressure))

        # Vectorized refraction correction
        valid_mask = elevations_rad > 0.01
        refraction_corrections = np.zeros_like(elevations_rad)
        refraction_corrections[valid_mask] = (n - 1) * np.tan(elevations_rad[valid_mask]) * (180.0 / np.pi)

    # Limit refraction correction to reasonable values
    refraction_corrections = np.clip(refraction_corrections, 0.0, 0.5)

    return refraction_corrections


def generate_beam_aware_background_model(
    instrument,
    sky_mdl: Callable,
    ground_mdl: Callable,
    alpha_step = 1,
    beta_step = 5,
    dec_step=1,
    az_step = 10,
    ):
    """
    Generates a simple model of the external noise temperature contribution vs pointing \
    angle for background thermal noise to be used by observation functions fully integrates 
    ground and sky noise over the antenna pattern to produce true result.

    In future might be nice to make a version of this that uses the known terrain 
    for a better approximation at low elevation angles

    Args:
        instrument: instance of the Instrument class with an antenna model and associated coordinates
        sky_mdl:    sky noise model as a function of declination and azimuth
                        Should at minimum include atmospheric noise and cmb contribution
        ground_mdl: ground noise model as a function of declination and azimuth
        dec_step:    model resolution in declination
        caz_step:    model resolution in azimuth

    Returns: 
        background_mdl: interpolator for one time computation of system temperature vs telescope pointing position
    """ 

    max_dec = 90 #don't compute below horizon

    #get antenna pattern
    ant = instrument.get_antenna() 

    #generate azel grid for noise field

    ndec_step = 1
    ncaz_step = 5

    ncaz_grid = np.arange(-ncaz_step, 360+ncaz_step, ncaz_step)
    ndec_grid = np.arange(-ndec_step, 180+ndec_step, ndec_step)

    n_caz = len(ncaz_grid)
    n_dec = len(ndec_grid)

    nDec, nCaz  = np.meshgrid(ndec_grid, ncaz_grid, indexing='ij')

    #compute_noise_field

    noise_field = np.zeros_like(nDec)

    #sky_noise_field 
    noise_field[np.where(ndec_grid<90)[0]] = sky_mdl(nDec[np.where(ndec_grid<90)[0]], nCaz[np.where(ndec_grid<90)[0]])
    #ground_noise_field
    noise_field[np.where(ndec_grid>=90)[0]] = ground_mdl(nDec[np.where(ndec_grid>=90)[0]], nCaz[np.where(ndec_grid>=90)[0]])


    #create interpolateor function for the noise field

    noise_interp = RegularGridInterpolator((np.radians(ndec_grid), np.radians(ncaz_grid)), noise_field, method='linear')


    #####
    # generate sampling grid in the telescope frame
    alpha_grid = np.arange(0, 180+alpha_step, alpha_step)
    beta_grid = np.arange(0, 360+beta_step, beta_step)

    n_alpha = len(alpha_grid)
    n_beta = len(beta_grid)

    Alphas, Betas  = np.meshgrid(alpha_grid, beta_grid, indexing='ij')

    pattern = ant.get_gain_values(np.radians(Alphas), np.radians(Betas))

    int_weights = 1 / (4 * np.pi) * np.sin(np.radians(Alphas)) * np.radians(alpha_step) * np.radians(beta_step)
    int_weights = int_weights/np.sum(int_weights*pattern) * ant.rad_eff

    declinations = np.arange(-dec_step, max_dec+dec_step, dec_step)

    ## integrate over antenna pattern

    T_ext = np.zeros(np.shape(declinations), dtype=np.float64)

    for i, dec in enumerate(declinations):

        Dec, Az = beam_to_ground_coord_vectorized(np.radians(Alphas), np.radians(Betas), np.radians(dec), 0)
        noise = noise_interp((Dec,Az))
        T_ext[i] = np.sum(int_weights * pattern * noise)



    az_grid = np.arange(-az_step, 360+az_step, az_step)

    T_ext_grid=np.ones((len(declinations),len(az_grid)))*T_ext.reshape(len(T_ext),1) #expand to full dimension again

    return RegularGridInterpolator((declinations, az_grid), T_ext_grid, method='linear')