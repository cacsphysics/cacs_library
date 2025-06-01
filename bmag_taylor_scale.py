import numpy as np

from .tools import gen_separation_array
from .bmag_spatial_correlation import gen_shot_bmag_spatial_correlation, flatline_correction
from .bmag_temporal_correlation import gen_shot_pos_bmag_correlation
from .richardson_extrapolation import gen_ordered_sequence, richardson_extrapolation
from .bmx import finding_Index_Time
from .time_of_flight import get_tau_spatial


def single_shot_bmag_biased_taylor_scales(refPos, shot, filt, window, norm='ref'):
    """ Outputs the biased Taylor scales from bmag correlations of shot referenced 
    to refPos windowed and filt."""
    wfMagCorr = gen_shot_bmag_spatial_correlation(
        refPos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = gen_separation_array(refPos=refPos)
    biasedTSArray, maxSeparationArray, uncertainty = biased_taylor_scales(
        wfMagCorr, separations)
    return biasedTSArray, maxSeparationArray, uncertainty


def single_shot_abs_bmag_biased_taylor_scales(refPos, shot, filt, window, norm='ref'):
    """ Outputs the biased Taylor scales from ABS-bmag correlations of shot referenced 
    to refPos windowed and filt."""
    wfMagCorr = gen_shot_bmag_spatial_correlation(
        refPos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = gen_separation_array(refPos=refPos)

    wfMagCorr = flatline_correction(wfMagCorr)
    wfMagCorr/wfMagCorr[0]
    biasedTSArray, maxSeparationArray, uncertainty = biased_taylor_scales(
        wfMagCorr, separations)
    return biasedTSArray, maxSeparationArray, uncertainty


def single_shot_abs_bmag_biased_taylor_scales_flatline(refPos, shot, filt, window, norm='ref'):
    """ Outputs the biased Taylor scales from ABS-bmag correlations of shot referenced 
    to refPos windowed and filt."""
    wfMagCorr = gen_shot_bmag_spatial_correlation(
        refPos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = gen_separation_array(refPos=refPos)

    # wfMagCorr = flatline_correction(np.abs(wfMagCorr))  # Flatline correction
    wfMagCorr = wfMagCorr/wfMagCorr[0]  # Flatline correction
    biasedTSArray, maxSeparationArray, uncertainty = biased_taylor_scales(
        wfMagCorr, separations)
    return biasedTSArray, maxSeparationArray, uncertainty


def single_shot_bmag_biased_temporal_taylor_scales(refPos, shot, filt, window, norm='ref', window2=[0, 13]):
    wfMagCorr, tau = gen_shot_pos_bmag_correlation(
        refPos=refPos, pos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = get_tau_spatial(tau, shot)
    startDex = finding_Index_Time(separations*1e-6, window2[0])
    endDex = finding_Index_Time(separations*1e-6, window2[1])
    biasedTSArray, maxSeparationArray, uncertainty = biased_taylor_scales(
        wfMagCorr[startDex:endDex], separations[startDex:endDex])
    return biasedTSArray, maxSeparationArray, uncertainty


def single_shot_abs_bmag_biased_temporal_taylor_scales(refPos, shot, filt, window, norm='ref', window2=[0, 13]):
    wfMagCorr, tau = gen_shot_pos_bmag_correlation(
        refPos=refPos, pos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = get_tau_spatial(tau, shot)
    startDex = finding_Index_Time(separations*1e-6, window2[0])
    endDex = finding_Index_Time(separations*1e-6, window2[1])
    biasedTSArray, maxSeparationArray, uncertainty = biased_taylor_scales(
        wfMagCorr[startDex:endDex], separations[startDex:endDex])
    return biasedTSArray, maxSeparationArray, uncertainty


def single_shot_bmag_temporal_taylor_y_intercepts(refPos, shot, filt, window, norm='ref', window2=[0, 13]):
    wfMagCorr, tau = gen_shot_pos_bmag_correlation(
        refPos=refPos, pos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = get_tau_spatial(tau, shot)
    startDex = finding_Index_Time(separations*1e-6, window2[0])
    endDex = finding_Index_Time(separations*1e-6, window2[1])
    yTSArray, maxSeparationArray, uncertainty, _ = y_intercepts(
        wfMagCorr[startDex:endDex], separations[startDex:endDex])
    return yTSArray, maxSeparationArray, uncertainty


def single_shot_abs_bmag_temporal_taylor_y_intercepts(refPos, shot, filt, window, norm='ref', window2=[0, 13]):
    """ Outputs the y_intercept array from the temporal correlations of the absolute Bmag with respect
    to shot.
    """
    wfMagCorr, tau = gen_shot_pos_bmag_correlation(
        refPos=refPos, pos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = get_tau_spatial(tau, shot)
    startDex = finding_Index_Time(separations*1e-6, window2[0])
    endDex = finding_Index_Time(separations*1e-6, window2[1])
    yTSArray, maxSeparationArray, uncertainty, _ = y_intercepts(
        wfMagCorr[startDex:endDex], separations[startDex:endDex])
    return yTSArray, maxSeparationArray, uncertainty


def single_shot_abs_bmag_taylor_y_intercepts(refPos, shot, filt, window, norm='ref'):
    """ Outputs the TS y-intercepts from bmag correlations of shot referenced 
    to refPos windowed and filt."""
    wfMagCorr = gen_shot_bmag_spatial_correlation(
        refPos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    wfMagCorr = flatline_correction(wfMagCorr)
    wfMagCorr = wfMagCorr/wfMagCorr[0]
    separations = gen_separation_array(refPos=refPos)
    y_interceptsTSArray, maxSeparationArray, uncertainty, slopeArray = y_intercepts(
        wfMagCorr, separations)
    return y_interceptsTSArray, maxSeparationArray, uncertainty, slopeArray


def single_shot_bmag_taylor_y_intercepts(refPos, shot, filt, window, norm='ref'):
    """ Outputs the biased Taylor scales from bmag correlations of shot referenced 
    to refPos windowed and filt."""
    wfMagCorr = gen_shot_bmag_spatial_correlation(
        refPos=refPos, shot=shot, filt=filt, window=window, norm=norm)
    separations = gen_separation_array(refPos=refPos)
    y_interceptsTSArray, maxSeparationArray, uncertainty, slopeArray = y_intercepts(
        wfMagCorr, separations)
    return y_interceptsTSArray, maxSeparationArray, uncertainty, slopeArray


def biased_taylor_scales(correlation, separation, uncertainties=None):
    """ Outputs the biased Taylor scales, uncertainties and max separation array"""
    biasedTaylorArray, errorArray, maxSeparationArray, _ = gen_ordered_sequence(
        correlation, separation, uncertainties)
    return biasedTaylorArray, maxSeparationArray, errorArray[0]


def y_intercepts(correlation, separation, uncertainties=None):
    """ Outputs the y-intercepts from the linear fits on the biased Taylor scales."""
    yInterceptArray, errorArray, maxSeparationArray, slope_Array = richardson_extrapolation(
        correlation, separation, uncertainties)
    for num, values in enumerate(errorArray):
        mask = np.ones(errorArray.size, dtype=bool)
        mask[num] = False
        if np.isinf(values):
            errorArray[num] = np.mean(errorArray[mask])

    return yInterceptArray, maxSeparationArray, errorArray, slope_Array
