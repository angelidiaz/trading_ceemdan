from PyEMD import EMD, CEEMDAN, Visualisation
import numpy as np
import pandas as pd
from scipy.signal import hilbert
from scipy.interpolate import CubicSpline
import matplotlib.pyplot as plt
from PyEMD.compact import filt6, pade6


def amplitud(sig):
    analytic_signal = hilbert(sig)
    return np.abs(analytic_signal)


def freq_aprx(freq_inst):
    interval_ifreq_inf = np.array(
        list(map(lambda freq: freq - 0.01, freq_inst)))
    interval_ifreq_sup = np.array(
        list(map(lambda freq: freq + 0.01, freq_inst)))
    freq_ap = np.array([0.0] * len(freq_inst))
    for cont1 in range(len(freq_inst)):
        if freq_ap[cont1] != 0:
            index = [cont1]
            for cont2 in range(cont1 + 1, len(freq_inst)):
                if abs(interval_ifreq_inf[cont1] -
                       interval_ifreq_inf[cont2]) < 0.02:
                    index.append(cont2)


def hilbert_expectrum_emd(series, time):
    emd = EMD()
    emd.emd(series)
    imfs, res = emd.get_imfs_and_residue()
    vis = Visualisation()
    ifreq = vis._calc_inst_freq(imfs, time, False, None)
    ampl = np.array(list(map(lambda imf: amplitud(imf), imfs)))
    hilbert_expectrum_imf = []
    ins_freq = []
    for num_imf in range(len(imfs)):
        freq_k = ifreq[num_imf]
        ampl_k = ampl[num_imf]
        ins_freq = ins_freq + list(set(freq_k))
        hilbert_exp = {}
        for freq in freq_k:
            for cont in range(len(freq_k)):
                if freq > freq_k[cont] - 0.01 and freq < freq_k[cont] + 0.01:
                    hilbert_exp[freq, time[cont]] = ampl_k[cont]
                else:
                    hilbert_exp[freq, time[cont]] = 0.0
        hilbert_expectrum_imf.append(hilbert_exp)
    ins_freq = list(set(ins_freq))
    hilbert_expectrum = {}
    for freq in ins_freq:
        for tm in time:
            key = (freq, tm)
            amp = 0
            for sp_hilb in hilbert_expectrum_imf:
                if key in sp_hilb.keys():
                    amp = amp + sp_hilb[key]
            hilbert_expectrum[key] = amp
    return hilbert_expectrum, ins_freq


def marginal_expectrum_emd(series, time):
    hilb_expectrum, inst_freq = hilbert_expectrum_emd(series, time)
    marginal = {}
    delta_time = abs(time[0] - time[1])
    for freq in inst_freq:
        sum_ampl = 0
        for tm in time:
            sum_ampl = sum_ampl + hilb_expectrum[(freq, tm)]
        marginal[freq] = sum_ampl / len(time)
    return marginal, hilb_expectrum, inst_freq


def degree_of_estacionarity_emd(series, time):
    marginal, hilb_expectrum, inst_freq = marginal_expectrum_emd(series, time)
    ds_est = []
    delta_time = abs(time[0] - time[1])
    for freq in inst_freq:
        est = 0
        for tm in time:
            est = est + ((1 -
                          (hilb_expectrum[(freq, tm)] / marginal[freq]))**2)
        ds_est.append(est / len(time))
    return ds_est, inst_freq


def hilbert_expectrum_imf(series):
    time_series = np.linspace(0, len(series) - 1, len(series))
    complex_form = hilbert(series)
    amplitud = np.abs(complex_form)
    phase = np.unwrap(np.angle(complex_form))
    phase_smooth = CubicSpline(time_series, phase)
    inst_freq = CubicSpline(
        time_series,
        ((phase_smooth(time_series + 0.00001) - phase_smooth(time_series)) /
         0.00001) / (2 * np.pi))
    inst_freq = inst_freq(time_series)
    hilbert_exp = {}
    for freq in inst_freq:
        for time in time_series[:len(time_series)]:
            if freq == inst_freq[int(time)]:
                hilbert_exp[freq, time] = amplitud[int(time)]
            else:
                hilbert_exp[freq, time] = 0.0
    inst_freq = list(set(inst_freq))
    return hilbert_exp, inst_freq


def hilbert_expectrum_prov(series):
    time_series = np.linspace(0, len(series) - 1, len(series))
    ceemdan = CEEMDAN()
    c_imfs = ceemdan(series)
    hilb_expec = {}
    hilbert_imf = []
    inst_freq = []
    for k in range(len(c_imfs) - 1):
        print(k)
        spectrum, freq = hilbert_expectrum_imf(c_imfs[k])
        hilbert_imf.append(spectrum)
        inst_freq = inst_freq + freq
    inst_freq = list(set(inst_freq))
    for freq in inst_freq:
        for time in time_series:
            key = (freq, time)
            amp = 0
            for sp_hilb in hilbert_imf:
                if key in sp_hilb.keys():
                    amp = amp + sp_hilb[key]
            hilb_expec[key] = amp
    return hilb_expec, inst_freq


def marginal_expectrum(series):
    hilb_expectrum, inst_freq = hilbert_expectrum(series)
    time_series = np.linspace(0, len(series) - 1, len(series))
    marginal = {}
    for freq in inst_freq:
        sum_ampl = 0
        for time in time_series:
            sum_ampl = sum_ampl + hilb_expectrum[(freq, time)]
        marginal[freq] = sum_ampl / len(time_series)
    return marginal, hilb_expectrum, inst_freq


def degree_of_estacionarity(series):
    marginal, hilb_expectrum, inst_freq = marginal_expectrum(series)
    ds_est = {}
    time_series = np.linspace(0, len(series) - 1, len(series))
    for freq in inst_freq:
        est = 0
        for time in time_series:
            est = est + (1 -
                         (hilb_expectrum[(freq, time)] / marginal[freq]))**2
        ds_est[freq] = est / len(time_series)
    return ds_est


def graph_of_imfs_ceemdan(series, time, flag=False):
    """
    Print the graphs IMFs of a stock or the graph of the stock

    PARAMETER
    ---------
    symbol : string
        Symbols of a stock

    RETURN
    ------
        None
        Show the graphs of IMFs
    """
    ceemdan = CEEMDAN()
    c_imfs = ceemdan(series)
    axis_x = time
    plt.plot(axis_x, series)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
    cont = 1
    if flag:
        for imf in c_imfs:
            axis_x = np.arange(len(imf))
            plt.plot(axis_x, imf)
            plt.xlabel('x')
            plt.ylabel('y')
            plt.title('IMF' + str(cont))
            plt.show()
            cont = cont + 1


def ceemdan_get_imfs_and_residue(series):
    ceemdan = CEEMDAN()
    c_imfs = ceemdan(series)
    imf_s = []
    for k in range(len(c_imfs) - 1):
        imf_s.append(c_imfs[k])
    return np.array(imf_s), c_imfs[len(c_imfs) - 1]


def visualization_imfs(time, series, string):
    if string == 'emd':
        plt.plot(time, series)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        emd = EMD()
        emd.emd(series)
        imfs, res = emd.get_imfs_and_residue()

        # imfs, res = ceemdan_get_imfs_and_residue(S)

        # Initiate visualisation with emd instance

        vis = Visualisation()

        # Create a plot with all IMFs and residue

        vis.plot_imfs(imfs=imfs, residue=res, t=time, include_residue=True)

        # Create a plot with instantaneous frequency of all IMFs

        vis.plot_instant_freq(t, imfs=imfs)
        ifreq = vis._calc_inst_freq(imfs, time, False, None)
        iphase = np.array(
            list(map(lambda imf: vis._calc_inst_phase(imf, alpha=None), imfs)))
        vis.plot_imfs(imfs=iphase, residue=res, t=time, include_residue=True)

        ampl = np.array(list(map(lambda imf: amplitud(imf), imfs)))

        vis.plot_imfs(imfs=ampl, residue=res, t=time, include_residue=True)
        # print(type(ifreq))
        # Show both plots
        vis.show()

    if string == 'ceemdan':
        plt.plot(time, series)
        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()

        # emd = EMD()
        # emd.emd(series)
        # imfs, res = emd.get_imfs_and_residue()

        imfs, res = ceemdan_get_imfs_and_residue(series)

        # Initiate visualisation with emd instance

        vis = Visualisation()

        # Create a plot with all IMFs and residue

        vis.plot_imfs(imfs=imfs, residue=res, t=time, include_residue=True)

        # Create a plot with instantaneous frequency of all IMFs

        vis.plot_instant_freq(time, imfs=imfs)
        # ifreq = vis._calc_inst_freq(imfs, time, False, None)

        iphase = np.array(
            list(map(lambda imf: vis._calc_inst_phase(imf, alpha=None), imfs)))
        vis.plot_imfs(imfs=iphase, residue=res, t=time, include_residue=True)

        ampl = np.array(list(map(lambda imf: amplitud(imf), imfs)))
        vis.plot_imfs(imfs=ampl, residue=res, t=time, include_residue=True)

        # print(ifreq)
        # Show both plots
        vis.show()


if __name__ == '__main__':
    t = np.linspace(0, 512, 2000)
    S = np.array(list(np.cos(5 * np.pi * t / 32))) + 5
    # graph_of_imfs_ceemdan(s, t, True)
    # print(degree_of_estacionarity(s))
    # Simple signal example
    # t = np.arange(0, 3, 0.01)
    # S = np.sin(13 * t + 0.2 * t**1.4) - np.cos(3 * t)

    # visualization_imfs(t, S, 'emd')
    visualization_imfs(t, S, 'ceemdan')
    # ds, freq = degree_of_estacionarity_emd(S, t)
    # print(len(freq), len(t))
    # plt.plot(freq, ds)
    # plt.xlabel('x')
    # plt.ylabel('y')
    # plt.show()
