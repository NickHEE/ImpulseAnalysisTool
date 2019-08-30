from scipy.fftpack import fft, ifft
from scipy.io import wavfile
import numpy as np
import pandas as pd
import time, sys
import re, os, math


blockStrengths = {
    'B21A1':    2.652581619,
    'B21A2':    3.092453552,
    'B21A3':    3.461177927,
    'B21A4':    0,
    'B22A2':    16.39041906,
    'B22A3':    17.52833888,
    'B22A4':    16.23113014,
    'B23A1':    3.961344055,
    'B23A2':    4.554439856,
    'B23A3':    4.920854419,
    'B23A4':    4.148203016,
    'B23A5':    4.787829241,
    'B23B2':    5.190615074,
    'B23B3':    5.005369706,
    'B23B4':    5.565680021,
    'B23B5':    5.027170837,
    'B24A2':    12.38299759,
    'B24A3':    16.72339835,
    'B24A4':    15.3734368,
    'B25A2':    10.06579821,
    'B25A3':    9.285894171,
    'B25A4':    7.72140754,
    'B25A5':    9.826887715,
    'B25B2':    9.925460976,
    'B25B3':    8.921608942,
    'B25B4':    8.962023116,
    'B26A2':    22.8495735,
    'B26A3':    23.07707526,
    'B26A4':    23.1,
    'B27A2':    5.608910173,
    'B27A4':    6.448196892,
    'B28A2':    12.25211235,
    'B28A3':    10.69477435,
    'B28A4':    11.2007553,
    'B22A1':    13.39927115,
    'B23B1':	4.402718356,
    'B24A1':	12.76151949,
    'B25A1':	8.059429201,
    'B25B1':	7.549532926,
    'B26A1':	18.28471611,
    'B27A1':	5.184051562,
    'B28A1':	10.32448909,
}

strength_equations = {'Cylinder': lambda f, l: 0.818730*math.exp(0.00162*2*l*f),
                      'Two-Hole': lambda f: 0.6977*math.exp(0.00254*f)}

def addWidgets(layout, widgets):
    for widget in widgets:
        if type(widget) is tuple:
            layout.addWidget(widget[0], widget[1])
        else:
            layout.addWidget(widget)

def addWidgets_to_grid(grid, widgets):
    for widget in widgets:
        grid.addWidget(*widget)


def setAlignments(layout, widgets):
    for widget in widgets:
        layout.setAlignment(*widget)


def readWaveData(file, save=False):
    """ Modified from 'myreadthewavedata' (James Booth) """

    data = wavfile.read(file)
    dt = 1.0 / data[0]

    channels = np.size(np.shape(data[1]))
    if channels == 1:
        ch1 = data[1][:]
        ch2 = ch1
    else:
        ch1 = np.array(data[1][:, 0], 'float')  # left and right channel data
        ch2 = np.array(data[1][:, 1], 'float')

    avgVals = (ch1 + ch2) / 2.0
    x = np.array([i*dt for i in range(len(avgVals))])

    return x, avgVals


def find_first_greater_than(l, v):
    # return min(range(len(l)), key=lambda i: abs(l[i] - v))
    try:
        index = next((index for index, value in enumerate(l) if value >= v))
    except StopIteration:
        return None
    return index

def find_turn_on(t, v, toff=0.0, thresholddivide=100.0):
    """ Created by: James Booth 'jFFTUtilities.findturnon'
        Modified by: Nicholas Huttemann, 2019

        This is a simple attempt to find when the hammer blow starts
        by calculating the derivatives of the signal."""

    # TODO: If threshold divide = 1 ..., append 0? + and - toff
    vDiff = np.diff(v)
    vDiffMax = np.max(vDiff) / thresholddivide

    vDiffMaxIndex = find_first_greater_than(vDiff, vDiffMax)
    t_At_DiffMax = t[vDiffMaxIndex]

    tDiffMax_Plus_Offset = t_At_DiffMax + toff
    tDiffMax_Plus_Offset_Index = find_first_greater_than(t, tDiffMax_Plus_Offset)

    return tDiffMax_Plus_Offset_Index, t[tDiffMax_Plus_Offset_Index]

def freq_filter(freq, amp, filterz):
    start = time.time()

    amp_Temp = amp
    avgAmp = np.average(amp)

    for filter in filterz:
        fMin, fMax = (filter[0], filter[1])
        fMin_Index = find_first_greater_than(freq, fMin)
        fMax_Index = find_first_greater_than(freq, fMax)

        if fMin_Index is not None and fMax_Index is not None:
            amp_Temp[fMin_Index:fMax_Index] = avgAmp
    end = time.time()
    print(f'freq_filter: {end-start}')
    return freq, amp_Temp

def ProcessFFT(xdata, ydata, normFlag=1, detail=1):
    """ Created by: James Booth 'jFFTUtilities.ProcessFFT'
        Modified by: Nicholas Huttemann, 2019

        This routine will take the sound data (time and voltage)
        and generate the FFTs. The Amplitude FFT and the Power FFT are
        both returned as normalized to the maximum amplitude. The routine
        also delivers the frequency axis used for the FFTs.

        :param normFlag: 1 - normalize FFT, 0 - no normalization"""

    nzeros = -1

    while nzeros < 0:
        NumPts = len(xdata)
        dt = xdata[1] - xdata[0]
        # adjust this
        Ntot = 65536 * detail

        nzeros = Ntot - NumPts
        prez = nzeros // 2
        endz = nzeros - prez
        if nzeros < 0:
            detail += 1

    x = np.linspace(0.0, Ntot*dt, Ntot)
    # Now construct the zero padded array for the fft
    # Pad half the points at the beginning and half at the end
    y = np.array([], 'float')
    y1 = np.zeros(prez)
    y2 = ydata
    y3 = np.zeros(endz)
    y = np.append(y, y1)
    y = np.append(y, y2)
    y = np.append(y, y3)

    yfft = fft(y)

    # Create the frequency array and the truncated FFT array
    xf  = np.linspace(0.0, 1/2.0/dt, Ntot//2)
    #yfft_amp  = 2.0/len(y)*np.abs(yfft[0:Ntot/2])
    yfft_amp = np.abs(yfft[0:Ntot//2])
    yfft_power = (yfft_amp*yfft_amp)
    yfft_abs_amp = np.abs(yfft_amp)


    xfil = np.linspace(0.0, 1/2.0/dt, Ntot//2)

    # Frequency range to chop out: [min, max] pairs
    # In this case, I will remove the low frequency range from 0Hz - 200 Hz

    # freq_filtered, powerfft_filtered = freq_filter(xfil, yfft_power, filterRange)
    # freq_filtered, ampfft_filtered = freq_filter(xfil, yfft_abs_amp, filterRange)

    yfft_power[np.argwhere((xfil < 200) | (xfil > 16000))] = np.average(yfft_power)
    yfft_abs_amp[np.argwhere((xfil < 200) | (xfil > 16000))] = np.average(yfft_abs_amp)

    freq_filtered = xfil
    powerfft_filtered = yfft_power
    ampfft_filtered = yfft_abs_amp

    if normFlag == 1:
        powerfft_filtered = powerfft_filtered/np.max(powerfft_filtered) # Normalize the FFT peaks
        #ampfft_filtered = ampfft_filtered/np.max(ampfft_filtered)

    return freq_filtered, ampfft_filtered, powerfft_filtered


def AnalyzeFFT(filePath, batch=False, toCSV=False, toJSON=False, toff=0, tChop=100, detail=1):
    """

    :param filePath:
    :param batch:
    :param toCSV:
    :param toJSON:
    :param toff:
    :param detail:
    :return:
    """

    fileNameRe = r'_?loc(\d+[a-z]?)_(\d*)'
    try:
        fileNameMatch = re.search(fileNameRe, filePath)
        location = fileNameMatch[1]
        trial = int(fileNameMatch[2])
    except:
        location = 0
        trial = os.path.basename(filePath)
    fileName = os.path.basename(filePath)
    blockName = os.path.basename(os.path.dirname(filePath))
    try:
        strength = blockStrengths[blockName]
    except:
        strength = 0

    t, v = readWaveData(filePath)
    if tChop is not None:
        tChop = tChop / 100
        tOn_Index, tOn = find_turn_on(t, v, toff=toff, thresholddivide=1.0)
        vChopped = v[tOn_Index - 1:]
        tChopped = np.array([t - tOn for t in t[tOn_Index - 1:]])
        vChopped = vChopped[:int(len(vChopped) * tChop)]
        tChopped = tChopped[:int(len(tChopped) * tChop)]

    else:
        tOn_Index, tOn = find_turn_on(t, v, toff=toff, thresholddivide=1.0)
        vChopped = v[tOn_Index - 1:tOn_Index + 11000]
        tChopped = np.array([t - tOn for t in t[tOn_Index - 1: tOn_Index + 11000]], dtype='float')

    try:
        fVals, ampFFT, powerFFT = ProcessFFT(tChopped, vChopped, detail=detail)
    except:
        print(f'Failed to process wav: {fileName}')
        return None

    peakFreqs, peakAmps = find_the_peaks(fVals, powerFFT, delt=0.2, width=40)

    if batch:
        peakDictLen = len(peakFreqs)
        dF = fVals[1] - fVals[0]

        peakAmpFFT = [ampFFT[math.ceil(freq / dF)] for freq in peakFreqs]
        peakPowerFFT = [powerFFT[math.ceil(freq / dF)] for freq in peakFreqs]
        peakRelFreq = [(freq / peakFreqs[np.where(peakAmps == 1)[0][0]]) for freq in peakFreqs]

        # TODO: Change from combined peaks to separate file?
        if toCSV:
            peakDict = {'Block ID': [blockName]*peakDictLen, 'Location': [location]*peakDictLen, 'Trial': [trial]*peakDictLen,
                        'Frequency': peakFreqs, 'Ampl. FFT': peakAmpFFT, 'Power FFT': peakPowerFFT, 'Rel Freq': peakRelFreq,
                        'Rel Ampl': peakAmps}

            FFTDict = {'Frequency': fVals, 'Ampl. FFT': ampFFT, 'Power FFT': powerFFT}

            peakFrame = pd.DataFrame(peakDict)
            FFTFrame = pd.DataFrame(FFTDict)

            return peakFrame, (filePath, FFTFrame) # TODO: Awkward

        if toJSON:
            data = {
                "shape": "2-Hole",
                "testData": {
                    "location": location,
                    "strength": strength,
                    "peaks": []},
                "waveData": powerFFT.tolist(),
                "freqData": fVals.tolist()
            }
            for freq, val in zip(peakFreqs, peakPowerFFT):
                data['testData']['peaks'].append({'frequency': freq, 'magnitude': val})

            return data, filePath

    else:
        return tChopped, vChopped, fVals, powerFFT, peakFreqs, peakAmps

def peakdet(v, delta, x=None):
    """
    ## Converted from MATLAB script at http://billauer.co.il/peakdet.html
    ## Returns two arrays
    ## function [maxtab, mintab]=peakdet(v, delta, x)
    ##% PEAKDET Detect peaks in a vector
    ##% [MAXTAB, MINTAB] = PEAKDET(V, DELTA) finds the local
    ##% maxima and minima ("peaks") in the vector V.
    ##% MAXTAB and MINTAB consists of two columns. Column 1
    ##% contains indices in V, and column 2 the found values.
    ##%
    ##% With [MAXTAB, MINTAB] = PEAKDET(V, DELTA, X) the indices
    ##% in MAXTAB and MINTAB are replaced with the corresponding
    ##% X-values.
    ##%
    ##% A point is considered a maximum peak if it has the maximal
    ##% value, and was preceded (to the left) by a value lower by
    ##% DELTA.
    ##% Eli Billauer, 3.4.05 (Explicitly not copyrighted).
    ##% This function is released to the public domain; Any use is allowed.
    """
    maxtab = []
    mintab = []
    if x is None:
        x = np.arange(len(v))
        v = np.asarray(v)
    if len(v) != len(x):
        sys.exit('Input vectors v and x must have same length')

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True

    for i in np.arange(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = x[i]
        if this < mn:
            mn = this
            mnpos = x[i]

        if lookformax:
            if this < mx - delta:
                maxtab.append((mxpos, mx))
                mn = this
                mnpos = x[i]
                lookformax = False
        else:
            if this > mn + delta:
                mintab.append((mnpos, mn))
                mx = this
                mxpos = x[i]
                lookformax = True

    return np.array(maxtab), np.array(mintab)


def find_the_peaks(f, signal, delt=0.2, width=100.0):
    '''This routine will take the power FFT signals and find the peaks in the spectrum
       subject to:
            a minimum amplitude value of delt*(maximum peak height),
       and insuring that the peak frequencies
           are separated from each other by at least "width" Hz. '''
    maxamp = signal.max()
    peakmax, peakmin = peakdet(v=signal, delta=delt * maxamp, x=f)
    foundpeaks = peakmax[:, 1]
    foundfreqs = peakmax[:, 0]

    # Now consolidate the peaks within width, w
    # First calculate next neighbour frequency differences
    thefreqdiffs = np.array([0.0], 'float')
    for ip in range(1, len(foundfreqs)):
        thefreqdiffs = np.append(thefreqdiffs, foundfreqs[ip] - foundfreqs[ip - 1])

    # Second, slice up these arrays into subarrays
    ipeaks = 1
    selectedfreqs = np.array([], 'float')
    selectedpeaks = np.array([], 'float')
    subfreqs = np.array([foundfreqs[0]])
    subpeaks = np.array([foundpeaks[0]])
    ibreakfreqs = np.array([], 'int')

    for iq in range(1, len(thefreqdiffs)):
        if thefreqdiffs[iq] > width:
            ibreakfreqs = np.append(ibreakfreqs, iq)
            ipeaks = ipeaks + 1

    #print("ibreakfreqs = ", ibreakfreqs)
    #print("foundpeaks = ", foundpeaks)
    #print("foundfreqs = ", foundfreqs)

    imin = 0
    imax = 0
    for il in range(len(ibreakfreqs)):
        imax = ibreakfreqs[il]
        # print "imin, imax = ", imin, imax
        subpeaks = foundpeaks[imin:imax]
        subfreqs = foundfreqs[imin:imax]

        ifmax = subpeaks.argmax()
        localfreqmax = subfreqs[ifmax]
        localmax = subpeaks[ifmax]
        if localfreqmax > 0:  # Set min frequency of 1000 Hz for cylinders
            selectedfreqs = np.append(selectedfreqs, localfreqmax)
            selectedpeaks = np.append(selectedpeaks, localmax)

        imin = imax

    subpeaks = foundpeaks[imax:len(foundfreqs)]
    subfreqs = foundfreqs[imax:len(foundfreqs)]
    ifmax = subpeaks.argmax()
    localfreqmax = subfreqs[ifmax]
    localmax = subpeaks[ifmax]
    selectedfreqs = np.append(selectedfreqs, localfreqmax)
    selectedpeaks = np.append(selectedpeaks, localmax)

    return selectedfreqs, selectedpeaks


def removeOutliers(x, y, outlierConstant=1):
    upper_quartile = np.percentile(x, 75)
    lower_quartile = np.percentile(x, 25)
    IQR = (upper_quartile - lower_quartile) * outlierConstant
    quartileSet = (lower_quartile - IQR, upper_quartile + IQR)

    notOutliers = np.where((x >= quartileSet[0]) & (x <= quartileSet[1]))

    return x[notOutliers], y[notOutliers]

def filter_time_data(t, v, tzero=0.0):
    """ Created by: James Booth 'jFFTUtilities.filter_time_data'
        Modified by: Nicholas Huttemann, 2019

        This function chops off the data prior to the impulse
        turning on. It also shifts the zero in time to the first data point."""
    pass


