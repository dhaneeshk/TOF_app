"""
pyTOF program

Required:
    pyqtgraph - https://pyqtgraph.readthedocs.io/en/latest/installation.html

Flow:
    A. Import library
    B. Create custom Qt Class, the tofViewBox
    C. Create main Qt GUI Class
        - Main class is made at Qt Designer
        - Functions are manually added
    D. Create Qt Objects
    E. Start Qt GUI

Dev Notes:
    v5 - Digitizer re-init everytime data is requested from it
    v6 - Digitizer no longer re-init every time data is requested
    v7 - Added Cumulative & Current TOF Spectra
    v8 - Added QThread-ing mechanism
    v9 - Migrating all button functions to class - QThread only request data
    v10 - Improving QThread speed - QThread request and draw data, and waits for main thread permission
            KA: IT WORKS BUT SLOW! (maybe because it is waiting for data? ping-pong?)
        - PLotting after worker emit works faster, but it crashes within 20-30 iterations... Why?
    spinoff: v10 - SINGLE: SINGLE Thread Edition!
    v11 - Data acquisition, Data saving, and Data drawing are now contained in worker thread - full of bugs
    KA: GUI IS SUPER SMOOTH, BUT WORKER THREAD SEEMS TO CRASH AFTER 10-20 ITERATIONS!
    v12 - Emit data to main GUI but redraw controlled by Timer
        WORKS! (LESSON: YOU *MUST* EMIT EVERYTHING!!! - even the ones you thought the main GUI wont update)
    v13 - Update GUI and fix BUG at Calibration
        1. Fix recomputation of POS_C and NEG_C
        2. Add textbox for POS_C and NEG_C
        3. Add tab for post-process spectra?
        4. LoadExtParam now checks if every variable is loaded
        5. Start/stop button now has 250ms timeout
    v14 - Bergmann BME Delay Generator Initialization and Control is now implemented
    v15 - Added de-calib button to display/save raw TOF data
    v16 - Added buttons for double acquisition to observe hi-mass data
        1. Streamlined many variable handling - it's tidier now
        2. 0-100 us acquisition now possible by halving sampling rate to 625 MHz
        3. Add option to ignore delay generator
        4. Calib calculator
        5. Focus implemented for cumu and ref curves
        6. Added delay generator GUI
        7. Fixed calibration/decalibration button
        8. Added 1st derivative button
    v17 - Revamp UI - now we are merging acquisition and analysis sofware together
        1. Simulation multithreading added
        2. Keyboard shortcut added
        3. File save directory organization enhanced
    v18 - TOFv5.ui now used
        1. Added GUI to quickly take difference between two peak on analysis tab
        2. The Invert Y in digitizer tab now preserve all data (instead of only retaining the positive signal)
    v19 - TOFv6.ui now used
        1. Added smooth function
        2. Shortcut to start the acquisition now is coupled to the Start-button timeout
    v20 - Now you can load multiple files.
    
    v21 -
        1. TOF v7 now used - Added checkmark for localization algorithm and blank spectral correction
        2. SINGLE connection to hardware!
    v22 - 
        1. Renormalization in cumulative curve
    v23
        1. Renormalization is removed.
        2. Digital hi-pass filter is now added.
        3. TOF v8 now used
            - FFT Lowpass added and SavGol filter removed.
            - Localization and blank checkmark removed.
        
"""


"""
# ---- Import Library

"""
# Typical python packages
import numpy as np
import datetime
import os
import time
import math
import scipy.signal
# For SPECTRUM Digitizer
import sys
from pyspcm import *
from spcm_tools import *
# For BERGMANN Delay Generator
# from ctypes import * ## ctypes import is already done at pyspcm import above

# I/O-Engine
def parse(s,delim=' ',ty='str'):
    temp = []
    s = s.lstrip()
    while '  ' in s:
        s = s.replace('  ',' ')
    while s:
        if s.find(delim) > 0:
            lim = s.find(delim)
            if ty == 'flo': temp.append( float(s[0:lim].strip()) )
            elif ty == 'int': temp.append( int(s[0:lim].strip()) )
            elif ty == 'str': temp.append( s[0:lim].strip() )
            s = s[lim+1:].lstrip()
        else:
            lim = len(s)
            if ty == 'flo': temp.append( float(s[0:lim].strip()) )
            elif ty == 'int': temp.append( int(s[0:lim].strip()) )
            elif ty == 'str': temp.append( s[0:lim].strip() )
            break
    return temp

"""
# ---- TOFviewBox class
Modified from pyqtgraph example re Customized ViewBox

MODIF: RE-defined mouse click and mouse drag actions to:
    L-Click: Reset zoom to show all
    M-Click: Reset zoom to show all
    R-Click: Call menu (eg. for fast PNG export to ESI PPT labbook)
    L-Drag: Zoom into X-range defined by box
    M-Drag: Zoom into X-range defined by box
    R-Drag: Define box to find TALLEST peak within (PS. the peak label can be moved!)

"""
import pyqtgraph as pg
from pyqtgraph.Qt import QtCore, QtGui, QtWidgets
from pyqtgraph import PlotWidget

## DEFINE STYLES
line_pen = pg.mkPen(color='g', width=1, style=QtCore.Qt.DashLine)  # for peak assignment
line_font = QtGui.QFont()
line_font.setPixelSize(16)
line_font.setBold(True)

## DEFINE CUSTOM VIEWBOX
class tofViewBox(pg.ViewBox):
    
    peakfind = QtCore.pyqtSignal(object)
    peaksend = QtCore.pyqtSignal(float)
    
    def __init__(self, *args, **kwds):
        # Inherit all ViewBox methods and properties
        pg.ViewBox.__init__(self, *args, **kwds)
        
        # Set Mouse Behavior
        self.setMouseMode(self.RectMode)
        
        # Create a InfiniteLine for peak annotation
        self.vLine = pg.InfiniteLine(pos=100, angle=90, movable=False, label="", pen=line_pen)
        self.vLine.label.setMovable(True)
        self.vLine.label.setFont(line_font)
        self.vLine.label.setColor('g')
        self.vLine.label.setPosition( 0.8 )
        self.addItem(self.vLine)
        self.pos = 100
        
        # Set behavior when X-range is changed
        self.sigXRangeChanged.connect(self.setYRange)
        
        # print('ViewBox generated')
    
    def setTgt(self, inpA, inpB):
        self.tgtX = inpA
        self.tgtY = inpB
    
    def setYRange(self):
        self.enableAutoRange(axis='y')
        self.setAutoVisible(y=True)
    
    ## Define mouse double click actions
    def mouseDoubleClickEvent(self, ev):
        
        if ev.button() == QtCore.Qt.LeftButton:
            self.peaksend.emit(self.pos)         # Executed if double left-click detected
    
    ## Re-define mouse click actions
    def mouseClickEvent(self, ev):
        
        if ev.button() == QtCore.Qt.LeftButton:          # Executed if left-click detected
            self.autoRange()
            
        elif ev.button() == QtCore.Qt.MiddleButton:      # Executed if middle-click detected
            self.autoRange()
            
        elif ev.button() == QtCore.Qt.RightButton:       # Executed if right-click detected
            ev.accept()
            self.raiseContextMenu(ev)
    
    ## Re-define mouse drag actions
    def mouseDragEvent(self, ev):
        
        if ev.button() == QtCore.Qt.LeftButton:          # Executed if left-drag detected
            ## pyQT Default: Zoom in based on mouse-defined rectangle
            if ev.isFinish():
                ## FKF Modif: Zoomn in and then Autoscale Y
                # self.sigXRangeChanged.connect(self.setYRange)
                pass
            pg.ViewBox.mouseDragEvent(self, ev)
        
        elif ev.button() == QtCore.Qt.MiddleButton:      # Executed if middle-drag detected
            ## pyQT Default: Zoom in based on mouse-defined rectangle
            if ev.isFinish():
                ### FKF Modif: Zoomn in and then Autoscale Y
                # self.sigXRangeChanged.connect(self.setYRange)
                pass
            pg.ViewBox.mouseDragEvent(self, ev)
        
        elif ev.button() == QtCore.Qt.RightButton:       # Executed if right-drag detected
            ## pyQT Default: Scale based on mouse movement
            ## FKF Modif: Disable scaling
            self.setMouseEnabled(x=False,y=False)
            ## FKF Modif: Detect peak within X-range of mouse-defined range
            if ev.isFinish():
                ## Clean up and prepare data
                self.rbScaleBox.hide()
                pos = ev.pos()
                ax = QtCore.QRectF(ev.buttonDownPos(ev.button()), pos)
                ax = self.childGroup.mapRectFromParent(ax)
                
                ## Detect Ymax within X-range
                xi,xf = ax.left(),ax.right()
                if xi > xf: xi, xf = xf, xi
                imin,fmin,imax,fmax = False,False,False,False
                if xi < np.min(self.tgtX): xi = np.min(self.tgtX); imin = True
                if xf < np.min(self.tgtX): xf = np.min(self.tgtX); fmin = True
                if xi > np.max(self.tgtX): xi = np.max(self.tgtX); imax = True
                if xf > np.max(self.tgtX): xf = np.max(self.tgtX); fmax = True
                # print('user1',xi,xf,np.min(self.tgtX),np.max(self.tgtX),imin,fmin,imax,fmax)
                
                if (imin and fmin) or (imax and fmax):
                    
                    print('Nothing is selected')
                    
                else:
                    
                    xi = np.where(abs(self.tgtX >= xi))[0][0]
                    xf = np.where(abs(self.tgtX <= xf))[0][-1]
                    if len( self.tgtY[xi:xf] ) == 0:
                        print('No data point within range')
                    else:
                        peak_index = np.where(self.tgtY[xi:xf] == np.max(self.tgtY[xi:xf]))
                        # print('user2',np.max(self.tgtY[xi:xf]),xi,xf,self.tgtX[xi], self.tgtX[xf],peak_index)
                        if len(peak_index[0])>1:
                            peak_index = peak_index[0][0]
                            print('Fix!')
                        peak_mass = (self.tgtX[xi+peak_index])[0]
                        print('Peak:',peak_mass)
                        
                        ## Output
                        self.vLine.setPos(peak_mass)
                        self.vLine.label.setText("%.3f"%(peak_mass))
                        self.peakfind.emit([ xi,xf,peak_mass ])
                        self.pos = peak_mass
                
            else:
                ## Update shape of scale box
                self.updateScaleBox(ev.buttonDownPos(), ev.pos())
                
            ev.ignore()
            pg.ViewBox.mouseDragEvent(self, ev)

"""
# ---- DataAcquisition Class
Define Data Acquisition Thread as a Class

"""
class Worker(QtCore.QObject):
    
    # Define signals to be emitted to main GUI thread
    finished = QtCore.pyqtSignal()
    rawComplete = QtCore.pyqtSignal(object)
    digiSignal = QtCore.pyqtSignal(str)
    delaySignal = QtCore.pyqtSignal(str)
    tofSignal = QtCore.pyqtSignal(str)
    
    def __init__(self, calib_pos, calib_neg, tlist, digiparam, parent = None):
        super(Worker, self).__init__(parent)
        
        # Some variables passed from main GUI declared as class variables in worker class
        self.POS_A = calib_pos[0]
        self.POS_B = calib_pos[1]
        self.POS_C = calib_pos[2]
        self.NEG_A = calib_neg[0]
        self.NEG_B = calib_neg[1]
        self.NEG_C = calib_neg[2]
        self.timelist = tlist
        self.pol = False if digiparam[0][0] == 'N' else True
        self.ave = int( digiparam[1] )
        self.trig = 'ch0' if digiparam[2][0] == 'S' else 'ext0'
        self.clock = 6.25e8 if digiparam[3] else 1.25e9
        # 625MHz gives 1.6ns (~1 mz) reso; 1.25GHz gives 0.8ns (~0.5 mz) reso
        self.lo = float(digiparam[4])
        self.hi = float(digiparam[5])
        self.inv = digiparam[6]
        self.abs = digiparam[7]
        self.baserem = digiparam[8]
        self.calib = digiparam[9]
        self.diff = digiparam[10]
        
        self.blank = digiparam[11]
        self.firstrun = digiparam[12]
        # print(digiparam)
        
        # if self.blank:
        #     self.timelist2 = tlist[:]
        #     self.timelist2[0] = 55.9878132678
        #     # self.timelist2[8] = not self.timelist2[8]
        #     # self.timelist2[9] = not self.timelist2[9]
        #     print(self.timelist)
        #     print(self.timelist2)
        
        self.tmpX, self.tmpY = np.array([]), np.array([])
        
    def run(self):
        
        """ START """
        # Main control flag passed from main GUI thread
        self.getdata = True
        
        # Start SPECTRUM card
        self.digiStatus, self.digiInfo = self.init_SPECTRUM(Ave=self.ave, V_range=500, trig=self.trig, clock=self.clock)
        
        # Diagnosing Digitizer
        if self.digiStatus:
            self.digiSignal.emit('Digitizer: Connected to '+self.digiInfo)
        else:
            self.digiSignal.emit('Digitizer: BAD, '+self.digiInfo)
        
        # Start BME card if external trigger mode is chosen
        if self.trig == 'ext0':
            
            if self.firstrun:
                self.delayStatus, self.delayInfo = self.init_BERGMANN()
                
                # Diagnosing Delay Generator
                if self.delayStatus:
                    self.delaySignal.emit('DelayGen: Connected to '+self.delayInfo)
                else:
                    self.delaySignal.emit('DelayGen: BAD, '+self.delayInfo)
                    
            else:
                self.delayStatus = True
                # self.delaySignal.emit('DelayGen: Connected to '+self.delayInfo)
                print('BME is already connected from previous run')
                self.continue_BERGMANN()
                
        else:
            self.delayStatus, self.delayInfo = True, 'DEMO MODE'
            self.delaySignal.emit('DelayGen: '+self.delayInfo)
        
        """ RUN """
        # If all is good, time to start!
        if self.digiStatus and self.delayStatus:
            
            # Init Time and Data Counter
            now = time.time()
            lastTime = now
            self.count = 0
            
            if self.blank == False:
            
                #Start Delay Generator if external trigger mode is chosen
                if self.trig == 'ext0':
                    self.run_BERGMANN(self.timelist)
                
                while self.getdata:
                    
                    # Request data from Digitizer
                    self.tmpX = np.arange( KILO_B(64) ) * self.T_raster         # in nanoseconds
                    self.tmpY = self.req_SPECTRUM() * self.V_raster / self.ave
                    
                    if self.diff:
                        self.tmpY = np.gradient(self.tmpY)
                    
                    # Calibrate data (IF YOU WANT IT)
                    if self.calib:
                        self.tmpX, self.tmpY = self.calib_SPECTRUM(self.tmpX, self.tmpY, pol=self.pol, trim=True, baserem=self.baserem, locut=self.lo, hicut=self.hi, abso=self.abs, inv=self.inv)
                    
                    # Add count by 1
                    self.count += 1
                    
                    # Timer setup to be displayed on main GUI thread
                    now = time.time()
                    dt = now - lastTime
                    lastTime = now
                    
                    # Inform user of data update information
                    self.tofSignal.emit('Tof thread: Spectra '+str(self.count)+'; Arriving every %.1f msec' % (dt*1000))
                    
                    # Send data to main thread
                    self.rawComplete.emit( [self.tmpX, self.tmpY] )
            
            elif self.blank:
                
                while self.getdata:
                
                    #Start Delay Generator for TOF signal
                    if self.trig == 'ext0':
                        # print('1st')
                        self.run_BERGMANN(self.timelist)
                    
                    # Request TOF data from Digitizer
                    self.tmpX = np.arange( KILO_B(64) ) * self.T_raster         # in nanoseconds
                    self.tmpY = self.req_SPECTRUM() * self.V_raster / self.ave
                    
                    # if self.calib:
                    #     self.tmX, self.tmY = self.calib_SPECTRUM(self.tmpX, self.tmpY, pol=self.pol, trim=True, baserem=self.baserem, locut=self.lo, hicut=self.hi, abso=self.abs, inv=self.inv)
                    # else:
                    #     self.tmX, self.tmY = self.tmpX, self.tmpY
                    # self.rawComplete.emit( [self.tmX, self.tmY] )
                    
                    #Start Delay Generator for blank signal
                    if self.trig == 'ext0':
                        # print('2nd')
                        self.pause_BERGMANN()
                        self.run_BERGMANN(self.timelist2)
                    
                    # Request blank data from Digitizer and take its difference
                    self.tmpY = self.tmpY - (self.req_SPECTRUM() * self.V_raster / self.ave)
                    # self.tmpY = self.req_SPECTRUM() * self.V_raster / self.ave
                    
                    # Calibrate data (IF YOU WANT IT)
                    if self.calib:
                        self.tmpX, self.tmpY = self.calib_SPECTRUM(self.tmpX, self.tmpY, pol=self.pol, trim=True, baserem=self.baserem, locut=self.lo, hicut=self.hi, abso=self.abs, inv=self.inv)
                    
                    # Add count by 1
                    self.count += 1
                    
                    # Timer setup to be displayed on main GUI thread
                    now = time.time()
                    dt = now - lastTime
                    lastTime = now
                    
                    # Inform user of data update information
                    self.tofSignal.emit('Tof thread: Spectra '+str(self.count)+'; Arriving every %.1f msec' % (dt*1000))
                    
                    # Send data to main thread
                    self.rawComplete.emit( [self.tmpX, self.tmpY] )
        
        while self.getdata:
            pass # Worker thread will be trapped here if init fails
        
        """ EXIT """
        # If the Digitizer was active
        if self.digiStatus:
            self.digiStatus = self.close_SPECTRUM()
            
        # If the Delay generator was active
        if self.trig == 'ext0' and self.delayStatus:
            self.pause_BERGMANN()
            # self.delayStatus = self.close_BERGMANN()
            self.delayStatus = True
        elif self.trig == 'ch0' and self.delayStatus:
            self.delayStatus = False
        
        self.digiSignal.emit('Digitizer: Not Connected')
        self.delaySignal.emit('DelayGen: Paused')
        self.finished.emit()
        # print('Thread exiting')
    
    
    
    """ SPECTRUM Digitizer Routines """
    
    def init_SPECTRUM(self, Ave=3000, V_range=500, trig='ext0', clock=1.25e-9):
        """
        Connects to SPECTRUM card, and send some settings to it.
        The most important stuff:
            1. Ave = how many spectra to be averaged? (Recommended: 2k-3k)
            2. V_range = what does the maximum V you want the card to detect? (Recommended: 500mV)
            3. trig = what is the trigger to start data acquisition? (Must use 'ext0' to use real TOF)
            4. clock = what is the sampling speed? this decides the time resolution (Maximum is 1.25e9 samples per second to give 0.8 ns resolution)
        
        """
        # Open card
        self.Card = spcm_hOpen (create_string_buffer (b'/dev/spcm0'))
        hCard = self.Card
        if hCard == None:
            # sys.stdout.write("no card found...\n")
            statusOK = False
            print('Digitizer NOT initialized')
            return statusOK, "No card found"
        
        # Read type, function and sn and check for A/D card
        lCardType = int32 (0)
        spcm_dwGetParam_i32 (hCard, SPC_PCITYP, byref (lCardType))
        lSerialNumber = int32 (0)
        spcm_dwGetParam_i32 (hCard, SPC_PCISERIALNO, byref (lSerialNumber))
        lFncType = int32 (0)
        spcm_dwGetParam_i32 (hCard, SPC_FNCTYPE, byref (lFncType))
        maxADC = int32 (0)
        spcm_dwGetParam_i32 (hCard, SPC_MIINST_MAXADCVALUE, byref (maxADC))
        
        self.T_raster = 1e9 * 1/clock  #in nano-seconds - only defined by the Clock (sample rate)
        self.V_raster = V_range / maxADC.value
        
        sCardName = szTypeToName (lCardType.value)
        if lFncType.value == SPCM_TYPE_AI:
            # sys.stdout.write("Found: {0} sn {1:05d}\n".format(sCardName,lSerialNumber.value))
            statusOK = True
        else:
            # sys.stdout.write("This is an example for A/D cards.\nCard: {0} sn {1:05d} not supported by example\n".format(sCardName,lSerialNumber.value))
            spcm_vClose (hCard)
            statusOK = False
            print('Digitizer NOT initialized')
            return statusOK, "{0} sn {1:05d} not supported".format(sCardName,lSerialNumber.value)
        
        #INPUT CHANNEL
        spcm_dwSetParam_i32 (hCard, SPC_CHENABLE,       CHANNEL0)            # Choose Input Channel 0
        spcm_dwSetParam_i64 (hCard, SPC_AMP0,           V_range)             # Set input range to +/- 500
        
        #DATA ACQUISITION CONFIG
        spcm_dwSetParam_i32 (hCard, SPC_CARDMODE,       SPC_REC_STD_AVERAGE) # Single average mode
        spcm_dwSetParam_i32 (hCard, SPC_MEMSIZE,        GIGA_B(1))           # Go for maximum - 1 GSamples
        spcm_dwSetParam_i32 (hCard, SPC_PRETRIGGER,     32)                  # Samples before trigger event - min is 32S
        spcm_dwSetParam_i32 (hCard, SPC_POSTTRIGGER,    KILO_B(64)-32)       # Samples after trigger event - max is 1kS - 32S
        spcm_dwSetParam_i32 (hCard, SPC_SEGMENTSIZE,    KILO_B(64))          # Size of triggered segment
        spcm_dwSetParam_i32 (hCard, SPC_AVERAGES,       Ave)                 # How many averages?
        
        #CLOCK CONFIG
        spcm_dwSetParam_i32 (hCard, SPC_CLOCKMODE,     SPC_CM_INTPLL)        # clock mode internal PLL
        spcm_dwSetParam_i64 (hCard, SPC_SAMPLERATE,    int(clock))          # SAMPLE RATE IS 1.25 GSample/s
        spcm_dwSetParam_i32 (hCard, SPC_CLOCKOUT,      0)                    # No clock output
        
        # Choose Trigger
        if 'soft' in trig:
            spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,      SPC_TMASK_SOFTWARE)
        elif 'ch0' in trig:
            spcm_dwSetParam_i32 (hCard, SPC_TRIG_CH_ORMASK0,  SPC_TMASK0_CH0)
        elif 'ext0' in trig:
            spcm_dwSetParam_i32 (hCard, SPC_TRIG_ORMASK,      SPC_TMASK_EXT0)
            spcm_dwSetParam_i32 (hCard, SPC_TRIG_EXT0_MODE,   SPC_TM_POS)              # Configure Ext0 External Trigger mode
            spcm_dwSetParam_i32 (hCard, SPC_TRIG_EXT0_LEVEL0, 1500)                    # Waits for +1500mV at Ext0
            spcm_dwSetParam_i32 (hCard, SPC_TRIG_DELAY,       int(round(2000 / (32*self.T_raster))*32)) # Set 3 us (3744 Samples) delay after trigger to avoid TOF pickup
        
        # Define the data buffer
        self.qwBufferSize = uint64 (KILO_B(64)* sizeof(int32))  # in bytes
        self.pvBuffer = c_void_p ()
        self.pvBuffer = pvAllocMemPageAligned (self.qwBufferSize.value)
        
        # print('Digitizer started')
        return statusOK, "{0} S/N: {1:05d}".format(sCardName,lSerialNumber.value)

    def req_SPECTRUM(self):
        """
        Request data from SPECTRUM card. The card MUST be initialized first!
        
        """
        
        hCard = self.Card
        pvBuffer = self.pvBuffer
        qwBufferSize = self.qwBufferSize
        
        # Transfer buffer info to card
        spcm_dwDefTransfer_i64 (hCard, SPCM_BUF_DATA, SPCM_DIR_CARDTOPC, 0, pvBuffer, 0, qwBufferSize)
        
        # Start card and DMA
        dwError = uint32 ()
        dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_START | M2CMD_CARD_ENABLETRIGGER | M2CMD_DATA_STARTDMA)
        
        # Error check
        szErrorTextBuffer = create_string_buffer (ERRORTEXTLEN)
        if dwError != 0: # != ERR_OK
            spcm_dwGetErrorInfo_i32 (hCard, None, None, szErrorTextBuffer)
            sys.stdout.write("{0}\n".format(szErrorTextBuffer.value))
            spcm_vClose (hCard)
            # exit (1)
        
        # Wait until acquisition has finished
        else:
            
            # More error check
            dwError = spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_WAITREADY | M2CMD_DATA_WAITDMA)
            
            if dwError != ERR_OK:
                if dwError == ERR_TIMEOUT:
                    sys.stdout.write ("... Timeout\n")
                else:
                    sys.stdout.write ("... Error: {0:d}\n".format(dwError))
        
            else:
                # Even more error checking
                lStatus = int32 ()
                lAvailUser = int32 ()
                lPCPos = int32 ()
                spcm_dwGetParam_i32 (hCard, SPC_M2STATUS,            byref (lStatus))
                spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_LEN, byref (lAvailUser))
                spcm_dwGetParam_i32 (hCard, SPC_DATA_AVAIL_USER_POS, byref (lPCPos))
                
                qwTotalMem = int32 ()
                qwToTransfer = int32 ()
                qwTotalMem.value = lAvailUser.value
                # sys.stdout.write ("Stat:{0:08x} Pos:{1:08x} Avail:{2:08x} Total:{3:.2f}MB/{4:.2f}MB\n".format(lStatus.value, lPCPos.value, lAvailUser.value, c_double (qwTotalMem.value).value / MEGA_B(1), c_double (qwToTransfer.value).value / MEGA_B(1)))
        
                # THIS IS THE DATA! #
                pbyData = cast  (pvBuffer, ptr32) # Cast to pointer to 32bit integer
                
                # Stop the card
                spcm_dwSetParam_i32 (hCard, SPC_M2CMD, M2CMD_CARD_STOP | M2CMD_CARD_DISABLETRIGGER | M2CMD_DATA_STOPDMA)
        
        return np.array( [pbyData[i] for i in range(KILO_B(64))] )

    def calib_SPECTRUM(self, x, y, pol=True, trim=True, locut=100, hicut=3000, baserem=True, abso=True, inv= False):
        """
        How should we process the raw data from Digitizer, and present them?
    
        """
        # CALIBRATE 
        # by y = Ax**2 + Bx + C, where y = mass-to-charge ratio, and x = tof (in ns)
        if pol:
            A = self.POS_A
            B = self.POS_B
            C = self.POS_C
        else:
            A = self.NEG_A
            B = self.NEG_B
            C = self.NEG_C
        x = (A*x*x) + (B*x) + C
        
        # FILTER
        # by mass range; retain anything heavier than locut, and lighter than hicut
        if trim:
            y = y[ np.logical_and( x >= locut , x <= hicut ) ]
            x = x[ np.logical_and( x >= locut , x <= hicut ) ]
        
        # REMOVE BASELINE
        # by 5th order polyfit, where outliers beyond 1 stdev are ignored
        if baserem:
            m = 1   # stdev
            o = 5   # polyfit order
            tmp = np.stack((x,y), axis=1)
            tmp = tmp[ np.abs(tmp[:,1] - np.mean(tmp[:,1])) < m * np.std(tmp[:,1]) ]
            base = np.poly1d(np.polyfit(tmp[:,0],tmp[:,1],o))
            y = y - base(x)
        
        # Cosmetics: Only show negative y-signal
        if inv:
            y = -y         # Invert raw data
            # y[ y<0 ] = 0   # Any y-values that are minus will be zeroed
            y = y - np.min(y)
            
        #  Cosmetics: Make y-axis absolute
        if abso:
            y = np.abs(y)
        
        return x, y
    
    def close_SPECTRUM(self):
        """
        This terminate the SPECTRUM card properly.
        Failing to do so will cause the card to not respond to future initialization requests.
        
        """
        spcm_vClose (self.Card)
        del self.pvBuffer
        del self.qwBufferSize
        # print('Digitizer stopped')
        return False

    """ BERGMANN Delay Generator Routines """
    
    def init_BERGMANN(self):
        
        # Load BME 64-bit DLL
        try:
            self.bme = CDLL('DelayGenerator.dll')
        except:
            # print('BME: DLL not found')
            return False, "DLL not found"
        
        # Check how many card are detected in the system
        dg_err = c_long()
        count = self.bme.DetectPciDelayGenerators( byref(dg_err) )
        
        if count==0:
            # print('BME: No card detected')
            return False, "No card detected"
        
        elif count==1:
            # Reserve memory block for the card
            setup = self.bme.Reserve_DG_Data( count )
            if setup != 0:
                # print('BME: Memory Reservation fails')
                return False, "Memory Reservation fails"
            
            self.dg_num = c_int(0)      # An index to describe detected DG 
            dg_prod = c_long()          # 47 means BME_SG08P4
            dg_bus = c_long()           # PCI bus of detected DG
            dg_slot = c_long()          # PCI slot of detected DG
            dg_mas = c_bool()           # Boolean to indicate Master or Slave in detected DG
            info = self.bme.GetPciBusDelayGenerator(byref(dg_prod), byref(dg_bus), byref(dg_slot), byref(dg_mas), self.dg_num )
            # print('DG Detected: model', dg_prod.value, 'at PCI bus', dg_bus.value, 'slot', dg_slot.value)
            # This will print DG Detected: model 47 at PCI bus 4 slot 5
            
            # Initialize the card
            init = self.bme.Initialize_Bus_DG_BME( dg_bus, dg_slot, dg_prod, self.dg_num )
            
            if init==0:
                # Return all good signal
                return True, 'Model '+str(dg_prod.value)+' at PCI bus '+str(dg_bus.value)+' slot '+str(dg_slot.value)
            else:
                # print('BME: Card init fails')
                return False, "Card init fails"
    
    def continue_BERGMANN(self):
        # Load BME 64-bit DLL
        try:
            self.bme = CDLL('DelayGenerator.dll')
        except:
            # print('BME: DLL not found')
            return False, "DLL not found"
        self.dg_num = c_int(0)      # An index to describe detected DG 
    
    def run_BERGMANN(self, tlist):
        
        """
        BME_DELAYGENERATORS (in microseconds)
        A is to trigger the Digitizer
        B and C are to trigger the HV pulse generator
        
        # Ions need to repopulate the TOF waiting room one it is emptied by Push-Pull
        # The speed of this repopulation depends on the kinetic energy of ion
        # e.g. To repopulate the 10 cm space, 3000 mz ions with 50eV energy needs 55us
        # PS. Old TOF config uses 60us
        
        Lo mass config (1.250 GHz sampling)
        A_DELAY = 0
        A_WIDTH = 56
        B_DELAY = 0
        B_WIDTH = 56
        C_DELAY = 0
        C_WIDTH = 56
        T_REPEAT = 111   # 56+55 (the latter is needed to fill the 10cm space of the TOF waiting room) 
        MZ-range: ~25 to ~10.8k
        
        Hi mass config (0.625 GHz)
        A_DELAY = 0
        A_WIDTH = 108
        B_DELAY = 0
        B_WIDTH = 108
        C_DELAY = 0
        C_WIDTH = 108
        T_REPEAT = 163  # i.e. 108+55 (the latter is needed to fill the 10cm space of the TOF waiting room)
        MZ-range: ~25 to ~41.1k
        """
        
        tA, wA = tlist[0], tlist[1]
        tB, wB = tlist[2], tlist[3]
        tC, wC = tlist[4], tlist[5]
        trep = tlist[6]
        polA = tlist[7]
        polB = tlist[8]
        polC = tlist[9]
        
         # Channel A
        A_firefirst = c_double(tA)      # in us
        A_pulsewidth = c_double(wA)     # in us
        A_outputmodulo = c_long(1)
        A_outputoffset = c_long(0)
        A_gosignal = c_long(0x10)       # Master Primary
        A_positive = c_bool(polA)
        A_terminate = c_bool(False)
        # Channel B
        B_firefirst = c_double(tB)      # in us
        B_pulsewidth = c_double(wB)     # in us
        B_outputmodulo = c_long(1)
        B_outputoffset = c_long(0)
        B_gosignal = c_long(0x10)       # Master Primary
        B_positive = c_bool(polB)
        B_terminate = c_bool(False)
        # Channel C
        C_firefirst = c_double(tC)      # in us
        C_pulsewidth = c_double(wC)     # in us
        C_outputmodulo = c_long(1)
        C_outputoffset = c_long(0)
        C_gosignal = c_long(0x10)       # Master Primary
        C_positive = c_bool(polC)
        C_terminate = c_bool(False)
        # Channel D
        D_firefirst = c_double(0)       # in us
        D_pulsewidth = c_double(0)      # in us
        D_outputmodulo = c_long(1)
        D_outputoffset = c_long(0)
        D_gosignal = c_long(0x10)       # Master Primary
        D_positive = c_bool(True)
        D_terminate = c_bool(True)      # No output
        # Channel E
        E_firefirst = c_double(0)       # in us
        E_pulsewidth = c_double(0)      # in us
        E_outputmodulo = c_long(1)
        E_outputoffset = c_long(0)
        E_gosignal = c_long(0x10)       # Master Primary
        E_positive = c_bool(True)
        E_terminate = c_bool(True)      # No output
        E_disconnect = c_bool(False)
        E_ontomsbus = c_bool(False)
        E_inputpositive = c_bool(True)
        # Channel F
        F_firefirst = c_double(0)       # in us
        F_pulsewidth = c_double(0)      # in us
        F_outputmodulo = c_long(1)
        F_outputoffset = c_long(0)
        F_gosignal = c_long(0x10)       # Master Primary
        F_positive = c_bool(True)
        F_terminate = c_bool(True)      # No output
        F_disconnect = c_bool(False)
        F_ontomsbus = c_bool(False)
        F_inputpositive = c_bool(True)
        
        # Gate Function
        gate_function = c_long(0x0)     # Dont use gate function
        
        # Clock Function
        cl_enable = c_bool(True)
        cl_oscillator_divider = c_long(16)      # Set to 16 to get the CORRECT time calibration
        cl_trigger_divider = c_long(1)
        cl_trigger_multiplier = c_long(1)
        cl_clock_source = c_long(1)             #  Use crystal Oscillator
        
        # Trigger Function
        tr_triggerterminate = c_bool(True)
        tr_gateterminate = c_bool(True)
        tr_triggerlevel = c_double(0)
        tr_gatelevel = c_double(0)
        tr_gatedelay = c_double(-1)             # Not used
        tr_MSbus = c_long(0x1)                  # Primary trigger signal is placed on master/slave bus
        tr_internalclock = c_double(trep)       # in usec
        tr_presetvalue = c_long(1)
        tr_gatedivider = c_long(1)
        tr_forcetrigger = c_double(-1)          # Not used
        tr_stepbacktime = c_double(-1)          # Not used
        tr_burstcounter = c_long(1)
        tr_positivegate = c_bool(True)
        tr_internaltrigger = c_bool(True)       # Card will be trigged by internal clock
        tr_ignoregate = c_bool(True)
        tr_syncgate = c_bool(False)
        tr_risingedge = c_bool(True)
        tr_stoponpreset = c_bool(False)
        tr_resetwhendone = c_bool(True)
        tr_triggerenable = c_bool(True)
        
        setParam = self.bme.Set_BME_G08( 
            A_firefirst, A_pulsewidth, A_outputmodulo, A_outputoffset, A_gosignal, A_positive, A_terminate,
            B_firefirst, B_pulsewidth, B_outputmodulo, B_outputoffset, B_gosignal, B_positive, B_terminate,
            C_firefirst, C_pulsewidth, C_outputmodulo, C_outputoffset, C_gosignal, C_positive, C_terminate,
            D_firefirst, D_pulsewidth, D_outputmodulo, D_outputoffset, D_gosignal, D_positive, D_terminate,
            E_firefirst, E_pulsewidth, E_outputmodulo, E_outputoffset, E_gosignal, E_positive, E_terminate, E_disconnect, E_ontomsbus, E_inputpositive, 
            F_firefirst, F_pulsewidth, F_outputmodulo, F_outputoffset, F_gosignal, F_positive, F_terminate, F_disconnect, F_ontomsbus, F_inputpositive, 
            gate_function,
            cl_enable, cl_oscillator_divider, cl_trigger_divider, cl_trigger_multiplier, cl_clock_source,
            tr_triggerterminate, tr_gateterminate, tr_triggerlevel, tr_gatelevel, tr_gatedelay, tr_MSbus, tr_internalclock, tr_presetvalue, tr_gatedivider, tr_forcetrigger, tr_stepbacktime, tr_burstcounter, tr_positivegate, tr_internaltrigger, tr_ignoregate, tr_syncgate, tr_risingedge, tr_stoponpreset, tr_resetwhendone, tr_triggerenable, 
            self.dg_num
            )
    
    def start_BERGMANN(self):
        # Turn on any ongoing pulses
        self.bme.Activate_DG_BME( self.dg_num )
    
    def pause_BERGMANN(self):
        # Turn off any ongoing pulses
        self.bme.Deactivate_DG_BME( self.dg_num )
    
    def close_BERGMANN(self):
        # Turn off any ongoing pulses
        self.bme.Deactivate_DG_BME( self.dg_num )
        # Release any allocated memory block
        self.bme.Release_DG_Data()
        return False

"""
# ---- MassSpec Simul
Define MassSpec Simulation Thread as a Class
Chemparse from zapaan ( https://github.com/Zapaan/python-chemical-formula-parser )

"""
import mendeleev as md
import itertools
import re
from collections import Counter
ATOM_REGEX = '([A-Z][a-z]*)(\d*)'
OPENERS = '({['
CLOSERS = ')}]'
def is_balanced(formula):
    """Check if all sort of brackets come in pairs."""
    # Very naive check, just here because you always need some input checking
    c = Counter(formula)
    return c['['] == c[']'] and c['{'] == c['}'] and c['('] == c[')']
def _dictify(tuples):
    """Transform tuples of tuples to a dict of atoms."""
    res = dict()
    for atom, n in tuples:
        try:
            res[atom] += int(n or 1)
        except KeyError:
            res[atom] = int(n or 1)
    return res
def _fuse(mol1, mol2, w=1):
    """Fuse 2 dicts representing molecules. Return a new dict."""
    return {atom: (mol1.get(atom, 0) + mol2.get(atom, 0)) * w for atom in set(mol1) | set(mol2)}
def _parse(formula):
    """
    Return the molecule dict and length of parsed part.
    Recurse on opening brackets to parse the subpart and
    return on closing ones because it is the end of said subpart.
    """
    q = []
    mol = {}
    i = 0
    while i < len(formula):
        # Using a classic loop allow for manipulating the cursor
        token = formula[i]
        if token in CLOSERS:
            # Check for an index for this part
            m = re.match('\d+', formula[i+1:])
            if m:
                weight = int(m.group(0))
                i += len(m.group(0))
            else:
                weight = 1
            submol = _dictify(re.findall(ATOM_REGEX, ''.join(q)))
            return _fuse(mol, submol, weight), i
        elif token in OPENERS:
            submol, l = _parse(formula[i+1:])
            mol = _fuse(mol, submol)
            # skip the already read submol
            i += l + 1
        else:
            q.append(token)
        i+=1
    # Fuse in all that's left at base level
    return _fuse(mol, _dictify( re.findall(ATOM_REGEX, ''.join(q)) )), i
def chemparse(formula):
    # Decipher common organic chemistry jargon
    if 'Me' in formula: formula = formula.replace('Me','(CH3)')
    if 'Ac' in formula: formula = formula.replace('Ac','(CH3CO)')  # Note that Ac also means Actinium
    if 'Ph' in formula: formula = formula.replace('Ph','(C6H5)')
    if 'Bz' in formula: formula = formula.replace('Bz','(C6H5CH2)')
    if 'Cp' in formula: formula = formula.replace('Cp','(C5H5)')
    if 'Py' in formula: formula = formula.replace('Py','(C5H5N)')
    if 'Et' in formula: formula = formula.replace('Et','(C2H5)')
    if 'Pr' in formula: formula = formula.replace('Pr','(C3H7)')
    if 'Bu' in formula: formula = formula.replace('Bu','(C4H9)')
    """Parse the formula and return a dict with occurences of each atom."""
    if not is_balanced(formula):
        return "Watch your brackets ![{]$[&?)]}!]"
    return _parse(formula)[0]
def mass(formula):
    return np.sum(np.array(list( map( lambda x: md.element(x).atomic_weight ,np.array(list( formula.keys() )) ) )) * np.array(list( formula.values() )))

class SimulWorker(QtCore.QObject):
    
    # Define signals to be emitted to main GUI thread
    finished = QtCore.pyqtSignal()
    completed = QtCore.pyqtSignal(object)
    askmass = QtCore.pyqtSignal(object)
    asksimple = QtCore.pyqtSignal(object)
    
    def __init__(self, simulparam, parent = None):
        super(SimulWorker, self).__init__(parent)
        
        self.formula = parse(simulparam[0], ',')
        self.mzrange = parse(simulparam[1], ',', ty='flo')
        self.charge = parse(simulparam[2], ',')
        self.mode = 'Iso' if simulparam[3][0]=='I' else 'Ave'
        self.pol = 'Neg' if simulparam[4][0]=='N' else 'Pos'
        
        self.tmpX, self.tmpY = np.array([]), np.array([])
        self.SPECIAL = np.array( [(')'+i) for i in 'nmop'] )
        # print(simulparam, self.mode)
        
    def SimulRun(self):
        
        """ START """
        newformula, newcharge = [], []
        
        ## Unpacking formula
        for i in range(len(self.formula)):
            
            # Remove all complicated brackets
            self.formula[i] = self.formula[i].replace('[','(')
            self.formula[i] = self.formula[i].replace('{','(')
            self.formula[i] = self.formula[i].replace(']',')')
            self.formula[i] = self.formula[i].replace('}',')')
            
            if True in [ (j in self.formula[i]) for j in self.SPECIAL ]:
                
                # Figure out special character found in formula
                self.special = self.SPECIAL[ np.array([ (j in self.formula[i]) for j in self.SPECIAL ]) ]
                
                # Replace all special character occurence
                tmp = self.formula[i]
                for j in self.SPECIAL: tmp = tmp.replace(j,')1')
                print('bef',self.formula[i],'aft',tmp)
                
                # Inquire isotope mass and abundance list in mendeleev database
                self.wait = True
                self.masslist = []
                self.asksimple.emit( chemparse(tmp) )
                while self.wait:
                    time.sleep(0.1)
                
                # Contain formula
                self.tmp = self.formula[i]
                self.cha = abs(float(self.charge[i]))
                self.result = []
                
                # Recursive search!
                self.FormulaLoop([])
                
                # Reattach the result to formula and charge
                newformula = newformula + self.result
                newcharge = newcharge + ([self.charge[i]]* len(self.result))
                
            else:
                # Nothing happens - no iterator detected
                newformula = newformula + [self.formula[i]]
                newcharge = newcharge + [self.charge[i]]
        
        # New formula and charge list - prepared to be converted to mass
        self.formula = newformula[:]
        self.charge = newcharge[:]
        # print('formula', self.formula)
        # print('charge', self.charge)
        
        ## Time to process every one of the formula 
        self.label = []
        for i in range(len(self.formula)):
        
            """ RUN """
            ## Parse the string into chemical formula
            s = chemparse(self.formula[i])
            self.tmp = abs(float(self.charge[i]))
            print('bef',self.formula[i],'aft',s)
            
            ## Inquire isotope mass and abundance list in mendeleev database
            self.wait = True
            self.masslist = []
            self.askmass.emit( [s, self.mode] )
            while self.wait:
                time.sleep(0.1)
            # Trap the thread in infinite loop
            # to reduce CPU load we put the thread to sleep for 0.1sec
            
            if self.mode[0] == 'A':       # Compute ave mass and export to tmpX,Y
                self.tmpX = np.append( self.tmpX, np.sum(self.masslist)/self.tmp )
                self.tmpY = np.append( self.tmpY, np.product(self.abunlist) )
                self.label.append( [self.formula[i]+'('+str(self.charge[i])+')', np.sum(self.masslist)/self.tmp] )
            
            elif self.mode[0] == 'I':     # Compute isotope pattern and export to tmpX,Y
                self.maxabun = 1
                self.maxmass = 0
                for j in range(len(self.abunlist)):
                    self.maxabun = self.maxabun * max(self.abunlist[j])
                    self.maxmass += self.masslist[j][ self.abunlist[j].index(max(self.abunlist[j])) ]
                self.label.append( [self.formula[i]+'('+str(self.charge[i])+')', self.maxmass/self.tmp] )
                
                # Recursion
                self.IsoLoop(0, 1, [])
        
        ## Huge loop done - send it back to main thread
        self.completed.emit( [self.tmpX, self.tmpY, self.label] )
        
        """ EXIT """
        ## All formula has been parsed and simulated - thread will exit
        self.finished.emit()
        
    def IsoLoop(self, n, abun, his):
        for i in range(len(self.abunlist[n])):
            
            # CONTINUE CONDITION
            if n < len(self.abunlist)-1 and (abun*self.abunlist[n][i])/self.maxabun > 0.1:
                self.IsoLoop(n+1, abun*self.abunlist[n][i], his + [i])
            
            # STOP CONDITION
            elif n == len(self.abunlist)-1 and (abun*self.abunlist[n][i])/self.maxabun > 0.1:
                tmp = his + [i]
                finmass = sum(list( map( lambda x: self.masslist[x][tmp[x]], range(len(self.abunlist)) ) ))
                finabun = abun * self.abunlist[n][i]/self.maxabun
                # print('accepted', '%.2f' %finmass, '%.2f' %finabun)
                self.tmpX = np.append( self.tmpX, finmass/self.tmp )
                self.tmpY = np.append( self.tmpY, finabun )
    
    def Mass(self, formula):
        return sum([ (self.masslist[ i ] * formula[ i ] ) for i in list(formula.keys()) ])/self.cha
    
    def FormulaLoop(self, his):
        i = 0
        while True:
            
            # Contain initial formula
            tmp = self.tmp
            n = len(his)
            
            # Enter the index for current iterator
            tmp = tmp.replace( self.special[n], ')'+str(i) )
            
            # Enter the index for past iterator stored in variable 'his'
            if n>0:
                for j in range(n): tmp = tmp.replace( self.special[j], ')'+str(his[j]) )
            
            # Make sure all other iterator is gone
            for j in self.SPECIAL: tmp = tmp.replace(j, ')0')
            
            # Current mass
            # print(tmp, chemparse(tmp))
            nextmass = self.Mass( chemparse(tmp) )
            
            if nextmass <= self.mzrange[1] :
                # END CONDITION: TERMINAL --> SAVE DATA
                if n == len(self.special)-1 and nextmass >= self.mzrange[0]:
                    self.result.append(tmp)
                
                # CONTINUE CONDITION
                if n < len(self.special)-1:
                    self.FormulaLoop(his+[i])
                
            else:
                # END CONDITION: PREMATURE --> KILL LOOP
                break
            
            # Step forward index
            i+=1
            
            
"""
# ---- GUI from QtDesigner

HOW TO:
1. Paste py from pyuic5 conversion of QtDesigner ui file.
    Run at win/conda terminal: pyuic5 -x YOUR_UI.ui -o YOUR_PY.py

2. Create tofViewBox instance in setupUi
    Add
        self.v1 = tofViewBox()
        self.v2 = tofViewBox()

3. Modify your PlotWidget class in QtDesigner py file.
    Change
        self.PW1 = PlotWidget(self.centralwidget)
        self.PW2 = PlotWidget(self.centralwidget)
    To
        self.PW1 = PlotWidget(self.centralwidget, viewBox=self.v1)
        self.PW2 = PlotWidget(self.centralwidget, viewBox=self.v2)

4. Append the section termed 'THE NUTS AND BOLTS OF pyTOF' to the Ui_MainWindow class

"""
class Ui_MainWindow(object):
    
    def setupUi(self, MainWindow):
        
        ### HUMAN EDIT START
        self.v1 = tofViewBox()
        self.v2 = tofViewBox()
        self.MW = MainWindow
        ### HUMAN EDIT END
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1367, 891)
        MainWindow.setAcceptDrops(True)
        MainWindow.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout_7 = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout_7.setObjectName("gridLayout_7")
        self.PW1 = PlotWidget(self.centralwidget, viewBox=self.v1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.PW1.sizePolicy().hasHeightForWidth())
        self.PW1.setSizePolicy(sizePolicy)
        self.PW1.setMinimumSize(QtCore.QSize(791, 241))
        self.PW1.setObjectName("PW1")
        self.gridLayout_7.addWidget(self.PW1, 2, 0, 1, 8)
        self.PW2 = PlotWidget(self.centralwidget, viewBox=self.v2)
        self.PW2.setMinimumSize(QtCore.QSize(791, 191))
        self.PW2.setObjectName("PW2")
        self.gridLayout_7.addWidget(self.PW2, 3, 0, 1, 8)
        self.txt_statref = QtWidgets.QLabel(self.centralwidget)
        self.txt_statref.setObjectName("txt_statref")
        self.gridLayout_7.addWidget(self.txt_statref, 1, 1, 1, 1)
        self.line_9 = QtWidgets.QFrame(self.centralwidget)
        self.line_9.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_9.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_9.setObjectName("line_9")
        self.gridLayout_7.addWidget(self.line_9, 0, 2, 1, 1)
        self.verticalLayout_2 = QtWidgets.QVBoxLayout()
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.label_4 = QtWidgets.QLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setUnderline(True)
        self.label_4.setFont(font)
        self.label_4.setAlignment(QtCore.Qt.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.verticalLayout_2.addWidget(self.label_4)
        self.bu_pol = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_pol.sizePolicy().hasHeightForWidth())
        self.bu_pol.setSizePolicy(sizePolicy)
        self.bu_pol.setMinimumSize(QtCore.QSize(0, 16))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.bu_pol.setFont(font)
        self.bu_pol.setCheckable(True)
        self.bu_pol.setFlat(False)
        self.bu_pol.setObjectName("bu_pol")
        self.verticalLayout_2.addWidget(self.bu_pol)
        self.gridLayout_7.addLayout(self.verticalLayout_2, 0, 3, 1, 1)
        self.line_19 = QtWidgets.QFrame(self.centralwidget)
        self.line_19.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_19.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_19.setObjectName("line_19")
        self.gridLayout_7.addWidget(self.line_19, 0, 4, 1, 1)
        self.bu_clear = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_clear.sizePolicy().hasHeightForWidth())
        self.bu_clear.setSizePolicy(sizePolicy)
        self.bu_clear.setMinimumSize(QtCore.QSize(75, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.bu_clear.setFont(font)
        self.bu_clear.setObjectName("bu_clear")
        self.gridLayout_7.addWidget(self.bu_clear, 0, 7, 1, 1)
        self.bu_start = QtWidgets.QPushButton(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_start.sizePolicy().hasHeightForWidth())
        self.bu_start.setSizePolicy(sizePolicy)
        self.bu_start.setMinimumSize(QtCore.QSize(75, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.bu_start.setFont(font)
        self.bu_start.setCheckable(True)
        self.bu_start.setObjectName("bu_start")
        self.gridLayout_7.addWidget(self.bu_start, 0, 5, 1, 1)
        self.line = QtWidgets.QFrame(self.centralwidget)
        self.line.setFrameShape(QtWidgets.QFrame.VLine)
        self.line.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line.setObjectName("line")
        self.gridLayout_7.addWidget(self.line, 0, 6, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.Status_tof = QtWidgets.QLabel(self.centralwidget)
        self.Status_tof.setObjectName("Status_tof")
        self.horizontalLayout.addWidget(self.Status_tof)
        self.line_6 = QtWidgets.QFrame(self.centralwidget)
        self.line_6.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_6.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_6.setObjectName("line_6")
        self.horizontalLayout.addWidget(self.line_6)
        self.Status_digi = QtWidgets.QLabel(self.centralwidget)
        self.Status_digi.setObjectName("Status_digi")
        self.horizontalLayout.addWidget(self.Status_digi)
        self.line_7 = QtWidgets.QFrame(self.centralwidget)
        self.line_7.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_7.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_7.setObjectName("line_7")
        self.horizontalLayout.addWidget(self.line_7)
        self.Status_bme = QtWidgets.QLabel(self.centralwidget)
        self.Status_bme.setObjectName("Status_bme")
        self.horizontalLayout.addWidget(self.Status_bme)
        self.gridLayout_7.addLayout(self.horizontalLayout, 4, 0, 1, 8)
        self.tab = QtWidgets.QTabWidget(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.tab.sizePolicy().hasHeightForWidth())
        self.tab.setSizePolicy(sizePolicy)
        self.tab.setMinimumSize(QtCore.QSize(791, 91))
        self.tab.setMaximumSize(QtCore.QSize(16777215, 91))
        self.tab.setContextMenuPolicy(QtCore.Qt.ActionsContextMenu)
        self.tab.setToolTipDuration(0)
        self.tab.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.tab.setTabPosition(QtWidgets.QTabWidget.North)
        self.tab.setUsesScrollButtons(True)
        self.tab.setMovable(False)
        self.tab.setObjectName("tab")
        self.tab1 = QtWidgets.QWidget()
        self.tab1.setObjectName("tab1")
        self.gridLayout = QtWidgets.QGridLayout(self.tab1)
        self.gridLayout.setObjectName("gridLayout")
        self.label_12 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_12.setFont(font)
        self.label_12.setAlignment(QtCore.Qt.AlignCenter)
        self.label_12.setObjectName("label_12")
        self.gridLayout.addWidget(self.label_12, 0, 3, 1, 1)
        self.line_16 = QtWidgets.QFrame(self.tab1)
        self.line_16.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_16.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_16.setObjectName("line_16")
        self.gridLayout.addWidget(self.line_16, 0, 1, 2, 2)
        self.bu_save = QtWidgets.QPushButton(self.tab1)
        self.bu_save.setMinimumSize(QtCore.QSize(75, 41))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.bu_save.setFont(font)
        self.bu_save.setCheckable(False)
        self.bu_save.setObjectName("bu_save")
        self.gridLayout.addWidget(self.bu_save, 0, 0, 2, 1)
        self.label = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label.setFont(font)
        self.label.setAlignment(QtCore.Qt.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 11, 1, 1)
        self.label_2 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_2.setFont(font)
        self.label_2.setAlignment(QtCore.Qt.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 10, 1, 1)
        self.label_3 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        font.setKerning(True)
        self.label_3.setFont(font)
        self.label_3.setAlignment(QtCore.Qt.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 0, 6, 1, 1)
        self.line_17 = QtWidgets.QFrame(self.tab1)
        self.line_17.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_17.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_17.setObjectName("line_17")
        self.gridLayout.addWidget(self.line_17, 0, 4, 2, 1)
        self.label_6 = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setAlignment(QtCore.Qt.AlignCenter)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 5, 1, 1)
        self.txt_saveindex = QtWidgets.QLabel(self.tab1)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.txt_saveindex.setFont(font)
        self.txt_saveindex.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_saveindex.setObjectName("txt_saveindex")
        self.gridLayout.addWidget(self.txt_saveindex, 1, 5, 1, 1)
        self.bu_savedir = QtWidgets.QPushButton(self.tab1)
        self.bu_savedir.setCheckable(True)
        self.bu_savedir.setFlat(False)
        self.bu_savedir.setObjectName("bu_savedir")
        self.gridLayout.addWidget(self.bu_savedir, 1, 2, 1, 1)
        self.txt_savedir = QtWidgets.QLineEdit(self.tab1)
        self.txt_savedir.setInputMask("")
        self.txt_savedir.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_savedir.setReadOnly(True)
        self.txt_savedir.setClearButtonEnabled(False)
        self.txt_savedir.setObjectName("txt_savedir")
        self.gridLayout.addWidget(self.txt_savedir, 1, 3, 1, 1)
        self.txt_comm = QtWidgets.QLineEdit(self.tab1)
        self.txt_comm.setText("")
        self.txt_comm.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_comm.setObjectName("txt_comm")
        self.gridLayout.addWidget(self.txt_comm, 1, 6, 1, 1)
        self.txt_mol = QtWidgets.QLineEdit(self.tab1)
        self.txt_mol.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_mol.setObjectName("txt_mol")
        self.gridLayout.addWidget(self.txt_mol, 1, 11, 1, 1)
        self.txt_surf = QtWidgets.QLineEdit(self.tab1)
        self.txt_surf.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_surf.setObjectName("txt_surf")
        self.gridLayout.addWidget(self.txt_surf, 1, 10, 1, 1)
        self.txt_Q1 = QtWidgets.QLineEdit(self.tab1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_Q1.sizePolicy().hasHeightForWidth())
        self.txt_Q1.setSizePolicy(sizePolicy)
        self.txt_Q1.setMaximumSize(QtCore.QSize(50, 16777215))
        self.txt_Q1.setText("")
        self.txt_Q1.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_Q1.setObjectName("txt_Q1")
        self.gridLayout.addWidget(self.txt_Q1, 1, 9, 1, 1)
        self.txt_Q2 = QtWidgets.QLineEdit(self.tab1)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_Q2.sizePolicy().hasHeightForWidth())
        self.txt_Q2.setSizePolicy(sizePolicy)
        self.txt_Q2.setMaximumSize(QtCore.QSize(50, 16777215))
        self.txt_Q2.setText("")
        self.txt_Q2.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_Q2.setObjectName("txt_Q2")
        self.gridLayout.addWidget(self.txt_Q2, 1, 8, 1, 1)
        self.txt_UV = QtWidgets.QLineEdit(self.tab1)
        self.txt_UV.setMaximumSize(QtCore.QSize(50, 16777215))
        self.txt_UV.setText("")
        self.txt_UV.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_UV.setObjectName("txt_UV")
        self.gridLayout.addWidget(self.txt_UV, 1, 7, 1, 1)
        self.label_34 = QtWidgets.QLabel(self.tab1)
        self.label_34.setAlignment(QtCore.Qt.AlignCenter)
        self.label_34.setObjectName("label_34")
        self.gridLayout.addWidget(self.label_34, 0, 9, 1, 1)
        self.label_35 = QtWidgets.QLabel(self.tab1)
        self.label_35.setAlignment(QtCore.Qt.AlignCenter)
        self.label_35.setObjectName("label_35")
        self.gridLayout.addWidget(self.label_35, 0, 8, 1, 1)
        self.label_36 = QtWidgets.QLabel(self.tab1)
        self.label_36.setAlignment(QtCore.Qt.AlignCenter)
        self.label_36.setObjectName("label_36")
        self.gridLayout.addWidget(self.label_36, 0, 7, 1, 1)
        self.tab.addTab(self.tab1, "")
        self.tab_4 = QtWidgets.QWidget()
        self.tab_4.setObjectName("tab_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.tab_4)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.txt_smooth = QtWidgets.QLineEdit(self.tab_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_smooth.sizePolicy().hasHeightForWidth())
        self.txt_smooth.setSizePolicy(sizePolicy)
        self.txt_smooth.setMaximumSize(QtCore.QSize(100, 16777215))
        self.txt_smooth.setLayoutDirection(QtCore.Qt.RightToLeft)
        self.txt_smooth.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_smooth.setObjectName("txt_smooth")
        self.gridLayout_5.addWidget(self.txt_smooth, 1, 8, 1, 1)
        self.label_33 = QtWidgets.QLabel(self.tab_4)
        self.label_33.setAlignment(QtCore.Qt.AlignCenter)
        self.label_33.setObjectName("label_33")
        self.gridLayout_5.addWidget(self.label_33, 0, 8, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.tab_4)
        self.label_11.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.label_11.setObjectName("label_11")
        self.gridLayout_5.addWidget(self.label_11, 0, 21, 1, 1)
        self.line_15 = QtWidgets.QFrame(self.tab_4)
        self.line_15.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_15.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_15.setObjectName("line_15")
        self.gridLayout_5.addWidget(self.line_15, 0, 9, 2, 1)
        self.label_5 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_5.setFont(font)
        self.label_5.setAlignment(QtCore.Qt.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout_5.addWidget(self.label_5, 0, 12, 1, 1)
        self.label_10 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_10.setFont(font)
        self.label_10.setAlignment(QtCore.Qt.AlignCenter)
        self.label_10.setObjectName("label_10")
        self.gridLayout_5.addWidget(self.label_10, 0, 11, 1, 1)
        self.label_15 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_15.setFont(font)
        self.label_15.setAlignment(QtCore.Qt.AlignCenter)
        self.label_15.setObjectName("label_15")
        self.gridLayout_5.addWidget(self.label_15, 0, 10, 1, 1)
        self.label_16 = QtWidgets.QLabel(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_16.setFont(font)
        self.label_16.setAlignment(QtCore.Qt.AlignCenter)
        self.label_16.setObjectName("label_16")
        self.gridLayout_5.addWidget(self.label_16, 0, 14, 1, 1)
        self.txt_lo = QtWidgets.QLineEdit(self.tab_4)
        self.txt_lo.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_lo.setObjectName("txt_lo")
        self.gridLayout_5.addWidget(self.txt_lo, 1, 12, 1, 1)
        self.line_10 = QtWidgets.QFrame(self.tab_4)
        self.line_10.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_10.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_10.setObjectName("line_10")
        self.gridLayout_5.addWidget(self.line_10, 0, 20, 2, 1)
        self.com_trig = QtWidgets.QComboBox(self.tab_4)
        self.com_trig.setObjectName("com_trig")
        self.com_trig.addItem("")
        self.com_trig.addItem("")
        self.gridLayout_5.addWidget(self.com_trig, 1, 21, 1, 1)
        self.txt_time = QtWidgets.QLineEdit(self.tab_4)
        self.txt_time.setInputMask("")
        self.txt_time.setText("")
        self.txt_time.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_time.setClearButtonEnabled(False)
        self.txt_time.setObjectName("txt_time")
        self.gridLayout_5.addWidget(self.txt_time, 1, 14, 1, 1)
        self.line_14 = QtWidgets.QFrame(self.tab_4)
        self.line_14.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_14.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_14.setObjectName("line_14")
        self.gridLayout_5.addWidget(self.line_14, 0, 2, 2, 1)
        self.txt_ave = QtWidgets.QLineEdit(self.tab_4)
        self.txt_ave.setInputMask("")
        self.txt_ave.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_ave.setClearButtonEnabled(False)
        self.txt_ave.setObjectName("txt_ave")
        self.gridLayout_5.addWidget(self.txt_ave, 1, 10, 1, 1)
        self.che_abs = QtWidgets.QCheckBox(self.tab_4)
        self.che_abs.setObjectName("che_abs")
        self.gridLayout_5.addWidget(self.che_abs, 0, 5, 1, 1)
        self.che_calib = QtWidgets.QCheckBox(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.che_calib.setFont(font)
        self.che_calib.setChecked(True)
        self.che_calib.setObjectName("che_calib")
        self.gridLayout_5.addWidget(self.che_calib, 0, 3, 1, 1)
        self.che_inv = QtWidgets.QCheckBox(self.tab_4)
        self.che_inv.setChecked(True)
        self.che_inv.setObjectName("che_inv")
        self.gridLayout_5.addWidget(self.che_inv, 1, 5, 1, 1)
        self.txt_hi = QtWidgets.QLineEdit(self.tab_4)
        self.txt_hi.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_hi.setObjectName("txt_hi")
        self.gridLayout_5.addWidget(self.txt_hi, 1, 11, 1, 1)
        self.che_baserem = QtWidgets.QCheckBox(self.tab_4)
        self.che_baserem.setChecked(True)
        self.che_baserem.setObjectName("che_baserem")
        self.gridLayout_5.addWidget(self.che_baserem, 1, 3, 1, 1)
        self.che_mode = QtWidgets.QCheckBox(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.che_mode.setFont(font)
        self.che_mode.setObjectName("che_mode")
        self.gridLayout_5.addWidget(self.che_mode, 0, 0, 1, 1)
        self.line_21 = QtWidgets.QFrame(self.tab_4)
        self.line_21.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_21.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_21.setObjectName("line_21")
        self.gridLayout_5.addWidget(self.line_21, 0, 1, 2, 1)
        self.line_20 = QtWidgets.QFrame(self.tab_4)
        self.line_20.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_20.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_20.setObjectName("line_20")
        self.gridLayout_5.addWidget(self.line_20, 0, 6, 2, 1)
        self.che_diff = QtWidgets.QCheckBox(self.tab_4)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.che_diff.setFont(font)
        self.che_diff.setChecked(False)
        self.che_diff.setObjectName("che_diff")
        self.gridLayout_5.addWidget(self.che_diff, 1, 0, 1, 1)
        self.tab.addTab(self.tab_4, "")
        self.tab_3 = QtWidgets.QWidget()
        self.tab_3.setObjectName("tab_3")
        self.gridLayout_6 = QtWidgets.QGridLayout(self.tab_3)
        self.gridLayout_6.setObjectName("gridLayout_6")
        self.label_18 = QtWidgets.QLabel(self.tab_3)
        self.label_18.setAlignment(QtCore.Qt.AlignCenter)
        self.label_18.setObjectName("label_18")
        self.gridLayout_6.addWidget(self.label_18, 0, 2, 1, 1)
        self.label_17 = QtWidgets.QLabel(self.tab_3)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_17.setFont(font)
        self.label_17.setAlignment(QtCore.Qt.AlignCenter)
        self.label_17.setObjectName("label_17")
        self.gridLayout_6.addWidget(self.label_17, 0, 0, 1, 1)
        self.line_11 = QtWidgets.QFrame(self.tab_3)
        self.line_11.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_11.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_11.setObjectName("line_11")
        self.gridLayout_6.addWidget(self.line_11, 0, 1, 2, 1)
        self.label_26 = QtWidgets.QLabel(self.tab_3)
        self.label_26.setAlignment(QtCore.Qt.AlignCenter)
        self.label_26.setObjectName("label_26")
        self.gridLayout_6.addWidget(self.label_26, 0, 4, 1, 1)
        self.line_12 = QtWidgets.QFrame(self.tab_3)
        self.line_12.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_12.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_12.setObjectName("line_12")
        self.gridLayout_6.addWidget(self.line_12, 0, 5, 2, 1)
        self.label_19 = QtWidgets.QLabel(self.tab_3)
        self.label_19.setAlignment(QtCore.Qt.AlignCenter)
        self.label_19.setObjectName("label_19")
        self.gridLayout_6.addWidget(self.label_19, 0, 3, 1, 1)
        self.label_21 = QtWidgets.QLabel(self.tab_3)
        self.label_21.setAlignment(QtCore.Qt.AlignCenter)
        self.label_21.setObjectName("label_21")
        self.gridLayout_6.addWidget(self.label_21, 0, 7, 1, 1)
        self.txt_dBwidth = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dBwidth.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dBwidth.setObjectName("txt_dBwidth")
        self.gridLayout_6.addWidget(self.txt_dBwidth, 1, 7, 1, 1)
        self.label_25 = QtWidgets.QLabel(self.tab_3)
        self.label_25.setAlignment(QtCore.Qt.AlignCenter)
        self.label_25.setObjectName("label_25")
        self.gridLayout_6.addWidget(self.label_25, 0, 9, 1, 1)
        self.line_13 = QtWidgets.QFrame(self.tab_3)
        self.line_13.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_13.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_13.setObjectName("line_13")
        self.gridLayout_6.addWidget(self.line_13, 0, 10, 2, 1)
        self.che_dA = QtWidgets.QCheckBox(self.tab_3)
        self.che_dA.setChecked(True)
        self.che_dA.setObjectName("che_dA")
        self.gridLayout_6.addWidget(self.che_dA, 1, 4, 1, 1)
        self.txt_dRep = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dRep.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dRep.setObjectName("txt_dRep")
        self.gridLayout_6.addWidget(self.txt_dRep, 1, 0, 1, 1)
        self.txt_dCdelay = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dCdelay.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dCdelay.setObjectName("txt_dCdelay")
        self.gridLayout_6.addWidget(self.txt_dCdelay, 1, 11, 1, 1)
        self.label_22 = QtWidgets.QLabel(self.tab_3)
        self.label_22.setAlignment(QtCore.Qt.AlignCenter)
        self.label_22.setObjectName("label_22")
        self.gridLayout_6.addWidget(self.label_22, 0, 11, 1, 1)
        self.label_23 = QtWidgets.QLabel(self.tab_3)
        self.label_23.setAlignment(QtCore.Qt.AlignCenter)
        self.label_23.setObjectName("label_23")
        self.gridLayout_6.addWidget(self.label_23, 0, 12, 1, 1)
        self.che_dC = QtWidgets.QCheckBox(self.tab_3)
        self.che_dC.setObjectName("che_dC")
        self.gridLayout_6.addWidget(self.che_dC, 1, 13, 1, 1)
        self.che_dB = QtWidgets.QCheckBox(self.tab_3)
        self.che_dB.setChecked(True)
        self.che_dB.setObjectName("che_dB")
        self.gridLayout_6.addWidget(self.che_dB, 1, 9, 1, 1)
        self.txt_dAwidth = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dAwidth.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dAwidth.setObjectName("txt_dAwidth")
        self.gridLayout_6.addWidget(self.txt_dAwidth, 1, 3, 1, 1)
        self.txt_dCwidth = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dCwidth.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dCwidth.setObjectName("txt_dCwidth")
        self.gridLayout_6.addWidget(self.txt_dCwidth, 1, 12, 1, 1)
        self.label_24 = QtWidgets.QLabel(self.tab_3)
        self.label_24.setAlignment(QtCore.Qt.AlignCenter)
        self.label_24.setObjectName("label_24")
        self.gridLayout_6.addWidget(self.label_24, 0, 13, 1, 1)
        self.txt_dAdelay = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dAdelay.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dAdelay.setObjectName("txt_dAdelay")
        self.gridLayout_6.addWidget(self.txt_dAdelay, 1, 2, 1, 1)
        self.label_20 = QtWidgets.QLabel(self.tab_3)
        self.label_20.setAlignment(QtCore.Qt.AlignCenter)
        self.label_20.setObjectName("label_20")
        self.gridLayout_6.addWidget(self.label_20, 0, 6, 1, 1)
        self.txt_dBdelay = QtWidgets.QLineEdit(self.tab_3)
        self.txt_dBdelay.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_dBdelay.setObjectName("txt_dBdelay")
        self.gridLayout_6.addWidget(self.txt_dBdelay, 1, 6, 1, 1)
        self.tab.addTab(self.tab_3, "")
        self.tab2 = QtWidgets.QWidget()
        self.tab2.setObjectName("tab2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.tab2)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_7 = QtWidgets.QLabel(self.tab2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_7.setFont(font)
        self.label_7.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_7.setObjectName("label_7")
        self.gridLayout_3.addWidget(self.label_7, 0, 8, 1, 1)
        self.line_2 = QtWidgets.QFrame(self.tab2)
        self.line_2.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_2.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_2.setObjectName("line_2")
        self.gridLayout_3.addWidget(self.line_2, 0, 10, 2, 1)
        self.txt_posA = QtWidgets.QLineEdit(self.tab2)
        self.txt_posA.setObjectName("txt_posA")
        self.gridLayout_3.addWidget(self.txt_posA, 0, 7, 1, 1)
        self.bu_saveini = QtWidgets.QPushButton(self.tab2)
        self.bu_saveini.setCheckable(True)
        self.bu_saveini.setFlat(False)
        self.bu_saveini.setObjectName("bu_saveini")
        self.gridLayout_3.addWidget(self.bu_saveini, 0, 13, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.tab2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_8.setFont(font)
        self.label_8.setAlignment(QtCore.Qt.AlignLeading|QtCore.Qt.AlignLeft|QtCore.Qt.AlignVCenter)
        self.label_8.setObjectName("label_8")
        self.gridLayout_3.addWidget(self.label_8, 1, 8, 1, 1)
        self.txt_initdir = QtWidgets.QLineEdit(self.tab2)
        self.txt_initdir.setInputMask("")
        self.txt_initdir.setAlignment(QtCore.Qt.AlignCenter)
        self.txt_initdir.setReadOnly(True)
        self.txt_initdir.setClearButtonEnabled(False)
        self.txt_initdir.setObjectName("txt_initdir")
        self.gridLayout_3.addWidget(self.txt_initdir, 1, 12, 1, 1)
        self.label_13 = QtWidgets.QLabel(self.tab2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_13.setFont(font)
        self.label_13.setAlignment(QtCore.Qt.AlignCenter)
        self.label_13.setObjectName("label_13")
        self.gridLayout_3.addWidget(self.label_13, 0, 12, 1, 1)
        self.txt_posB = QtWidgets.QLineEdit(self.tab2)
        self.txt_posB.setObjectName("txt_posB")
        self.gridLayout_3.addWidget(self.txt_posB, 0, 6, 1, 1)
        self.bu_initdir = QtWidgets.QPushButton(self.tab2)
        self.bu_initdir.setCheckable(True)
        self.bu_initdir.setFlat(False)
        self.bu_initdir.setObjectName("bu_initdir")
        self.gridLayout_3.addWidget(self.bu_initdir, 1, 13, 1, 1)
        self.txt_negA = QtWidgets.QLineEdit(self.tab2)
        self.txt_negA.setObjectName("txt_negA")
        self.gridLayout_3.addWidget(self.txt_negA, 1, 7, 1, 1)
        self.txt_negB = QtWidgets.QLineEdit(self.tab2)
        self.txt_negB.setObjectName("txt_negB")
        self.gridLayout_3.addWidget(self.txt_negB, 1, 6, 1, 1)
        self.txt_posC = QtWidgets.QLineEdit(self.tab2)
        self.txt_posC.setObjectName("txt_posC")
        self.gridLayout_3.addWidget(self.txt_posC, 0, 5, 1, 1)
        self.bu_decalib = QtWidgets.QPushButton(self.tab2)
        self.bu_decalib.setObjectName("bu_decalib")
        self.gridLayout_3.addWidget(self.bu_decalib, 1, 4, 1, 1)
        self.bu_calib = QtWidgets.QPushButton(self.tab2)
        self.bu_calib.setObjectName("bu_calib")
        self.gridLayout_3.addWidget(self.bu_calib, 0, 4, 1, 1)
        self.txt_negC = QtWidgets.QLineEdit(self.tab2)
        self.txt_negC.setObjectName("txt_negC")
        self.gridLayout_3.addWidget(self.txt_negC, 1, 5, 1, 1)
        self.tab.addTab(self.tab2, "")
        self.tab_2 = QtWidgets.QWidget()
        self.tab_2.setObjectName("tab_2")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.tab_2)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_9 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_9.setFont(font)
        self.label_9.setAlignment(QtCore.Qt.AlignCenter)
        self.label_9.setObjectName("label_9")
        self.gridLayout_4.addWidget(self.label_9, 0, 2, 1, 1)
        self.txt_calibtof = QtWidgets.QLineEdit(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_calibtof.sizePolicy().hasHeightForWidth())
        self.txt_calibtof.setSizePolicy(sizePolicy)
        self.txt_calibtof.setObjectName("txt_calibtof")
        self.gridLayout_4.addWidget(self.txt_calibtof, 0, 3, 1, 1)
        self.line_3 = QtWidgets.QFrame(self.tab_2)
        self.line_3.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_3.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_3.setObjectName("line_3")
        self.gridLayout_4.addWidget(self.line_3, 0, 6, 2, 1)
        self.bu_calibgo = QtWidgets.QPushButton(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_calibgo.sizePolicy().hasHeightForWidth())
        self.bu_calibgo.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setBold(False)
        font.setUnderline(False)
        font.setWeight(50)
        self.bu_calibgo.setFont(font)
        self.bu_calibgo.setObjectName("bu_calibgo")
        self.gridLayout_4.addWidget(self.bu_calibgo, 0, 7, 2, 1)
        self.txt_calibresult = QtWidgets.QTextEdit(self.tab_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Maximum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_calibresult.sizePolicy().hasHeightForWidth())
        self.txt_calibresult.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(7)
        self.txt_calibresult.setFont(font)
        self.txt_calibresult.setReadOnly(True)
        self.txt_calibresult.setAcceptRichText(False)
        self.txt_calibresult.setObjectName("txt_calibresult")
        self.gridLayout_4.addWidget(self.txt_calibresult, 0, 8, 2, 1)
        self.label_14 = QtWidgets.QLabel(self.tab_2)
        font = QtGui.QFont()
        font.setBold(True)
        font.setUnderline(True)
        font.setWeight(75)
        self.label_14.setFont(font)
        self.label_14.setAlignment(QtCore.Qt.AlignCenter)
        self.label_14.setObjectName("label_14")
        self.gridLayout_4.addWidget(self.label_14, 1, 2, 1, 1)
        self.txt_calibmz = QtWidgets.QLineEdit(self.tab_2)
        self.txt_calibmz.setObjectName("txt_calibmz")
        self.gridLayout_4.addWidget(self.txt_calibmz, 1, 3, 1, 1)
        self.che_calibmzreceive = QtWidgets.QCheckBox(self.tab_2)
        self.che_calibmzreceive.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.che_calibmzreceive.setChecked(True)
        self.che_calibmzreceive.setObjectName("che_calibmzreceive")
        self.gridLayout_4.addWidget(self.che_calibmzreceive, 1, 1, 1, 1)
        self.che_calibtofreceive = QtWidgets.QCheckBox(self.tab_2)
        self.che_calibtofreceive.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.che_calibtofreceive.setObjectName("che_calibtofreceive")
        self.gridLayout_4.addWidget(self.che_calibtofreceive, 0, 1, 1, 1)
        self.tab.addTab(self.tab_2, "")
        self.tab_5 = QtWidgets.QWidget()
        self.tab_5.setObjectName("tab_5")
        self.gridLayout_2 = QtWidgets.QGridLayout(self.tab_5)
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.label_27 = QtWidgets.QLabel(self.tab_5)
        self.label_27.setObjectName("label_27")
        self.gridLayout_2.addWidget(self.label_27, 0, 4, 1, 1)
        self.label_28 = QtWidgets.QLabel(self.tab_5)
        self.label_28.setObjectName("label_28")
        self.gridLayout_2.addWidget(self.label_28, 0, 5, 1, 1)
        self.label_32 = QtWidgets.QLabel(self.tab_5)
        self.label_32.setObjectName("label_32")
        self.gridLayout_2.addWidget(self.label_32, 0, 6, 1, 1)
        self.label_30 = QtWidgets.QLabel(self.tab_5)
        self.label_30.setObjectName("label_30")
        self.gridLayout_2.addWidget(self.label_30, 1, 17, 1, 1)
        self.label_29 = QtWidgets.QLabel(self.tab_5)
        self.label_29.setObjectName("label_29")
        self.gridLayout_2.addWidget(self.label_29, 0, 17, 1, 1)
        self.label_31 = QtWidgets.QLabel(self.tab_5)
        self.label_31.setObjectName("label_31")
        self.gridLayout_2.addWidget(self.label_31, 1, 13, 1, 1)
        self.line_5 = QtWidgets.QFrame(self.tab_5)
        self.line_5.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_5.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_5.setObjectName("line_5")
        self.gridLayout_2.addWidget(self.line_5, 0, 7, 2, 1)
        self.line_4 = QtWidgets.QFrame(self.tab_5)
        self.line_4.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_4.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_4.setObjectName("line_4")
        self.gridLayout_2.addWidget(self.line_4, 0, 1, 2, 1)
        self.com_simulmode = QtWidgets.QComboBox(self.tab_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Preferred, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.com_simulmode.sizePolicy().hasHeightForWidth())
        self.com_simulmode.setSizePolicy(sizePolicy)
        self.com_simulmode.setInsertPolicy(QtWidgets.QComboBox.NoInsert)
        self.com_simulmode.setObjectName("com_simulmode")
        self.com_simulmode.addItem("")
        self.com_simulmode.addItem("")
        self.gridLayout_2.addWidget(self.com_simulmode, 0, 9, 2, 1)
        self.bu_loadref = QtWidgets.QPushButton(self.tab_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_loadref.sizePolicy().hasHeightForWidth())
        self.bu_loadref.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.bu_loadref.setFont(font)
        self.bu_loadref.setCheckable(True)
        self.bu_loadref.setFlat(False)
        self.bu_loadref.setObjectName("bu_loadref")
        self.gridLayout_2.addWidget(self.bu_loadref, 0, 0, 2, 1)
        self.txt_simulformula = QtWidgets.QLineEdit(self.tab_5)
        self.txt_simulformula.setObjectName("txt_simulformula")
        self.gridLayout_2.addWidget(self.txt_simulformula, 0, 11, 1, 6)
        self.line_18 = QtWidgets.QFrame(self.tab_5)
        self.line_18.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_18.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_18.setObjectName("line_18")
        self.gridLayout_2.addWidget(self.line_18, 1, 14, 1, 1)
        self.bu_simul = QtWidgets.QPushButton(self.tab_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_simul.sizePolicy().hasHeightForWidth())
        self.bu_simul.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.bu_simul.setFont(font)
        self.bu_simul.setCheckable(True)
        self.bu_simul.setFlat(False)
        self.bu_simul.setObjectName("bu_simul")
        self.gridLayout_2.addWidget(self.bu_simul, 0, 8, 2, 1)
        self.txt_simulmz = QtWidgets.QLineEdit(self.tab_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.txt_simulmz.sizePolicy().hasHeightForWidth())
        self.txt_simulmz.setSizePolicy(sizePolicy)
        self.txt_simulmz.setObjectName("txt_simulmz")
        self.gridLayout_2.addWidget(self.txt_simulmz, 1, 12, 1, 1)
        self.txt_simulcharge = QtWidgets.QLineEdit(self.tab_5)
        self.txt_simulcharge.setText("")
        self.txt_simulcharge.setObjectName("txt_simulcharge")
        self.gridLayout_2.addWidget(self.txt_simulcharge, 1, 16, 1, 1)
        self.che_simuluseplot = QtWidgets.QCheckBox(self.tab_5)
        self.che_simuluseplot.setLayoutDirection(QtCore.Qt.LeftToRight)
        self.che_simuluseplot.setChecked(True)
        self.che_simuluseplot.setObjectName("che_simuluseplot")
        self.gridLayout_2.addWidget(self.che_simuluseplot, 1, 11, 1, 1)
        self.line_8 = QtWidgets.QFrame(self.tab_5)
        self.line_8.setFrameShape(QtWidgets.QFrame.VLine)
        self.line_8.setFrameShadow(QtWidgets.QFrame.Sunken)
        self.line_8.setObjectName("line_8")
        self.gridLayout_2.addWidget(self.line_8, 0, 3, 2, 1)
        self.txt_peak1 = QtWidgets.QLabel(self.tab_5)
        self.txt_peak1.setObjectName("txt_peak1")
        self.gridLayout_2.addWidget(self.txt_peak1, 1, 6, 1, 1)
        self.txt_peak2 = QtWidgets.QLabel(self.tab_5)
        self.txt_peak2.setObjectName("txt_peak2")
        self.gridLayout_2.addWidget(self.txt_peak2, 1, 5, 1, 1)
        self.txt_peakdiff = QtWidgets.QLabel(self.tab_5)
        font = QtGui.QFont()
        font.setBold(True)
        font.setWeight(75)
        self.txt_peakdiff.setFont(font)
        self.txt_peakdiff.setObjectName("txt_peakdiff")
        self.gridLayout_2.addWidget(self.txt_peakdiff, 1, 4, 1, 1)
        self.bu_clearref = QtWidgets.QPushButton(self.tab_5)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.bu_clearref.sizePolicy().hasHeightForWidth())
        self.bu_clearref.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setBold(True)
        font.setWeight(75)
        self.bu_clearref.setFont(font)
        self.bu_clearref.setObjectName("bu_clearref")
        self.gridLayout_2.addWidget(self.bu_clearref, 0, 2, 2, 1)
        self.tab.addTab(self.tab_5, "")
        self.gridLayout_7.addWidget(self.tab, 0, 1, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        self.tab.setCurrentIndex(0)
        self.com_simulmode.setCurrentIndex(0)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "pyTOF"))
        self.txt_statref.setText(_translate("MainWindow", "Spectra selected: None"))
        self.label_4.setText(_translate("MainWindow", "Polarity"))
        self.bu_pol.setToolTip(_translate("MainWindow", "<html><head/><body><p>Current beam polarity</p></body></html>"))
        self.bu_pol.setText(_translate("MainWindow", "Pos"))
        self.bu_clear.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Clear existing data (CTRL+W)</span></p></body></html>"))
        self.bu_clear.setText(_translate("MainWindow", "Clear"))
        self.bu_start.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Start/Stop data acquisition (CTRL+R)</span></p></body></html>"))
        self.bu_start.setText(_translate("MainWindow", "Start"))
        self.Status_tof.setText(_translate("MainWindow", "Tof thread: Idle"))
        self.Status_digi.setText(_translate("MainWindow", "Digitizer: Not Connected"))
        self.Status_bme.setText(_translate("MainWindow", "DelayGen: Not Connected"))
        self.label_12.setText(_translate("MainWindow", "Save dir:"))
        self.bu_save.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:8pt; font-weight:400;\">Save cumulative spectra on SaveDir (CTRL+S)</span></p></body></html>"))
        self.bu_save.setText(_translate("MainWindow", "Save"))
        self.label.setText(_translate("MainWindow", "Molecule"))
        self.label_2.setText(_translate("MainWindow", "Surface"))
        self.label_3.setText(_translate("MainWindow", "Comment "))
        self.label_6.setText(_translate("MainWindow", "Index"))
        self.txt_saveindex.setToolTip(_translate("MainWindow", "<html><head/><body><p>Indexing to prevent file duplicate</p></body></html>"))
        self.txt_saveindex.setText(_translate("MainWindow", "0"))
        self.bu_savedir.setToolTip(_translate("MainWindow", "<html><head/><body><p>Define SaveDir</p></body></html>"))
        self.bu_savedir.setText(_translate("MainWindow", "..."))
        self.txt_savedir.setToolTip(_translate("MainWindow", "<html><head/><body><p>Save directory</p></body></html>"))
        self.txt_savedir.setText(_translate("MainWindow", "./"))
        self.txt_savedir.setPlaceholderText(_translate("MainWindow", "Save directory"))
        self.txt_comm.setToolTip(_translate("MainWindow", "<html><head/><body><p>Emitter,CID,IFRF,Feelings?</p></body></html>"))
        self.txt_comm.setPlaceholderText(_translate("MainWindow", "Emitter,CID,IFRF, Feelings?"))
        self.txt_mol.setToolTip(_translate("MainWindow", "<html><head/><body><p>Mol in spray solution</p></body></html>"))
        self.txt_mol.setPlaceholderText(_translate("MainWindow", "Molecule"))
        self.txt_surf.setToolTip(_translate("MainWindow", "<html><head/><body><p>Surface system</p></body></html>"))
        self.txt_surf.setPlaceholderText(_translate("MainWindow", "Surface"))
        self.txt_Q1.setPlaceholderText(_translate("MainWindow", "80"))
        self.txt_Q2.setPlaceholderText(_translate("MainWindow", "80"))
        self.txt_UV.setPlaceholderText(_translate("MainWindow", "0"))
        self.label_34.setText(_translate("MainWindow", "Q1"))
        self.label_35.setText(_translate("MainWindow", "Q2"))
        self.label_36.setText(_translate("MainWindow", "UV"))
        self.tab.setTabText(self.tab.indexOf(self.tab1), _translate("MainWindow", "Acq: File Save"))
        self.txt_smooth.setToolTip(_translate("MainWindow", "<html><head/><body><p>Perform LOW PASS filter on incoming signal.</p><p>0 = NO FILTERING<br/>0.1 = SAFE<br/>0.075 = OPTIMAL<br/>0.05 = MAY GIVE ARTIFACTS - USE AT YOUR OWN RISK!</p></body></html>"))
        self.txt_smooth.setText(_translate("MainWindow", "0.1"))
        self.label_33.setText(_translate("MainWindow", "Low pass filter"))
        self.label_11.setText(_translate("MainWindow", "Digitizer Trigger"))
        self.label_5.setText(_translate("MainWindow", "Low mz cutoff"))
        self.label_10.setText(_translate("MainWindow", "Hi mz cutoff"))
        self.label_15.setText(_translate("MainWindow", "Average"))
        self.label_16.setText(_translate("MainWindow", "Update time"))
        self.txt_lo.setToolTip(_translate("MainWindow", "<html><head/><body><p>Trim spectra below this limit</p></body></html>"))
        self.txt_lo.setText(_translate("MainWindow", "50"))
        self.txt_lo.setPlaceholderText(_translate("MainWindow", "Lo-cutoff (e.g. 50)"))
        self.com_trig.setToolTip(_translate("MainWindow", "<html><head/><body><p>What defines t=0 of digitizer?</p></body></html>"))
        self.com_trig.setItemText(0, _translate("MainWindow", "External (from BME_card)"))
        self.com_trig.setItemText(1, _translate("MainWindow", "Software (demo)"))
        self.txt_time.setToolTip(_translate("MainWindow", "<html><head/><body><p>Replot after n msec (600ms OK, 0ms is realtime and CPU intensive)</p></body></html>"))
        self.txt_time.setPlaceholderText(_translate("MainWindow", "0"))
        self.txt_ave.setToolTip(_translate("MainWindow", "<html><head/><body><p>Sum n spectra before plotting (3000 is recommended)</p></body></html>"))
        self.txt_ave.setText(_translate("MainWindow", "3000"))
        self.txt_ave.setPlaceholderText(_translate("MainWindow", "0"))
        self.che_abs.setToolTip(_translate("MainWindow", "<html><head/><body><p>Apply absolute to y-axis of TOF spectra</p></body></html>"))
        self.che_abs.setText(_translate("MainWindow", "Absolute Y"))
        self.che_calib.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibrate x-axis of TOF spectra</p></body></html>"))
        self.che_calib.setText(_translate("MainWindow", "Calib TOF"))
        self.che_inv.setToolTip(_translate("MainWindow", "<html><head/><body><p>Invert y-axis and remove negative y-values</p></body></html>"))
        self.che_inv.setText(_translate("MainWindow", "Invert Y"))
        self.txt_hi.setToolTip(_translate("MainWindow", "<html><head/><body><p>Trim spectra above this limit</p></body></html>"))
        self.txt_hi.setText(_translate("MainWindow", "4000"))
        self.txt_hi.setPlaceholderText(_translate("MainWindow", "Hi-cutoff (e.g. 3000)"))
        self.che_baserem.setToolTip(_translate("MainWindow", "<html><head/><body><p>Remove baseline of y-axis</p></body></html>"))
        self.che_baserem.setText(_translate("MainWindow", "Remove baseline"))
        self.che_mode.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:400;\">Obtain TOF data as far as 100us (mz 10 to 41.1k), instead of 50us (mz 10 to 10.8k) - note: time resolution halved, acquisition time doubled</span></p></body></html>"))
        self.che_mode.setText(_translate("MainWindow", "Hi mz-range"))
        self.che_diff.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:400;\">Differentiate TOF spectra</span></p></body></html>"))
        self.che_diff.setText(_translate("MainWindow", "Use dV/dt"))
        self.tab.setTabText(self.tab.indexOf(self.tab_4), _translate("MainWindow", "Acq: Digitizer Setting"))
        self.label_18.setText(_translate("MainWindow", "ChA_delay"))
        self.label_17.setText(_translate("MainWindow", "Rep Rate"))
        self.label_26.setText(_translate("MainWindow", "ChA_pol"))
        self.label_19.setText(_translate("MainWindow", "ChA_width"))
        self.label_21.setText(_translate("MainWindow", "ChB_width"))
        self.txt_dBwidth.setToolTip(_translate("MainWindow", "<html><head/><body><p>Controls the push-pull trigger</p></body></html>"))
        self.txt_dBwidth.setText(_translate("MainWindow", "56"))
        self.label_25.setText(_translate("MainWindow", "ChB_pol"))
        self.che_dA.setText(_translate("MainWindow", "POS"))
        self.txt_dRep.setToolTip(_translate("MainWindow", "<html><head/><body><p>Repeat cycle of the whole measurement (usec)</p></body></html>"))
        self.txt_dRep.setText(_translate("MainWindow", "111"))
        self.txt_dCdelay.setToolTip(_translate("MainWindow", "<html><head/><body><p>Controls the push-pull trigger</p></body></html>"))
        self.txt_dCdelay.setText(_translate("MainWindow", "0"))
        self.label_22.setText(_translate("MainWindow", "ChC_delay"))
        self.label_23.setText(_translate("MainWindow", "ChC_width"))
        self.che_dC.setText(_translate("MainWindow", "POS"))
        self.che_dB.setText(_translate("MainWindow", "POS"))
        self.txt_dAwidth.setToolTip(_translate("MainWindow", "<html><head/><body><p>Controls the t=0 for digitizer</p></body></html>"))
        self.txt_dAwidth.setText(_translate("MainWindow", "56"))
        self.txt_dCwidth.setToolTip(_translate("MainWindow", "<html><head/><body><p>Controls the push-pull trigger</p></body></html>"))
        self.txt_dCwidth.setText(_translate("MainWindow", "56"))
        self.label_24.setText(_translate("MainWindow", "ChC_pol"))
        self.txt_dAdelay.setToolTip(_translate("MainWindow", "<html><head/><body><p>Controls the t=0 for digitizer</p></body></html>"))
        self.txt_dAdelay.setText(_translate("MainWindow", "0"))
        self.label_20.setText(_translate("MainWindow", "ChB_delay"))
        self.txt_dBdelay.setToolTip(_translate("MainWindow", "<html><head/><body><p>Controls the push-pull trigger</p></body></html>"))
        self.txt_dBdelay.setText(_translate("MainWindow", "0"))
        self.tab.setTabText(self.tab.indexOf(self.tab_3), _translate("MainWindow", "Acq: DelayGen Setting"))
        self.label_7.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:400;\">Equation is y = Ax^2 + Bx + C</span></p></body></html>"))
        self.label_7.setText(_translate("MainWindow", "Calib POS:"))
        self.txt_posA.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibration constant (A) to convert time-of-flight (ns) to mass-over-charge ratio (m/z)</p></body></html>"))
        self.txt_posA.setPlaceholderText(_translate("MainWindow", "A-Calib"))
        self.bu_saveini.setToolTip(_translate("MainWindow", "<html><head/><body><p>Save init file</p></body></html>"))
        self.bu_saveini.setText(_translate("MainWindow", "Save config"))
        self.label_8.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-weight:400;\">Equation is y = Ax^2 + Bx + C</span></p></body></html>"))
        self.label_8.setText(_translate("MainWindow", "Calib NEG:"))
        self.txt_initdir.setToolTip(_translate("MainWindow", "<html><head/><body><p>Where pyTOF will find initial parameters.</p></body></html>"))
        self.txt_initdir.setText(_translate("MainWindow", "./PyTOF.ini"))
        self.txt_initdir.setPlaceholderText(_translate("MainWindow", "pyTOF ini file"))
        self.label_13.setText(_translate("MainWindow", "Init URL:"))
        self.txt_posB.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibration constant (B) to convert time-of-flight (ns) to mass-over-charge ratio (m/z)</p></body></html>"))
        self.txt_posB.setPlaceholderText(_translate("MainWindow", "B-Calib"))
        self.bu_initdir.setToolTip(_translate("MainWindow", "<html><head/><body><p>Load init file</p></body></html>"))
        self.bu_initdir.setText(_translate("MainWindow", "Load config"))
        self.txt_negA.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibration constant (A) to convert time-of-flight (ns) to mass-over-charge ratio (m/z)</p></body></html>"))
        self.txt_negA.setPlaceholderText(_translate("MainWindow", "A-Calib"))
        self.txt_negB.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibration constant (B) to convert time-of-flight (ns) to mass-over-charge ratio (m/z)</p></body></html>"))
        self.txt_negB.setPlaceholderText(_translate("MainWindow", "B-Calib"))
        self.txt_posC.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibration constant (C) to convert time-of-flight (ns) to mass-over-charge ratio (m/z)</p></body></html>"))
        self.txt_posC.setPlaceholderText(_translate("MainWindow", "C-Calib"))
        self.bu_decalib.setToolTip(_translate("MainWindow", "<html><head/><body><p>Convert mz spectra to TOF spectra (ns)</p></body></html>"))
        self.bu_decalib.setText(_translate("MainWindow", "Calib->Raw"))
        self.bu_calib.setToolTip(_translate("MainWindow", "<html><head/><body><p>Convert TOF spectra (ns) to mz spectra</p></body></html>"))
        self.bu_calib.setText(_translate("MainWindow", "Raw->Calib"))
        self.txt_negC.setToolTip(_translate("MainWindow", "<html><head/><body><p>Calibration constant (C) to convert time-of-flight (ns) to mass-over-charge ratio (m/z)</p></body></html>"))
        self.txt_negC.setPlaceholderText(_translate("MainWindow", "C-Calib"))
        self.tab.setTabText(self.tab.indexOf(self.tab2), _translate("MainWindow", "Acq: Calibration"))
        self.label_9.setText(_translate("MainWindow", "TOF (ns)"))
        self.txt_calibtof.setPlaceholderText(_translate("MainWindow", "100.00, 232.00, ... (separate by comma)"))
        self.bu_calibgo.setText(_translate("MainWindow", "Compute\n" "Calib"))
        self.txt_calibresult.setPlaceholderText(_translate("MainWindow", "Calculation Result"))
        self.label_14.setText(_translate("MainWindow", "M/Z"))
        self.txt_calibmz.setPlaceholderText(_translate("MainWindow", "50.33, 200.15, ... (separate by comma)"))
        self.che_calibmzreceive.setToolTip(_translate("MainWindow", "<html><head/><body><p>Left double-click on plot sends the peak to MZ</p></body></html>"))
        self.che_calibmzreceive.setText(_translate("MainWindow", "Use plot"))
        self.che_calibtofreceive.setToolTip(_translate("MainWindow", "<html><head/><body><p>Left double-click on plot sends the peak to MZ</p></body></html>"))
        self.che_calibtofreceive.setText(_translate("MainWindow", "Use plot"))
        self.tab.setTabText(self.tab.indexOf(self.tab_2), _translate("MainWindow", "Tool: Calib Calculator"))
        self.label_27.setText(_translate("MainWindow", "Difference"))
        self.label_28.setText(_translate("MainWindow", "Peak_2"))
        self.label_32.setText(_translate("MainWindow", "Peak_1"))
        self.label_30.setText(_translate("MainWindow", "Charge:"))
        self.label_29.setText(_translate("MainWindow", "Formula:"))
        self.label_31.setText(_translate("MainWindow", "MZ range:"))
        self.com_simulmode.setToolTip(_translate("MainWindow", "<html><head/><body><p>Use average mass or isotope mass (Warning: CPU intensive!)</p></body></html>"))
        self.com_simulmode.setItemText(0, _translate("MainWindow", "Average"))
        self.com_simulmode.setItemText(1, _translate("MainWindow", "Isotope"))
        self.bu_loadref.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:400;\">Load Reference TOF spectra.</span></p></body></html>"))
        self.bu_loadref.setText(_translate("MainWindow", "Load"))
        self.txt_simulformula.setToolTip(_translate("MainWindow", "<html><head/><body><p>Type chemical formula - use ()n,()m,()o,()p to simulate series of species</p></body></html>"))
        self.txt_simulformula.setPlaceholderText(_translate("MainWindow", "(CsI)nCs, HMoS4, Fe(H2O)n, ..."))
        self.bu_simul.setToolTip(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:12pt; font-weight:400;\">Simulate TOF</span></p></body></html>"))
        self.bu_simul.setText(_translate("MainWindow", "Simul"))
        self.txt_simulmz.setToolTip(_translate("MainWindow", "<html><head/><body><p>MZ range - tick \'Use Plot\' to use the X-range from the right-click dragging action</p></body></html>"))
        self.txt_simulmz.setPlaceholderText(_translate("MainWindow", "100,200"))
        self.txt_simulcharge.setToolTip(_translate("MainWindow", "<html><head/><body><p>Define charge start for every formula defined above</p></body></html>"))
        self.txt_simulcharge.setPlaceholderText(_translate("MainWindow", "0,1, 2, ..."))
        self.che_simuluseplot.setText(_translate("MainWindow", "Use plot"))
        self.txt_peak1.setText(_translate("MainWindow", "100"))
        self.txt_peak2.setText(_translate("MainWindow", "200"))
        self.txt_peakdiff.setText(_translate("MainWindow", "300"))
        self.bu_clearref.setToolTip(_translate("MainWindow", "<html><head/><body><p>Clear selected spectra only (DEL)</p></body></html>"))
        self.bu_clearref.setText(_translate("MainWindow", "Clear"))
        self.tab.setTabText(self.tab.indexOf(self.tab_5), _translate("MainWindow", "Tool: File Load and Simul"))
    
    
    
    """ THE NUTS AND BOLTS OF pyTOF """
    
    
    
    def initAll(self, cInitDir, cTestDir):
        # Create timer object
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect( self.update )
        
        # Load variables from inputted pyini path at cInitDir
        self.txt_initdir.setText(cInitDir)
        self.loadExtParam(cInitDir)
        
        # Set booleans
        self.started = False
        self.threadstatus = False
        
        # Set common variables
        self.Timeout = 500 # in msec (for disabling Start/Stop button)
        
        ##  PLOTTING BUSINESS
        # cumu = cumulative data
        # curr = calibrated data from digitizer
        # ref = reference data to be displayed together with cumulative data
        # raw = a container that interacts with worker thread data acquisition
        # Define data variable
        self.cumuX, self.cumuY = self.load_SPECTRUM( cTestDir )
        self.currX, self.currY = self.load_SPECTRUM( cTestDir )
        self.rawX, self.rawY = np.array([]), np.array([])
        # Define pen
        self.sel_pen = pg.mkPen(color='y', width=1)     # for cumu/curr curve
        self.nosel_pen = pg.mkPen(color='#FFF8', width=1, style=QtCore.Qt.DashLine)   # for ref curve
        # Connect data variable to PlotWidget objects
        self.plot_cumu = self.PW1.plot(self.cumuX, self.cumuY, pen=self.sel_pen)
        self.plot_curr = self.PW2.plot(self.currX, self.currY, pen=self.sel_pen)
        # Set label for every PlotWidget
        self.PW1.setTitle('Cumulative')
        self.PW2.setTitle('Realtime')
        # Set target data for peak finding
        self.v1.setTgt(self.cumuX, self.cumuY)
        self.v2.setTgt(self.currX, self.currY)
        # Set X-link
        self.v1.setXLink(self.v2)
        self.v2.setXLink(self.v1)
        # Set click behavior for plot_cumu
        self.plot_cumu.curve.setClickable(s = True, width = 10)
        self.plot_cumu.sigClicked.connect( self.setFocus )
        # Set click behavior for plot_ref
        # PlotWidget behavior
        self.v1.peakfind.connect(self.setMZrange)
        self.v2.peakfind.connect(self.setMZrange)
        self.v1.peaksend.connect(self.sendpeak)
        self.v2.peaksend.connect(self.sendpeak)
        self.peakcount = 0
        
        # Reference setting 
        self.plotlist = []
        self.urllist = []
        self.labellist = []
        self.setFocus(self.plot_cumu)
        # self.che_blank.setChecked(True)
        
        # Keyboard shortcut
        # QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self.MW).activated.connect(lambda: print('CTRL+S detected')) # FOR EXAMPLE
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+S'), self.MW).activated.connect(self.saveCumuSpec)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+Space'), self.MW).activated.connect(self.queueStart)
        QtWidgets.QShortcut(QtGui.QKeySequence('Ctrl+W'), self.MW).activated.connect(self.clearAll)
        QtWidgets.QShortcut(QtGui.QKeySequence('Del'), self.MW).activated.connect(self.clearSel)
        QtWidgets.QShortcut(QtGui.QKeySequence('Space'), self.MW).activated.connect(self.sendpeak)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Right), self.MW).activated.connect(self.moveupFocus)
        QtWidgets.QShortcut(QtGui.QKeySequence('D'), self.MW).activated.connect(self.moveupFocus)
        QtWidgets.QShortcut(QtGui.QKeySequence(QtCore.Qt.Key_Left), self.MW).activated.connect(self.movedownFocus)
        QtWidgets.QShortcut(QtGui.QKeySequence('A'), self.MW).activated.connect(self.movedownFocus)
        
        # Main control
        self.bu_start.released.connect(self.start)
        self.bu_clear.released.connect(self.clearAll)
        self.bu_pol.released.connect(self.polSwitch)
        
        # 1. File Save
        self.bu_save.released.connect(self.saveCumuSpec)
        self.bu_savedir.released.connect(self.setSaveDir)
        
        # 2. Digitizer
        self.txt_ave.editingFinished.connect(self.setN_Average)
        self.che_mode.stateChanged.connect(self.setCutoff)
        self.txt_time.editingFinished.connect(self.setTime)
        
        # 3. Delay Generator
        self.firstrun = True
        
        # 4. Calibration
        self.bu_saveini.released.connect(self.saveIni)
        self.bu_initdir.released.connect(self.loadInitDir)
        self.bu_calib.released.connect(self.applyCalib)
        self.bu_decalib.released.connect(self.applyDecalib)
        
        # 5. Calib Fitting
        self.bu_calibgo.released.connect(self.calibcompute)
        
        # 6. Ref load and Simulation
        self.bu_loadref.released.connect(self.loadRefSpec)
        self.bu_simul.released.connect(self.SimulStart)
        self.bu_clearref.released.connect(self.clearSel)
        self.txt_simulformula.editingFinished.connect(self.setCharge)
        
        
    """ DATA ACQUISITION MACHINERY """
    # ---- __DATA_ACQUISITION__
    
    def start(self):
        
        """ Start data acquisition thread """
        
        ## Time out the Start/Stop button
        self.bu_start.setEnabled(False)
        self.Timeout = 0
        
        # Flip started state
        self.setFocus(self.plot_cumu)
        self.started = not self.started
        
        if self.started:
            
            """ TOF STARTS ACQUIRING DATA """
            
            self.bu_start.setText('Stop')
            
            # Package data
            self.calib_p = [ float(self.txt_posA.text()),
                            float(self.txt_posB.text()),
                            float(self.txt_posC.text()) ]
            self.calib_n = [ float(self.txt_negA.text()),
                            float(self.txt_negB.text()),
                            float(self.txt_negC.text()) ]
            self.DigiParam = [ self.bu_pol.text(),
                              self.txt_ave.text(),
                              self.com_trig.currentText(),
                              self.che_mode.isChecked(), 
                              self.txt_lo.text(), 
                              self.txt_hi.text(), 
                              self.che_inv.isChecked(), 
                              self.che_abs.isChecked(),
                              self.che_baserem.isChecked(),
                              self.che_calib.isChecked(),
                              self.che_diff.isChecked(),
                              False,
                              self.firstrun]
            self.tlist = [ float(self.txt_dAdelay.text()),
                          float(self.txt_dAwidth.text()),
                          float(self.txt_dBdelay.text()),
                          float(self.txt_dBwidth.text()),
                          float(self.txt_dCdelay.text()),
                          float(self.txt_dCwidth.text()),
                          float(self.txt_dRep.text()),
                          float(self.che_dA.isChecked()),
                          float(self.che_dB.isChecked()),
                          float(self.che_dC.isChecked()) ]
            
            # Switch first run symbol
            if self.firstrun: self.firstrun = False
            
            # Empty raw data container
            self.rawX, self.rawY = np.array([]), np.array([])
            
            ## Start thread
            # Create a QThread object
            self.thread = QtCore.QThread()
            
            # Create a worker object and inject some variables to the worker class
            self.worker = Worker(
                calib_pos = self.calib_p,
                calib_neg = self.calib_n,
                tlist=self.tlist,
                digiparam = self.DigiParam)
            
            # Move worker to the thread
            self.worker.moveToThread(self.thread)
            
            # Connect signals and slots
            # Thread handling : DONT TOUCH
            self.thread.started.connect(self.worker.run)
            self.worker.finished.connect(self.threadUpdate)
            self.worker.finished.connect(self.thread.quit)
            self.worker.finished.connect(self.worker.deleteLater)
            self.thread.finished.connect(self.thread.deleteLater)
            # Event handling
            self.worker.rawComplete.connect(self.rawUpdate)
            self.worker.digiSignal.connect(self.digiUpdate)
            self.worker.delaySignal.connect(self.delayUpdate)
            self.worker.tofSignal.connect(self.tofUpdate)
            
            # Start the thread
            self.threadstatus = True
            self.thread.start()
            
        else:
            
            """ TOF STOPS ACQUIRING DATA """
            
            self.bu_start.setText('Start')
            
            ## Kill worker thread by switching its main loop boolean
            if self.threadstatus: self.worker.getdata = False
            
        
        ## Time out the Start/Stop button
        self.bu_start.setEnabled(False)
    
    def threadUpdate(self):
        self.threadstatus = False
    
    def digiUpdate(self, text):
        self.Status_digi.setText(text)
        
    def delayUpdate(self, text):
        self.Status_bme.setText(text)
    
    def tofUpdate(self, text):
        self.Status_tof.setText(text)
    
    def rawUpdate(self, data):
        
        # data[1] will have its DC component removed immediately
        self.rawX, self.rawY = data[0], data[1]-np.mean(data[1])
        
        # SavGol filter - REMOVED 28.02.2023
        # if not self.txt_smooth.text() == '':
        #     if int(self.txt_smooth.text()) > 0:
        #         self.rawY = savgol_filter(data[1], 1+(100*int(self.txt_smooth.text())), 5, mode='mirror')
                # print('data smoothing', int(self.txt_smooth.text()))
                
        # FFT Low pass filter - ADDED 28.02.2023
        if not self.txt_smooth.text() == '0':
            if float(self.txt_smooth.text()) > 0 and float(self.txt_smooth.text()) <= 1:
                b, a = scipy.signal.butter(3, float(self.txt_smooth.text()),'lowpass')
                self.rawY = scipy.signal.filtfilt(b, a, self.rawY, method="gust")
            
    def update(self):
        
        # To make the data flow consistent
        if self.started:
            
            # Change data in Current Plot
            self.currX, self.currY = self.rawX, self.rawY
            self.plot_curr.setData(self.currX, self.currY)
            self.v2.setTgt(self.currX, self.currY)
            self.v2.setYRange()
            
            # Change data in Cumulative Plot
            if len(self.cumuX) == 0 or len(self.cumuX) != len(self.currX):
                self.cumuX, self.cumuY = self.currX, self.currY
            else:
                self.cumuX, self.cumuY = self.currX, ( self.currY + self.cumuY )
            self.plot_cumu.setData(self.cumuX, self.cumuY)
            self.v1.setTgt(self.cumuX, self.cumuY)
            self.v1.setYRange()
            
            # Force redraw to complete
            QtWidgets.QApplication.processEvents()
        
        # To recover the Start/Stop button behavior
        if self.Timeout <= 250:
            self.Timeout += float( self.txt_time.text() )
        else:
            self.bu_start.setEnabled(True)
    
    def queueStart(self):
        if self.bu_start.isEnabled():
            self.start()
        else:
            print('Start button is still disable. Wait a moment.')
    
    # ---- __DATA_ANALYSIS__
    
    def SimulStart(self):
        
        """ Start simulation thread """
        
        # Disable button
        self.bu_simul.setEnabled(False)
        
        # Package data
        self.simulparam = [ self.txt_simulformula.text(),
                           self.txt_simulmz.text(),
                           self.txt_simulcharge.text(),
                           self.com_simulmode.currentText(),
                           self.bu_pol.text() ]
        
        # Start new thread for MassSpec Simulation
        # Create a QThread object
        self.sthread = QtCore.QThread()
        
        # Create a worker object and inject some variables to the worker class
        self.sworker = SimulWorker(simulparam = self.simulparam)
        
        # Move worker to the thread
        self.sworker.moveToThread(self.sthread)
        
        # Connect signals and slots
        # Thread handling : DONT TOUCH
        self.sthread.started.connect(self.sworker.SimulRun)
        self.sworker.finished.connect(self.sthread.quit)
        self.sworker.finished.connect(self.sworker.deleteLater)
        self.sthread.finished.connect(self.sthread.deleteLater)
        # Event handling
        self.sworker.completed.connect(self.SimulPlot)
        self.sworker.askmass.connect(self.SimulAskMass)
        self.sworker.asksimple.connect(self.SimulAskSimple)
        
        # Start the thread
        self.sthread.start()
    
    def SimulPlot(self, x):
        
        """ Receives numpy array from Simul thread to prep and plot them """
        
        # Plot padding
        h = self.v1.state['targetRange'][1][1]
        padX = np.arange( np.min(x[0])-100, np.max(x[0])+100, 0.5 )
        padY = np.zeros( padX.shape ) + np.random.normal(0,0.003*h,len(padX))
        
        # Add data padding to peak data
        padX = np.append( padX, x[0] )
        padY = np.append( padY, x[1]*0.2*h )
        
        # Clean up label info
        info = np.array(x[2])
        # print(info)
        
        # Create new instance of plot
        padX, padY = zip( *sorted( zip( padX, padY ) ) )
        self.urllist.append( str(info[:,0]) )
        self.plotlist.append( self.PW1.plot(padX, padY, pen=self.nosel_pen) )
        
        # Set signal behavior
        self.plotlist[-1].curve.setClickable(s = True, width = 10)
        self.plotlist[-1].sigClicked.connect( self.setFocus )
        
        # Set Label on simul plot
        tmp = []
        for i in info:
            text = pg.TextItem(html='<div><span style="color: #0FF; font-size: 9pt;">%.1f</span>'%(float(i[1]))+'<br><span style="color: #FFF; font-size: 9pt;"><b>'+str(i[0])+'</b></span></div>' , anchor=(0.5,0))
            ###
            # Magic spell from
            # https://stackoverflow.com/questions/62601498/pyqtgraph-textitem-center-align-text
            it = text.textItem
            option = it.document().defaultTextOption()
            option.setAlignment(QtCore.Qt.AlignCenter)
            it.document().setDefaultTextOption(option)
            it.setTextWidth(it.boundingRect().width())
            ###
            text.setPos(float(i[1]),0)
            tmp.append(text)
            self.PW1.addItem(text)
            
        self.labellist.append( tmp )
        
        self.bu_simul.setEnabled(True)
    
    def SimulAskMass(self, x):
        
        """ Outputting mass and abundance list from chemparse result """
        
        Elem = np.array(list( x[0].keys() ))
        Count = np.array(list( x[0].values() ))
        masslist, abunlist = [], []
        
        if x[1][0] == 'A':       # Setup mass list for average mode
            masslist = np.array(list( map( lambda i: md.element(i).atomic_weight, Elem ) )) * Count
            abunlist = [1] * len(Elem)
        
        elif x[1][0] == 'I':     # Setup mass list for isotope mode
            Elem = np.repeat(Elem, Count)
            for i in Elem:
                tmpmass, tmpabun = [], []
                for j in md.element(i).isotopes:
                    if isinstance(j.abundance, float):
                        if j.abundance > 0.05:
                            tmpmass.append(j.mass)
                            tmpabun.append(j.abundance)
                masslist.append(tmpmass)
                abunlist.append(tmpabun)
            
        self.sworker.masslist = masslist
        self.sworker.abunlist = abunlist
        self.sworker.wait = False
    
    def SimulAskSimple(self, x):
        
        """ Modifying chemparse dictionary into mass list """
        
        for i in list( x.keys() ): x[i] = md.element(i).atomic_weight
        self.sworker.masslist = x
        self.sworker.wait = False
    
    # ---- __0.__MAIN_PANEL__
    
    def setFocus(self, p):
        
        """ Select on spectra clicked by the mouse """
        
        if not self.started:
            # Set all plot into not-selected color
            self.plot_cumu.setPen(self.nosel_pen)
            for i in self.plotlist:
                i.setPen(self.nosel_pen)
            
            # Set specific plot into selected color
            try:
                self.plotsele = self.plotlist.index(p)
                self.v1.setTgt( p.getData()[0], p.getData()[1] )
                p.setPen(self.sel_pen)
                self.txt_statref.setText('Spectra selected: '+self.urllist[self.plotsele])
            except:
                self.plotsele = -1
                self.v1.setTgt( self.plot_cumu.getData()[0], self.plot_cumu.getData()[1] )
                self.plot_cumu.setPen(self.sel_pen)
                self.txt_statref.setText('Spectra selected: '+'CumuTOF')
    
    def moveupFocus(self):
        # print(self.plotsele,'up')
        self.plotsele += 1
        # Reset plotsele index
        if self.plotsele == len(self.plotlist): self.plotsele = -1
        # Set Focus
        if self.plotsele>=0:
            self.setFocus(self.plotlist[self.plotsele])
        else:
            self.setFocus(self.plot_cumu)
    
    def movedownFocus(self):
        # print(self.plotsele,'down')
        self.plotsele += -1
        # Reset plotsele index
        if self.plotsele == -2: self.plotsele = len(self.plotlist)-1
        # Set Focus
        if self.plotsele>=0:
            self.setFocus(self.plotlist[self.plotsele])
        else:
            self.setFocus(self.plot_cumu)
    
    def sendpeak(self, x):
        
        """ Store peak data on text field """
        
        if self.che_calibtofreceive.isChecked():
            self.txt_calibtof.setText( self.txt_calibtof.text()+', '+str(x) )
            if self.txt_calibtof.text()[0]==',': self.txt_calibtof.setText(self.txt_calibtof.text()[1:])
        
        if self.che_calibmzreceive.isChecked():
            self.txt_calibmz.setText( self.txt_calibmz.text()+', '+str(x) )
            if self.txt_calibmz.text()[0]==',': self.txt_calibmz.setText(self.txt_calibmz.text()[1:])
    
    def polSwitch(self):
        
        """ Switch Positive label to Negative label and vice versa """
        
        if self.bu_pol.text()[0] == 'N':
            self.bu_pol.setText('Pos')
        else:
            self.bu_pol.setText('Neg')
    
    def clearAll(self):
        
        """ Clear all spectra """
        
        #Clear current spectrum and its container
        self.currX, self.currY = np.array([]), np.array([])
        self.rawX, self.rawY = np.array([]), np.array([])
        self.plot_curr.setData(self.currX, self.currY)
        #Clear cumulative spectrum
        self.cumuX, self.cumuY = self.currX, self.currY
        self.plot_cumu.setData(self.cumuX, self.cumuY)
        
        #Clear ref spectrum
        while len(self.plotlist) > 0:
            # Remove text description of selected spectra
            self.urllist.remove( self.urllist[ 0 ] )
            # Remove label of selected spectra
            while len(self.labellist[0]) > 0:
                self.PW1.removeItem( self.labellist[0][0] )
                self.labellist[0].remove(self.labellist[0][0])
            self.labellist.remove( self.labellist[ 0 ] )
            # Remove plot of selected spectra
            self.PW1.removeItem( self.plotlist[ 0 ] )
            self.plotlist.remove( self.plotlist[ 0 ] )
        self.setFocus( self.plot_cumu )
    
    
    # ---- __1.__FILE_SAVE__
    
    def setSaveDir(self):
        
        """ Call dialog to set save directory """
        
        SaveDir =  QtWidgets.QFileDialog.getExistingDirectory()
        if SaveDir:
            self.txt_savedir.setText( SaveDir )
    
    def saveCumuSpec(self):
        
        """ Save cumulative spectra """
        
        # Create save directory if it doesnt exist
        pdir = self.txt_savedir.text() + '/' + str(datetime.date.today()) + '_' + self.txt_mol.text() + '_' + self.txt_surf.text() + '_' + self.bu_pol.text()[:3]
        if not os.path.isdir(pdir):
            print(pdir,'doesnt exists. Creating folder...')
            os.mkdir(pdir)
            self.txt_saveindex.setText('0')
        
        # Check duplicate file name
        # i = int(self.txt_saveindex.text())
        # pname = str(datetime.date.today()) + '_' + self.txt_mol.text() + '_' + self.txt_surf.text() + '_' + self.bu_pol.text()[:3] + '_' + self.txt_comm.text() + '_' + str(i) + '.pytof'
        # # Check if any file exist - v1 : full file name
        # while os.path.isfile(pdir + '/' + pname):
        #     print(pname,'exists')
        #     i += 1
        #     pname = str(datetime.date.today()) + '_' + self.txt_mol.text() + '_' + self.txt_surf.text() + '_' + self.bu_pol.text()[:3] + '_' + self.txt_comm.text() + '_' + str(i) + '.pytof'
        #     print('checking',pname)
        
        # Check duplicate file name v2
        i = 0
        # print([ j[j.rfind('.'):] for j in os.listdir(pdir) ])
        # print((str(i)+'_') in [ j[:j.find('_')+1] for j in os.listdir(pdir) ])
        # print('.pytof' in [ j[j.rfind('_'):] for j in os.listdir(pdir) ])
        while (str(i)+'_') in [ j[:j.find('_')+1] for j in os.listdir(pdir) ] and '.pytof' in [ j[j.rfind('.'):] for j in os.listdir(pdir) ] :
            # print([ j[j.rfind('_'):] for j in os.listdir(pdir) ])
            # print((str(i)+'_') in [ j[:j.find('_')+1] for j in os.listdir(pdir) ])
            # print('.pytof' in [ j[j.rfind('_'):] for j in os.listdir(pdir) ])
            i += 1
        #pname = str(datetime.date.today()) + '_' + self.txt_mol.text() + '_' + self.txt_surf.text() + '_' + self.bu_pol.text()[:3] + '_' + self.txt_comm.text() + '_' + str(i) + '.pytof'
        
        Q1 = "" if self.txt_Q1.text() == "" else 'Q1-' + self.txt_Q1.text() + '_'
        Q2 = "" if self.txt_Q2.text() == "" else 'Q2-' + self.txt_Q2.text() + '_'
        UV = "" if self.txt_UV.text() == "" else 'UV' + self.txt_UV.text() + '_'
        
        pname = str(i) + '_' + Q1 + Q2 + UV + self.txt_comm.text() + '.pytof'
        
        # Save the file
        self.write_SPECTRUM(pdir + '/' + pname, self.cumuX, self.cumuY)
        self.txt_saveindex.setText(str(i+1))
    
    def write_SPECTRUM(self, path, xdata, ydata):
        
        """ Writing data to file """
        
        now = datetime.datetime.now()
        f = open(path,'w')
        f.write('## pyTOF data saved on '+str(now.strftime("%Y-%m-%d %H:%M:%S"))+'\n')
        f.write('## '+self.txt_mol.text()+'_'+self.txt_surf.text()+'_'+self.txt_comm.text()+'\n')
        f.write('## '+self.bu_pol.text()+' MODE\n')
        if self.bu_pol.text()[0] == 'P':
            f.write('## Calib POS:\n')
            f.write('## POS_A:'+self.txt_posA.text()+'\n')
            f.write('## POS_B:'+self.txt_posB.text()+'\n')
            f.write('## POS_C:'+self.txt_posC.text()+'\n')
        else:
            f.write('## Calib NEG:\n')
            f.write('## NEG_A:'+self.txt_negA.text()+'\n')
            f.write('## NEG_B:'+self.txt_negB.text()+'\n')
            f.write('## NEG_C:'+self.txt_negC.text()+'\n')
        for i in range(len(xdata)):
            f.write(str(xdata[i])+'  '+str(ydata[i])+'\n')
        f.close()
    
    
    # ---- __2.__DIGITIZER__
    
    def setN_Average(self):
        
        """ Automatically recompute recommended update time when N_Average changes"""
        
        self.txt_time.setText( str( float(self.txt_ave.text()) *2*0.120790204355556 ) )
        self.setTime()
    
    def setTime(self):
        
        """ Setting update time """
        
        if int( float(self.txt_time.text()) ) <= 10: self.txt_time.setText('10')
        self.timer.start( int( float(self.txt_time.text()) ) ) #msec
    
    def setCutoff(self):
        
        """ Automatically set Delay Generator Timings when acq mode is changed """
        
        if self.che_mode.isChecked():
            # Hi mass config (0.625 GHz sampling)
            # MZ-range: ~25 to ~41.1k
            self.txt_hi.setText('41000')
            self.txt_dAdelay.setText('0')
            self.txt_dAwidth.setText('108')
            self.txt_dBdelay.setText('0')
            self.txt_dBwidth.setText('108')
            self.txt_dCdelay.setText('0')
            self.txt_dCwidth.setText('108')
            self.txt_dRep.setText('163')
        else:
            # Lo mass config (1.250 GHz sampling)
            # MZ-range: ~25 to ~10.8k
            self.txt_hi.setText('4000')
            self.txt_dAdelay.setText('0')
            self.txt_dAwidth.setText('56')
            self.txt_dBdelay.setText('0')
            self.txt_dBwidth.setText('56')
            self.txt_dCdelay.setText('0')
            self.txt_dCwidth.setText('56')
            self.txt_dRep.setText('111')
    
    
    # ---- __3.__DELAYGEN__
    
    pass
    
    
    # ---- __4.__CALIB__
    
    def applyDecalib(self):
        
        """ Decalibrate selected spectra """
        
        if not self.started:
            # We need to de-calibrate cumuX and currX
            if self.bu_pol.text()[0] == 'P':
                a = float( self.txt_posA.text() )
                b = float( self.txt_posB.text() )
                c = float( self.txt_posC.text() )
            else:
                a = float( self.txt_negA.text() )
                b = float( self.txt_negB.text() )
                c = float( self.txt_negC.text() )
            if self.plotsele >= 0:
                p = self.plotlist[ self.plotsele ]
            else:
                p = self.plot_cumu
            X, Y = p.getData()
            X = ( np.sqrt( X-c+(b*b/(4*a))  ) - (b/(2*math.sqrt(a))) ) / math.sqrt(a)
            p.setData( X, Y )
            self.setFocus(p)
            
    def applyCalib(self):
        
        """ Calibrate selected spectra """
        
        if not self.started:
            # We need to calibrate cumuX and currX
            if self.bu_pol.text()[0] == 'P':
                a = float( self.txt_posA.text() )
                b = float( self.txt_posB.text() )
                c = float( self.txt_posC.text() )
            else:
                a = float( self.txt_negA.text() )
                b = float( self.txt_negB.text() )
                c = float( self.txt_negC.text() )
            if self.plotsele >= 0:
                p = self.plotlist[ self.plotsele ]
            else:
                p = self.plot_cumu
            X, Y = p.getData()
            X = (a*X*X) + (b*X) + (c)
            p.setData( X, Y )
            self.setFocus(p)
    
    def saveIni(self):
        
        """ Save existing parameter """
        
        path = QtWidgets.QFileDialog.getSaveFileName()[0]
        if path:
            now = datetime.datetime.now()
            f = open(path,'w')
            f.write('## pyTOF.ini file saved by software on '+str(now.strftime("%Y-%m-%d %H:%M:%S"))+'\n')
            f.write('\n')
            f.write('SaveDir = '+self.txt_savedir.text()+'\n')
            f.write('N_Average = '+self.txt_ave.text()+'\n')
            f.write('POS_A = '+self.txt_posA.text()+'\n')
            f.write('POS_B = '+self.txt_posB.text()+'\n')
            f.write('POS_C = '+self.txt_posC.text()+'\n')
            f.write('NEG_A = '+self.txt_negA.text()+'\n')
            f.write('NEG_B = '+self.txt_negB.text()+'\n')
            f.write('NEG_C = '+self.txt_negC.text()+'\n')
            # f.write('A_DELAY = '+self.txt_dAdelay.text()+'\n')
            # f.write('A_WIDTH = '+self.txt_dAwidth.text()+'\n')
            # f.write('B_DELAY = '+self.txt_dBdelay.text()+'\n')
            # f.write('B_WIDTH = '+self.txt_dBwidth.text()+'\n')
            # f.write('C_DELAY = '+self.txt_dCdelay.text()+'\n')
            # f.write('C_WIDTH = '+self.txt_dCwidth.text()+'\n')
            # f.write('T_REPEAT = '+self.txt_dRep.text()+'\n')
            f.write('\n')
            f.write('## End Automated pyTOF.ini writer')
            f.close()
    
    def loadInitDir(self):
        
        """ Call dialog to load external ini file """
        
        InitDir = QtWidgets.QFileDialog.getOpenFileName()[0]
        if InitDir:
            self.txt_initdir.setText( InitDir )
            self.loadExtParam( InitDir )
    
    def loadExtParam(self, path = './pyTOF.ini'):
        
        """ Load external ini file """
        
        with open(path,'r') as f:
            for line in f:
                if '#' not in line:   # Ignore any line that has '#' on it
                    tm = parse(line)
                    if 'SaveDir' in tm: self.txt_savedir.setText( tm[2] )
                    elif 'N_Average' in tm: self.txt_ave.setText( tm[2] )
                    elif 'POS_A' in tm: self.txt_posA.setText( tm[2] )
                    elif 'POS_B' in tm: self.txt_posB.setText( tm[2] )
                    elif 'POS_C' in tm: self.txt_posC.setText( tm[2] )
                    elif 'NEG_A' in tm: self.txt_negA.setText( tm[2] )
                    elif 'NEG_B' in tm: self.txt_negB.setText( tm[2] )
                    elif 'NEG_C' in tm: self.txt_negC.setText( tm[2] )
                    # elif 'A_DELAY' in tm: self.txt_dAdelay.setText( tm[2] )
                    # elif 'A_WIDTH' in tm: self.txt_dAwidth.setText( tm[2] )
                    # elif 'B_DELAY' in tm: self.txt_dBdelay.setText( tm[2] )
                    # elif 'B_WIDTH' in tm: self.txt_dBwidth.setText( tm[2] )
                    # elif 'C_DELAY' in tm: self.txt_dCdelay.setText( tm[2] )
                    # elif 'C_WIDTH' in tm: self.txt_dCwidth.setText( tm[2] )
                    # elif 'T_REPEAT' in tm: self.txt_dRep.setText( tm[2] )
        self.setN_Average()
        self.setCutoff()
    
    
    # ---- __5.__CALIB_TOOL__
    
    def calibcompute(self):
        
        """ Compute A, B, C calibration values from entered points """
        
        try:
            time = np.array(parse( self.txt_calibtof.text(),',',ty='flo' ))
            mass = np.array(parse( self.txt_calibmz.text(),',',ty='flo' ))
            print(time, mass)
            fit = np.poly1d(np.polyfit(time,mass,2))
            if self.bu_pol.text()[0] == 'P':
                key = 'POS_'
            else:
                key = 'NEG_'
            self.txt_calibresult.setText(key+'A = '+str(fit[2])+'\n'+key+'B = '+str(fit[1])+'\n'+key+'C = '+str(fit[0]))
        except:
            self.txt_calibresult.setText('Error in fitting - Check your data?')
    
    
    # ---- __6.__FILE_LOAD_&_SIMUL__
    
    def loadRefSpec(self):
        
        """ Load reference spectra """
        
        i = QtWidgets.QFileDialog.getOpenFileNames()[0]
        
        for RefDir in i:
            if RefDir:
                # Load data
                datX, datY = self.load_SPECTRUM( RefDir )
                datY = datY/max(datY)
                
                # Create new instance of plot
                self.urllist.append( RefDir[RefDir.rfind('/'):] )
                self.plotlist.append( self.PW1.plot(datX, datY, pen=self.nosel_pen) )
                self.labellist.append( [] )
                # Set signal behavior
                self.plotlist[-1].curve.setClickable(s = True, width = 10)
                self.plotlist[-1].sigClicked.connect( self.setFocus )
    
    def load_SPECTRUM(self, path):
        
        """ Parse and read external pytof mass spec file """
        
        dat = []
        with open(path,'r') as f:
            for line in f:
                if '#' not in line:
                    dat.append(parse(line,ty='flo'))
        dat = np.array(dat)
        return dat[:,0] , dat[:,1]
    
    def clearSel(self):
        
        """ Remove selected spectra """ 
        
        if self.plotsele == -1:
            # Clear current spectrum and its container
            self.currX, self.currY = np.array([]), np.array([])
            self.rawX, self.rawY = np.array([]), np.array([])
            self.plot_curr.setData(self.currX, self.currY)
            # Clear cumulative spectrum
            self.cumuX, self.cumuY = self.currX, self.currY
            self.plot_cumu.setData(self.cumuX, self.cumuY)
            
        else:
            # Clear selected spectra
            # Remove text description of selected spectra
            self.urllist.remove( self.urllist[self.plotsele] )
            # Remove label in selected spectra
            while len(self.labellist[self.plotsele]) > 0:
                self.PW1.removeItem( self.labellist[self.plotsele][0] )
                self.labellist[self.plotsele].remove(self.labellist[self.plotsele][0])
            self.labellist.remove( self.labellist[self.plotsele] )
            # Remove plot of selected spectra
            self.PW1.removeItem( self.plotlist[self.plotsele] )
            self.plotlist.remove( self.plotlist[self.plotsele] )
            if len(self.plotlist) > 0:
                self.setFocus( self.plotlist[-1] )
            else:
                self.setFocus( self.plot_cumu )
    
    def setMZrange(self, x):
        
        """ Automatically set MZ range when peak-detect is successful in the plot """
        
        if len(x)==3 and self.che_simuluseplot.isChecked():
            
            #Send peak to MZ range
            if self.plotsele == -1:
                self.txt_simulmz.setText(str(round(self.cumuX[x[0]],2))+', '+str(round(self.cumuX[x[1]],2)))
            else:
                p = self.plotlist[self.plotsele].getData()[0]
                self.txt_simulmz.setText(str(round(p[x[0]],2))+', '+str(round(p[x[1]],2)))
        
        if len(x)==3:
            
            # Send peak to Differentiator
            if self.peakcount == 0:
                self.txt_peak1.setText('%.3f' %x[2])
            if self.peakcount == 1:
                self.txt_peak2.setText('%.3f' %x[2])
            
            self.peakcount +=1
            
            if self.peakcount>1:
                self.txt_peakdiff.setText('%.3f' % (float(self.txt_peak1.text())-float(self.txt_peak2.text())) )
                self.peakcount = 0
            
    
    def setCharge(self):
        
        """ Automatically set charge after formula is entered """
        
        s = parse(self.txt_simulformula.text(), ',')
        if self.bu_pol.text()[0] == 'N':
            self.txt_simulcharge.setText( ('-1, '*len(s))[0:-2] )
        else:
            self.txt_simulcharge.setText( ('1, '*len(s))[0:-2] )


"""
# ---- QtObjects Creation

"""
## Always start by initializing Qt (only once per application)
app = QtWidgets.QApplication(sys.argv)

## Define a top-level widget to hold everything
MainWindow = QtWidgets.QMainWindow()
ui = Ui_MainWindow()
ui.setupUi(MainWindow)

## Load defaults
ui.initAll('./pytof.ini', './TestData.pytof')

## Display the widget as a new window
MainWindow.show()
ui.v1.autoRange()


"""
# ---- Start GUI

"""
## Start Qt event loop unless running in interactive mode or using pyside.
def appExec():
    app.exec_()
    try:
        ui.worker.getdata = False
        print('Thread exit during application exit')
    except:
        print('No worker thread detected')
    try:
        if ui.worker.digiStatus:
            print('Digitizer exiting... after QtApp terminated!')
            ui.worker.close_SPECTRUM()
    except:
        print('No worker thread detected')
    try:
        if ui.worker.delayStatus:
            print('DelayGen exiting... after QtApp terminated!')
            ui.worker.close_BERGMANN()
    except:
        print('No worker thread detected')
    print('QtApp exiting...')

if __name__ == '__main__':
    if (sys.flags.interactive != 1) or not hasattr(QtCore, 'PYQT_VERSION'):
        sys.exit( appExec() )