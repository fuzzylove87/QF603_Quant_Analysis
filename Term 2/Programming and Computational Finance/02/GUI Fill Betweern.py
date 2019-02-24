# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 22:22:44 2018

@author: ykfri
"""

#more imports
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg \
     import FigureCanvasQTAgg as FigureCanvas

#UI file
qtCreatorFile = "AppFillBetween.ui"

#more initialization
self.data=pd.read_csv('CC3.SI.csv',index_col=0,parse_dates=True)
self.data.drop(self.data.index[self.data['Volume']==0],inplace=True)
self.ma=self.data['Close'].rolling(15).mean()
self.mstd=self.data['Close'].rolling(15).std()

self.checkBox_ma.setChecked(True)
self.checkBox_mstd.setChecked(True)

self.fig1 = Figure()
self.ax1 = self.fig1.add_subplot(111)
self.ax1.plot(self.data.index, self.data['Close'], 'k')
self.plot2=self.ax1.plot(self.ma.index, self.ma, 'b')
self.plot3=self.ax1.fill_between(self.mstd.index, self.ma-2*self.mstd, 
                          self.ma+2*self.mstd, color='b', alpha=0.2)
self.canvas1 = FigureCanvas(self.fig1)        
self.verticalLayout.addWidget(self.canvas1)
self.canvas1.draw()
        
self.checkBox_ma.stateChanged.connect(self.updategraph)
self.checkBox_mstd.stateChanged.connect(self.updategraph)

# more functions/methods
def updategraph(self):

    if self.checkBox_ma.isChecked():
        plt.setp(self.plot2, visible=True)
    else:
        plt.setp(self.plot2, visible=False)

    if self.checkBox_mstd.isChecked():
        plt.setp(self.plot3, visible=True)
    else:
        plt.setp(self.plot3, visible=False)
    self.canvas1.draw()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main = Main()
    main.show()
    sys.exit(app.exec_())
    