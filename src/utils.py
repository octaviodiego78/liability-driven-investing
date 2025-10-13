"""
Utility functions for the LDI project.
Contains all helper functions for calculations and data processing.
"""

import math
import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from datetime import date
import plotly.graph_objects as go
import plotly.io as pio
from pathlib import Path

from . import config
from .config import (VARModel, Mapping, Mortality, Census, histMF, histAR, cholMF, 
                    cholNormal, cholRecession, termmix, migration, RecFun, scale, lumpSumProb, lumpSumDR, valuationDate)
from .config import (fMI, mMI, salaryGr, salaryAvgPeriod, benefitRate, COLA, maxCOLA, salaryMultiple, gender, salary, retireDate, weight, occupation)



# Visualization Functions
def plot_model_comparison(results_dict, column_name, output_dir):
    """Create comparison plot for different models."""
    fig = go.Figure()
    
    for model_name, df in results_dict.items():
        fig.add_trace(go.Scatter(
            x=df['period'].unique().tolist(),
            y=df.groupby('period')[column_name].mean(),
            mode='lines',
            name=model_name
        ))
    
    fig.update_layout(
        title=f'Comparison of {column_name} Over Time',
        xaxis_title='Period',
        yaxis_title=column_name,
        hovermode='x unified'
    )
    
    output_path = output_dir / f'{column_name}_comparison.html'
    pio.write_html(fig, str(output_path))
    print(f"Plot saved as '{output_path}'")

def save_results(results, model_name, output_dir):
    """Save model results to CSV."""
    df = pd.DataFrame(results)
    output_path = output_dir / f'{model_name}_results.csv'
    df.to_csv(output_path, index=False)
    print(f"Results saved to '{output_path}'")
    return df

def run_experiment(model_class, data, num_episodes, num_sims):
    """Run a single experiment with the specified model."""
    model = model_class()
    print(f"Training {model.__class__.__name__}...")
    model.train(num_episodes, num_sims)
    print("Evaluating model...")
    results = model.evaluate(config.n_sim)
    return results

# Core Financial Functions
def recession(fun, vals):
    """Calculate recession probability based on economic indicators."""
    vals.insert(0, 1)
    prob = 1/(1 + math.exp(-sum([x*y for x, y in zip(fun, vals)])))
    return 1 if prob > 0.5 else 0

def curvefitting(value, term=np.array([0.25,1,2,3,5,7,10,20,30]), inter="linear", extro="cfr", target=0.04):
    """Fit yield curve using interpolation and extrapolation methods."""
    if value.shape[0] != term.shape[0]:
        return -1
    else:
        startT = term[0]
        endT = term[term.shape[0]-1]
        secondEndT = term[term.shape[0]-2]
        xnew = np.linspace(startT, endT, num=int(round((endT-startT)*4+1)), endpoint=True)
        if inter=="linear":
            f = interp1d(term, value)
        else:
            f = interp1d(term, value, kind='cubic')
        ynew = f(xnew)
        if endT < 100:
            xnew2 = np.linspace(endT+0.25, 100, num=int(round((100-endT)*4)), endpoint=True)
            ynew2 = []
            if extro=="cfr":
                cfr = (((1+value[value.shape[0]-1]) ** endT)/((1+value[value.shape[0]-2]) ** secondEndT)) ** (1/(endT-secondEndT))-1
                cumfac = (1+value[value.shape[0]-1]) ** endT
                for ix in xnew2:
                    cumfac = cumfac*(1+cfr)**(1/4)
                    ynew2.append(cumfac**(1/ix)-1)
            else:
                p = np.poly1d(np.polyfit(np.array([secondEndT,endT,100]), np.array([value[value.shape[0]-2],value[value.shape[0]-1],target]), deg=2))
                ynew2 = p(xnew2)
        x = np.append(xnew, xnew2)
        y = np.append(ynew, ynew2)
        return x, y

def bondPrice(terms, rates, term, freq, coupon, redemption, inter="linear", extro="tar", expectedRate=0.04):
    """Calculate bond price based on yield curve and bond characteristics."""
    xi, yi = curvefitting(rates, terms, inter=inter, extro=extro, target=expectedRate)
    f = interp1d(xi, yi)
    if term > 0.25:
        bondvalue = redemption*(1+coupon*freq)/(1+f(term))**term
    else:
        bondvalue = redemption
    for i in np.arange(term-freq, 0.01, -freq):
        bondvalue = bondvalue + redemption*coupon*freq/(1+f(i))**(i)
    return bondvalue

def sampleMF(var,fundmap,histMF,histAR,cholMF,cholNormal,cholRecession,dyn,n,stochastic=True):
	var = np.array(var)
	fundmap = np.array(fundmap)
	histMF = np.array(histMF)
	histAR = np.array(histAR)
	cholMF = np.array(cholMF)
	cholNormal = np.array(cholNormal)
	cholRecession = np.array(cholRecession)
	if stochastic==True and dyn>=n:
		sd = var[:,[8]]
		var = var[:,[range(0,9)]]
		hmf = histMF
		hmf3 = np.insert(hmf,0,hmf[hmf.shape[0]-2,:],0)
		recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
		rece = recession(RecFun,recessionVal)
		hmf = hmf[hmf.shape[0]-1,:]
		thmf = np.append(hmf,rece)
		sdtnormal = np.sqrt(fundmap[:,[27]])/scale
		sdtrecession = np.sqrt(fundmap[:,[28]])/scale
		sdinormal = np.sqrt(fundmap[:,[29]])/scale
		sdirecession = fundmap[:,[30]]/scale
		corrnormal = fundmap[:,[31]]
		corrrecession = fundmap[:,[32]]
		fundmap = fundmap[:,[range(0,27)]]
		har = histAR
		har3 = np.insert(har,0,har[har.shape[0]-2,:],0)
		har = har[har.shape[0]-1,:]
		thmf = np.append(thmf,har)
		hmf=np.append(hmf,1)
		for i in range(0,n):
			hmf = var.dot(hmf.T).T
			hmf = hmf + np.multiply(sd.T,np.dot(cholMF,np.array(np.random.normal(0,1,8)).T))
			hmf3[0] = hmf3[1]
			hmf3[1] = hmf3[2]
			hmf3[2] = hmf
			recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
			rece = recession(RecFun,recessionVal)
			for j in range(0,har.shape[0]):
				hartmp = hmf3.flatten('F')
				hartmp = np.insert(hartmp,0,[1,har3[1,j],har3[2,j]])
				har[j] = fundmap[j].dot(hartmp.T).T
			if rece==0:
				rnds = np.multiply(sdinormal.T, np.dot(cholNormal,np.array(np.random.normal(0,1,33)).T))
				rnds = np.multiply((np.multiply(corrnormal.T,har) + np.multiply(np.sqrt(1-np.multiply(corrnormal.T,corrnormal.T)),rnds)),sdinormal.T)
				rnds = np.divide(rnds,np.sqrt(np.multiply(np.multiply(corrnormal.T,corrnormal.T),np.multiply(sdtnormal.T,sdtnormal.T))+np.multiply((1-np.multiply(corrnormal.T,corrnormal.T)),np.multiply(sdinormal.T,sdinormal.T))))
				rnds[0,5]=0 #set random numbers to be zero for those already generated in economic factors.
				rnds[0,8]=0
				rnds[0,10]=0
				rnds[0,30]=0
				rnds[0,31]=0
				har = har + rnds.flatten('F')
			else:
				rnds = np.multiply(sdirecession.T, np.dot(cholRecession,np.array(np.random.normal(0,1,33)).T))
				rnds = np.multiply((np.multiply(corrrecession.T,har) + np.multiply(np.sqrt(1-np.multiply(corrrecession.T,corrrecession.T)),rnds)),sdirecession.T)
				rnds = np.divide(rnds,np.sqrt(np.multiply(np.multiply(corrrecession.T,corrrecession.T),np.multiply(sdtrecession.T,sdtrecession.T))+np.multiply((1-np.multiply(corrrecession.T,corrrecession.T)),np.multiply(sdirecession.T,sdirecession.T))))
				rnds[0,5]=0
				rnds[0,8]=0
				rnds[0,10]=0
				rnds[0,30]=0
				rnds[0,31]=0
				har = har + rnds.flatten('F')
			for ix in range(0,8):
				har[ix]=max(0.1,har[ix])
			for ix in [10,12,14,16]:
				har[ix]=max(0,har[ix])
			for ix in range(0,har.shape[0]):
				har[ix]=max(-99,har[ix])
			har3[0] = har3[1]
			har3[1] = har3[2]
			har3[2] = har
			thmf = np.vstack([thmf, np.append(np.append(hmf,rece),har.flatten())])
			hmf=np.append(hmf,1)
		return thmf
	elif stochastic==True and dyn>0 and dyn<n:
		sd = var[:,[8]]
		var = var[:,[range(0,9)]]
		hmf = histMF
		hmf3 = np.insert(hmf,0,hmf[hmf.shape[0]-2,:],0)
		recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
		rece = recession(RecFun,recessionVal)
		hmf = hmf[hmf.shape[0]-1,:]
		thmf = np.append(hmf,rece)
		sdtnormal = fundmap[:,[27]]/scale
		sdtrecession = fundmap[:,[28]]/scale
		sdinormal = fundmap[:,[29]]/scale
		sdirecession = fundmap[:,[30]]/scale
		corrnormal = fundmap[:,[31]]
		corrrecession = fundmap[:,[32]]
		fundmap = fundmap[:,[range(0,27)]]
		har = histAR
		har3 = np.insert(har,0,har[har.shape[0]-2,:],0)
		har = har[har.shape[0]-1,:]
		thmf = np.append(thmf,har)
		hmf=np.append(hmf,1)
		for i in range(0,dyn):
			hmf = var.dot(hmf.T).T
			hmf = hmf + np.multiply(sd.T,np.dot(cholMF,np.array(np.random.normal(0,1,8)).T))
			hmf3[0] = hmf3[1]
			hmf3[1] = hmf3[2]
			hmf3[2] = hmf
			recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
			rece = recession(RecFun,recessionVal)
			for j in range(0,har.shape[0]):
				hartmp = hmf3.flatten('F')
				hartmp = np.insert(hartmp,0,[1,har3[1,j],har3[2,j]])
				har[j] = fundmap[j].dot(hartmp.T).T
			if rece==0:
				rnds = np.multiply(sdinormal.T, np.dot(cholNormal,np.array(np.random.normal(0,1,33)).T))
				rnds = np.multiply((np.multiply(corrnormal.T,har) + np.multiply(np.sqrt(1-np.multiply(corrnormal.T,corrnormal.T)),rnds)),sdinormal.T)
				rnds = np.divide(rnds,np.sqrt(np.multiply(np.multiply(corrnormal.T,corrnormal.T),np.multiply(sdtnormal.T,sdtnormal.T))+np.multiply((1-np.multiply(corrnormal.T,corrnormal.T)),np.multiply(sdinormal.T,sdinormal.T))))
				rnds[0,5]=0
				rnds[0,8]=0
				rnds[0,10]=0
				rnds[0,30]=0
				rnds[0,31]=0
				har = har + rnds.flatten('F')
			else:
				rnds = np.multiply(sdirecession.T, np.dot(cholRecession,np.array(np.random.normal(0,1,33)).T))
				rnds = np.multiply((np.multiply(corrrecession.T,har) + np.multiply(np.sqrt(1-np.multiply(corrrecession.T,corrrecession.T)),rnds)),sdirecession.T)
				rnds = np.divide(rnds,np.sqrt(np.multiply(np.multiply(corrrecession.T,corrrecession.T),np.multiply(sdtrecession.T,sdtrecession.T))+np.multiply((1-np.multiply(corrrecession.T,corrrecession.T)),np.multiply(sdirecession.T,sdirecession.T))))
				rnds[0,5]=0
				rnds[0,8]=0
				rnds[0,10]=0
				rnds[0,30]=0
				rnds[0,31]=0
				har = har + rnds.flatten('F')
			for ix in range(0,8):
				har[ix]=max(0.1,har[ix])
			for ix in [10,12,14,16]:
				har[ix]=max(0,har[ix])
			for ix in range(0,har.shape[0]):
				har[ix]=max(-99,har[ix])
			har3[0] = har3[1]
			har3[1] = har3[2]
			har3[2] = har
			thmf = np.vstack([thmf, np.append(np.append(hmf,rece),har.flatten())])
			hmf=np.append(hmf,1)
		for i in range(dyn,n):
			hmf = var.dot(hmf.T).T
			hmf3[0] = hmf3[1]
			hmf3[1] = hmf3[2]
			hmf3[2] = hmf
			recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
			rece = recession(RecFun,recessionVal)
			for j in range(0,har.shape[0]):
				hartmp = hmf3.flatten('F')
				hartmp = np.insert(hartmp,0,[1,har3[1,j],har3[2,j]])
				har[j] = fundmap[j].dot(hartmp.T).T
			for ix in range(0,8):
				har[ix]=max(0.1,har[ix])
			for ix in [10,12,14,16]:
				har[ix]=max(0,har[ix])
			for ix in range(0,har.shape[0]):
				har[ix]=max(-99,har[ix])
			har3[0] = har3[1]
			har3[1] = har3[2]
			har3[2] = har
			thmf = np.vstack([thmf, np.append(np.append(hmf,rece),har.flatten())])
			hmf=np.append(hmf,1)
		return thmf
	else:
		var = var[:,[range(0,9)]]
		hmf = histMF
		hmf3 = np.insert(hmf,0,hmf[hmf.shape[0]-2,:],0)
		recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
		rece = recession(RecFun,recessionVal)
		hmf = hmf[hmf.shape[0]-1,:]
		thmf = np.append(hmf,rece)
		sdtnormal = fundmap[:,[27]]/scale
		sdtrecession = fundmap[:,[28]]/scale
		sdinormal = fundmap[:,[29]]/scale
		sdirecession = fundmap[:,[30]]/scale
		corrnormal = fundmap[:,[31]]
		corrrecession = fundmap[:,[32]]
		fundmap = fundmap[:,[range(0,27)]]
		har = histAR
		har3 = np.insert(har,0,har[har.shape[0]-2,:],0)
		har = har[har.shape[0]-1,:]
		thmf = np.append(thmf,har)
		hmf=np.append(hmf,1)
		for i in range(0,n):
			hmf = var.dot(hmf.T).T
			hmf3[0] = hmf3[1]
			hmf3[1] = hmf3[2]
			hmf3[2] = hmf
			recessionVal = [hmf3[2,3],hmf3[2,7],hmf3[2,1],hmf3[2,0],hmf3[1,3],hmf3[1,7],hmf3[1,1],hmf3[1,0],hmf3[0,3],hmf3[0,7],hmf3[0,1],hmf3[0,0]]
			rece = recession(RecFun,recessionVal)
			for j in range(0,har.shape[0]):
				hartmp = hmf3.flatten('F')
				hartmp = np.insert(hartmp,0,[1,har3[1,j],har3[2,j]])
				har[j] = fundmap[j].dot(hartmp.T).T
			for ix in range(0,8):
				har[ix]=max(0.1,har[ix])
			for ix in [10,12,14,16]:
				har[ix]=max(0,har[ix])
			for ix in range(0,har.shape[0]):
				har[ix]=max(-99,har[ix])
			har3[0] = har3[1]
			har3[1] = har3[2]
			har3[2] = har
			thmf = np.vstack([thmf, np.append(np.append(hmf,rece),har.flatten())])
			hmf=np.append(hmf,1)
		return thmf

#Bond fund return calculator for each credit rating with a target duration and rebalancing strategy
def bondReturn(thmf,mix,migrationM,dyn=10,n=10,rating=2,rebalance=0.5,bondfreq=0.25,inter="linear",extro="tar",expectedRate=0.04):
#rating: 0-Govt Bond; 1-AAA; 2-AA; 3-A; 4-BBB
	term = np.array([0.25,1,2,3,5,7,10,20,30])
	initialVal = 1000000
	migrationM = np.array(migrationM)/100
	recoveryM = migrationM[:,6]*100
	mix = np.array(mix)
	#rating=2
	#bondfreq=0.25
	MVM = mix[rating] * initialVal
	value = thmf[0,[4,9,10,11,12,13,14,15,16]]
	if rating>0:
		value = value + thmf[0,16+rating*2]
	default= np.append(0,thmf[0,[19,21,23,25]])/100
	xnew = np.linspace(0.25,30,num=120,endpoint=True)
	inter="linear"
	if inter=="linear":
		f = interp1d(term,value)
	else:
		f = interp1d(term,value,kind='cubic')
	CouponM = np.array(f(xnew))
	FVM = np.copy(MVM)
	for i in range(0,120):
		FVM[i] = FVM[i]/bondPrice(term,value/100,(i+1)*0.25,bondfreq,CouponM[i]/100,1,inter=inter,extro=extro,expectedRate=expectedRate)
	nBSM = np.repeat(0.0,120)
	RedemptionM = np.repeat(0.0,120)
	tMVM = np.copy(MVM)
	tFVM = np.copy(FVM)
	tBSM = np.copy(nBSM)
	tRDM = np.copy(RedemptionM)
	tCRM = np.copy(CouponM)
	cashRtn = []
	priceRtn = []

	for i in range(1,thmf.shape[0]):
		bcurve = thmf[i,[4,9,10,11,12,13,14,15,16]]
		bcurve = np.vstack([bcurve,bcurve + thmf[i,18]])
		bcurve = np.vstack([bcurve,bcurve[0] + thmf[i,20]])
		bcurve = np.vstack([bcurve,bcurve[0] + thmf[i,22]])
		bcurve = np.vstack([bcurve,bcurve[0] + thmf[i,24]])
		bcurve = np.vstack([bcurve,bcurve[4] + 2.499197])
		#if i==1:
			#print bcurve
		nMVM = np.repeat(0.0,120)
		nFVM = np.repeat(0.0,120)
		nBSM = np.repeat(0.0,120)
		RedemptionM = np.repeat(0.0,120)

		totalBS = 0.0
		xnew = np.linspace(0.25,30,num=120,endpoint=True)
		inter="linear"
		if inter=="linear":
			f = interp1d(term,bcurve[rating])
		else:
			f = interp1d(term,bcurve[rating],kind='cubic')
		nCouponM = np.array(f(xnew))
		for j in range(0,120):
			for k in range(0,6):
				if k==rating:
					nMVM[j] = bondPrice(term,bcurve[rating]/100,(j+1)/4.0-0.25,bondfreq,CouponM[j]/100,FVM[j],inter=inter,extro=extro,expectedRate=expectedRate)*migrationM[rating,rating]*(1-default[rating]*(1-recoveryM[rating]))
					nFVM[j] = FVM[j]*migrationM[rating,rating]*(1-default[rating]*(1-recoveryM[rating]))
				else:
					totalBS = totalBS + bondPrice(term,bcurve[k]/100,(j+1)/4.0-0.25,bondfreq,CouponM[j]/100,FVM[j],inter=inter,extro=extro,expectedRate=expectedRate)*migrationM[rating,k]*(1-default[rating]*(1-recoveryM[rating]))
			if j==0:
				RedemptionM[j] = nMVM[j]*(1-default[rating]*(1-recoveryM[rating]))
			if ((j+1)%int(bondfreq*4)==1 and bondfreq>0.25) or bondfreq==0.25:
				RedemptionM[j] = RedemptionM[j] + FVM[j]*CouponM[j]/100*bondfreq*(1-default[rating]*(1-recoveryM[rating]))
	
		totalBS = totalBS + np.sum(RedemptionM)
		if rebalance==0:
			nBSM=totalBS*mix[rating]
			nBSFM = np.repeat(0.0,120)
			for j in range(0,120):
				nBSFM[j] = nBSM[j]/bondPrice(term,bcurve[rating]/100,(j+1)/4.0,bondfreq,nCouponM[j]/100,1)
			for j in range(1,120):
				MVM[j-1]=nMVM[j]
				FVM[j-1]=nFVM[j]
			MVM[119]=0
			FVM[119]=0
			for j in range(1,120):
				if (FVM[j-1]+nBSFM[j-1]) != 0:
					CouponM[j-1]=(FVM[j-1]*CouponM[j]+nBSFM[j-1]*nCouponM[j-1])/(FVM[j-1]+nBSFM[j-1])
				else:
					CouponM[j-1]=CouponM[j]
			CouponM[119]=nCouponM[119]
			MVM = MVM + nBSM
			FVM = FVM + nBSFM
		elif i%int(rebalance*4)==0:
			totalBS = totalBS + np.sum(nMVM)
			nBSM=totalBS*mix[rating]
			for j in range(0,119):
				nBSM[j] = nBSM[j]-nMVM[j+1]
			nBSFM = np.repeat(0.0,120)
			for j in range(0,120):
				if nBSM[j]<0:
					nBSFM[j] = nBSM[j]/bondPrice(term,bcurve[rating]/100,(j+1)/4.0,bondfreq,CouponM[j+1]/100,1,inter=inter,extro=extro,expectedRate=expectedRate)
				else:
					nBSFM[j] = nBSM[j]/bondPrice(term,bcurve[rating]/100,(j+1)/4.0,bondfreq,nCouponM[j]/100,1,inter=inter,extro=extro,expectedRate=expectedRate)				
			for j in range(1,120):
				MVM[j-1]=nMVM[j]
				FVM[j-1]=nFVM[j]
			MVM[119]=0
			FVM[119]=0
			for j in range(1,120):
				if nBSM[j-1]>0 and (FVM[j-1]+nBSFM[j-1])!=0:
					CouponM[j-1]=(FVM[j-1]*CouponM[j]+nBSFM[j-1]*nCouponM[j-1])/(FVM[j-1]+nBSFM[j-1])
				else:
					CouponM[j-1]=CouponM[j]
			CouponM[119]=nCouponM[119]
			MVM = MVM + nBSM
			FVM = FVM + nBSFM
		else:
			nBSM=totalBS*mix[rating]
			nBSFM = np.repeat(0.0,120)
			for j in range(0,120):
				nBSFM[j] = nBSM[j]/bondPrice(term,value/100,(j+1)/4,bondfreq,nCouponM[j]/100,1,inter=inter,extro=extro,expectedRate=expectedRate)
			for j in range(1,120):
				MVM[j-1]=nMVM[j]
				FVM[j-1]=nFVM[j]
			MVM[119]=0
			FVM[119]=0
			for j in range(1,120):
				if (FVM[j-1]+nBSFM[j-1]) != 0:
					CouponM[j-1]=(FVM[j-1]*CouponM[j]+nBSFM[j-1]*nCouponM[j-1])/(FVM[j-1]+nBSFM[j-1])
				else:
					CouponM[j-1]=CouponM[j]
			CouponM[119]=nCouponM[119]
			MVM = MVM + nBSM
			FVM = FVM + nBSFM
		tMVM = np.vstack([tMVM,MVM])
		tFVM = np.vstack([tFVM,FVM])
		tBSM = np.vstack([tBSM,nBSM])
		tRDM = np.vstack([tRDM,RedemptionM])
		tCRM = np.vstack([tCRM,CouponM])
		cashRtn = np.append(cashRtn,np.sum(RedemptionM)/initialVal)
		priceRtn = np.append(priceRtn,(np.sum(MVM)-np.sum(RedemptionM))/initialVal-1)
		default= np.append(0,thmf[i,[19,21,23,25]])/100
		initialVal = np.sum(MVM)
	return cashRtn,priceRtn


def AddBondReturn(thmf, mix, migrationM, dyn=10, n=10, rebalance=0, bondfreq=0.25, inter="linear", extro="tar", expectedRate=0.04):
    """Add bond fund returns to stochastic scenarios."""
    for ir in range(0,5):
        cashR,priceR = bondReturn(thmf=thmf,mix=mix,migrationM=migrationM,dyn=10,n=n,rating=ir,rebalance=rebalance,bondfreq=bondfreq,inter=inter,extro=extro,expectedRate=expectedRate)
        thmf = np.append(thmf,np.transpose(np.vstack([np.insert(cashR,0,0),np.insert(priceR,0,0)])),axis=1)
    return thmf

def AdjustBondReturn(thmf, thmfBase, thmfFull, dyn=10, n=10):
    """Adjust bond fund returns for periods after dynamic projection time."""
    thmf = np.append(thmf,thmfBase[:,range(42,52)],axis=1)
    for i in range(0,n):
        if i> dyn:
            for ir in [43,45,47,49,51]:
                thmf[i,ir] = thmfBase[i,ir]+(thmfBase[i,14]-thmfFull[i,14])/400
            for ir in [45,47,49,51]:
                thmf[i,ir] = thmfBase[i,ir]+(thmfFull[i,ir-27]-thmfBase[i,ir-27] - thmfFull[i,ir-26]+thmfBase[i,ir-26])/400
    return thmf

def pbo(gender,dateOfBirth, startDate,retireDate,salary,occupation,weight,thmf,dyn,inter="linear",extro="tar",target=0.04):
	wTerms = int(max(0,(retireDate.year - valuationDate.year)*4+(retireDate.month-valuationDate.month)/4))+1
	retireArray = np.repeat(0,wTerms)
	retireArray = np.append(retireArray,np.repeat(1,361-wTerms))
	servicePeriod = (valuationDate.year - startDate.year) + (valuationDate.month - startDate.month)/12.
	spArray = np.repeat(servicePeriod,361)
	age = (valuationDate.year - dateOfBirth.year) + (valuationDate.month - startDate.month)/12.
	ageArray = np.repeat(age,361)
	qScn = thmf.shape[0]-1
	wageIndex = np.repeat(1.,361)
	COLAArray = np.repeat(1.,361)
	salaryArray = np.repeat(salary,361)
	supSalaryArray = np.repeat(salary,361+salaryAvgPeriod*4-1)
	for i in range(0,salaryAvgPeriod*4-1):
		if i%4 == 0:
			supSalaryArray[i] = salary/(1+salaryGr*salaryMultiple[occupation-1])**(salaryAvgPeriod-1-i/4)
		else:
			supSalaryArray[i] = supSalaryArray[i-1]
	avgSalaryArray = np.repeat(np.average(supSalaryArray[0:salaryAvgPeriod*4]),361)
	accBenefitArray = np.repeat(0.,361)
	prjBenefitArray = np.repeat(0.,361)
	paymentArray = np.repeat(0.,361)
	prjPaymentArray = np.repeat(0.,361)

	mortTables = pd.read_csv(Mortality)
	mortTables = np.array(mortTables)
	if gender == "M":
		baseMortArray = np.repeat(1-(1-mortTables[min(110,int(round(age))),1])**0.25,361)
	else:
		baseMortArray = np.repeat(1-(1-mortTables[min(110,int(round(age))),2])**0.25,361)

	survivorship = np.repeat(1.,361)

	ldrArray = np.repeat(1.,361)
	pcdrArray = np.repeat(1.,361)
	lsdrArray = np.repeat(1.,361)

	value = thmf[dyn,[4,9,10,11,12,13,14,15,16]]
	value = value + thmf[dyn,20]
	xi,yi = curvefitting(value/100,term=np.array([0.25,1,2,3,5,7,10,20,30]),inter=inter,extro=extro,target=target)

	for t in range(1,361):

		if retireArray[t] == 1:
			spArray[t] = spArray[t-1]
		else:
			spArray[t] = servicePeriod + t/4.

		ageArray[t] = age + t/4.

		if retireArray[t] == 1:
			wageIndex[t] = wageIndex[t-1]
		elif t%4 == 0:
			wageIndex[t] = wageIndex[t-1]*(1+((1+thmf[min(t,qScn),38]/100)*(1+thmf[min(t-1,qScn),38]/100)*(1+thmf[min(t-2,qScn),38]/100)*(1+thmf[min(t-3,qScn),38]/100)-1)*salaryMultiple[occupation-1])
		else:
			wageIndex[t] = wageIndex[t-1]

		if retireArray[t] == 0:
			COLAArray[t] = 1
		elif t%4 == 0:
			COLAArray[t] = COLAArray[t-1] * min(1+maxCOLA,(1-COLA)+COLA*(1+thmf[min(t,qScn),17]/100)*(1+thmf[min(t-1,qScn),17]/100)*(1+thmf[min(t-2,qScn),17]/100)*(1+thmf[min(t-3,qScn),17]/100))
		else:
			COLAArray[t] = COLAArray[t-1]

		if retireArray[t] == 0:
			salaryArray[t] = salary * wageIndex[t]
			supSalaryArray[t+salaryAvgPeriod*4-1] = salaryArray[t]
		else:
			salaryArray[t] = salaryArray[t-1]
			supSalaryArray[t+salaryAvgPeriod*4-1] = salaryArray[t]

		avgSalaryArray[t] = np.average(supSalaryArray[t:(t+salaryAvgPeriod*4)])

		accBenefitArray[t] = avgSalaryArray[t]*benefitRate*min(servicePeriod+dyn/4.,spArray[t])*COLAArray[t]/4
		prjBenefitArray[t] = avgSalaryArray[t]*benefitRate*spArray[t]*COLAArray[t]/4

		if gender == "M":
			baseMortArray[t] = 1-(1-mortTables[min(110,int(round(ageArray[t]))),1])**0.25
			survivorship[t] = survivorship[t-1]*(1-baseMortArray[t]*(1-mMI))
		else:
			baseMortArray[t] = 1-(1-mortTables[min(110,int(round(ageArray[t]))),2])**0.25
			survivorship[t] = survivorship[t-1]*(1-baseMortArray[t]*(1-fMI))

		lsdrArray[t] = lsdrArray[t-1]/(1+lumpSumDR)**0.25
		mt = thmf.shape[0]-1
		if t <dyn:
			ldrArray[t] = 1
			pcdrArray[t] = 1
		else:
			ldrArray[t] = ldrArray[t-1]/(1+yi[t-dyn])**0.25

	lumpSum = np.sum(np.multiply(np.multiply(np.multiply(lsdrArray,accBenefitArray),survivorship),retireArray))/lsdrArray[wTerms]/survivorship[wTerms]
	for t in range(1,361):
		if wTerms == 0:
			paymentArray[t] = accBenefitArray[t]*retireArray[t]*survivorship[t]
			prjPaymentArray[t] = prjBenefitArray[t]*retireArray[t]*survivorship[t]
		elif t == wTerms:
			paymentArray[t] = accBenefitArray[t]*retireArray[t]*survivorship[t]*(1-lumpSumProb)+lumpSum*lumpSumProb
			prjPaymentArray[t] = prjBenefitArray[t]*retireArray[t]*survivorship[t]
		else:
			paymentArray[t] = accBenefitArray[t]*retireArray[t]*survivorship[t]*(1-lumpSumProb)
			prjPaymentArray[t] = prjBenefitArray[t]*retireArray[t]*survivorship[t]
	dynArray = np.repeat(0,361)
	for t in range(0,361):
		if t > dyn:
			dynArray[t]=1

	dynPBO = np.sum(np.multiply(np.multiply(paymentArray,ldrArray),dynArray))/survivorship[dyn]
	if retireArray[dyn]==0:
		planCost = 0.25/max(spArray)*np.sum(np.multiply(prjPaymentArray,pcdrArray))/survivorship[dyn]
	else:
		planCost = 0
	return [dynPBO,planCost,paymentArray[dyn],prjPaymentArray[dyn]]


def liabMultiple(employees,thmf,planliab=10000000.,inter="linear",extro="tar",target=0.04):
	liabSum = 0
	employees = np.array(employees)
	for employID in range(0,employees.shape[0]):
		gender=employees[employID,2]
		dateOfBirth=date(*map(int, employees[employID,3].split('-')))
		startDate=date(*map(int, employees[employID,4].split('-')))
		retireDate=date(*map(int, employees[employID,6].split('-')))
		salary=employees[employID,5]
		occupation=employees[employID,7]
		weight=employees[employID,8]
		wTerms = int(max(0,(retireDate.year - valuationDate.year)*4+(retireDate.month-valuationDate.month)/4))+1
		retireArray = np.repeat(0,wTerms)
		retireArray = np.append(retireArray,np.repeat(1,361-wTerms))
		servicePeriod = (valuationDate.year - startDate.year) + (valuationDate.month - startDate.month)/12.
		spArray = np.repeat(servicePeriod,361)
		age = (valuationDate.year - dateOfBirth.year) + (valuationDate.month - startDate.month)/12.
		ageArray = np.repeat(age,361)
		qScn = thmf.shape[0]-1
		wageIndex = np.repeat(1.,361)
		COLAArray = np.repeat(1.,361)
		salaryArray = np.repeat(salary,361)
		supSalaryArray = np.repeat(salary,361+salaryAvgPeriod*4-1)
		for i in range(0,salaryAvgPeriod*4-1):
			if i%4 == 0:
				supSalaryArray[i] = salary/(1+salaryGr*salaryMultiple[occupation-1])**(salaryAvgPeriod-1-i/4)
			else:
				supSalaryArray[i] = supSalaryArray[i-1]
		avgSalaryArray = np.repeat(np.average(supSalaryArray[0:salaryAvgPeriod*4]),361)
		accBenefitArray = np.repeat(0.,361)
		paymentArray = np.repeat(0.,361)

		mortTables = pd.read_csv(Mortality)
		mortTables = np.array(mortTables)
		if gender == "M":
			baseMortArray = np.repeat(1-(1-mortTables[min(110,int(round(age))),1])**0.25,361)
		else:
			baseMortArray = np.repeat(1-(1-mortTables[min(110,int(round(age))),2])**0.25,361)

		survivorship = np.repeat(1.,361)

		ldrArray = np.repeat(1.,361)
		lsdrArray = np.repeat(1.,361)

		value = thmf[0,[4,9,10,11,12,13,14,15,16]]
		value = value + thmf[0,20]
		xi,yi = curvefitting(value/100,term=np.array([0.25,1,2,3,5,7,10,20,30]),inter=inter,extro=extro,target=target)

		for t in range(1,361):

			if retireArray[t] == 1:
				spArray[t] = spArray[t-1]
			else:
				spArray[t] = servicePeriod + t/4.

			ageArray[t] = age + t/4.

			if retireArray[t] == 1:
				wageIndex[t] = wageIndex[t-1]
			elif t%4 == 0:
				wageIndex[t] = wageIndex[t-1]*(1+((1+thmf[min(t,qScn),38]/100)*(1+thmf[min(t-1,qScn),38]/100)*(1+thmf[min(t-2,qScn),38]/100)*(1+thmf[min(t-3,qScn),38]/100)-1)*salaryMultiple[occupation-1])
			else:
				wageIndex[t] = wageIndex[t-1]

			if retireArray[t] == 0:
				COLAArray[t] = 1
			elif t%4 == 0:
				COLAArray[t] = COLAArray[t-1] * min(1+maxCOLA,(1-COLA)+COLA*(1+thmf[min(t,qScn),17]/100)*(1+thmf[min(t-1,qScn),17]/100)*(1+thmf[min(t-2,qScn),17]/100)*(1+thmf[min(t-3,qScn),17]/100))
			else:
				COLAArray[t] = COLAArray[t-1]

			if retireArray[t] == 0:
				salaryArray[t] = salary * wageIndex[t]
				supSalaryArray[t+salaryAvgPeriod*4-1] = salaryArray[t]
			else:
				salaryArray[t] = salaryArray[t-1]
				supSalaryArray[t+salaryAvgPeriod*4-1] = salaryArray[t]

			avgSalaryArray[t] = np.average(supSalaryArray[t:(t+salaryAvgPeriod*4)])

			accBenefitArray[t] = avgSalaryArray[t]*benefitRate*servicePeriod*COLAArray[t]/4

			if gender == "M":
				baseMortArray[t] = 1-(1-mortTables[min(110,int(round(ageArray[t]))),1])**0.25
				survivorship[t] = survivorship[t-1]*(1-baseMortArray[t]*(1-mMI))
			else:
				baseMortArray[t] = 1-(1-mortTables[min(110,int(round(ageArray[t]))),2])**0.25
				survivorship[t] = survivorship[t-1]*(1-baseMortArray[t]*(1-fMI))

		
			lsdrArray[t] = lsdrArray[t-1]/(1+lumpSumDR)**0.25
			ldrArray[t] = ldrArray[t-1]/(1+yi[t])**0.25
	
		lumpSum = np.sum(np.multiply(np.multiply(np.multiply(lsdrArray,accBenefitArray),survivorship),retireArray))/lsdrArray[wTerms]/survivorship[wTerms]
		for t in range(1,361):
			if wTerms == 0:
				paymentArray[t] = accBenefitArray[t]*retireArray[t]*survivorship[t]
			elif t == wTerms:
				paymentArray[t] = accBenefitArray[t]*retireArray[t]*survivorship[t]*(1-lumpSumProb)+lumpSum*lumpSumProb
			else:
				paymentArray[t] = accBenefitArray[t]*retireArray[t]*survivorship[t]*(1-lumpSumProb)

		dynPBO = np.sum(np.multiply(paymentArray,ldrArray))
		#if employID==8:
			#print accBenefitArray,paymentArray
		liabSum = liabSum + dynPBO*weight
		#print(dynPBO, weight)
	multiple = planliab/liabSum
	return multiple

def aggPBO(employees, thmf, dyn, multiple, planliab=10000000., inter="linear", extro="tar", target=0.04):
    """Calculate aggregated projected benefit obligation for all plan participants."""
    aggResult = np.array([0,0,0,0])
    employees = np.array(employees)
    for employID in range(0,employees.shape[0]):
        gender=employees[employID,2]
        dateOfBirth=date(*map(int, employees[employID,3].split('-')))
        startDate=date(*map(int, employees[employID,4].split('-')))
        retireDate=date(*map(int, employees[employID,6].split('-')))
        salary=employees[employID,5]
        occupation=employees[employID,7]
        weight=employees[employID,8]
        aggResult = aggResult + np.multiply(weight, np.array(pbo(gender,dateOfBirth, startDate,retireDate,salary,occupation,weight,thmf,dyn,inter,extro,target)))
    aggResult = np.multiply(aggResult, multiple)
    return aggResult