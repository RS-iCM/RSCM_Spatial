#include <stdio.h>
#include <stdlib.h>
#include <malloc.h>
#include <math.h>
#include <time.h>
#include "powell_min.c"

/*********************************************************************************************
  Notes:
    In C, memory for matrices is allocated such that rows occupy contiguous space.
    avec: the array storing the elements in the matrix or 3D array.
    (nrow,ncol): the dimension of the matrix.
    (ndep,nrow,ncol): the dimension of the 3D array.
*********************************************************************************************/

/* Create a matrix of double precision floating point numbers. */
double **ddmatrix(double *avec, int nrow, int ncol)
{ int i;
  double **amat;
  amat=(double **)malloc((unsigned) nrow*sizeof(double*));
  for(i=0;i<nrow;i++) amat[i]=avec+i*ncol;
  return amat;
}

/* Create a matrix of integers. */
int **bbmatrix(int *avec, int nrow, int ncol)
{ int i;
  int **amat;
  amat=(int **)malloc((unsigned) nrow*sizeof(int*));
  for(i=0;i<nrow;i++) amat[i]=avec+i*ncol;
  return amat;
}

/* Create a 3D array of double precision floating point numbers. */
double ***d3array(double *avec, int nrow, int ncol, int ndep)
{ int i;
  double **a2;
  double ***a3;
  a3=(double ***)malloc((unsigned) nrow*sizeof(double**));
  a2=(double **)malloc((unsigned) nrow*ncol*sizeof(double*));
  for(i=0;i<nrow*ncol;i++)
      a2[i] = avec+i*ndep;
  for(i=0;i<nrow;i++)
    a3[i]=a2+i*ncol;
  return a3;
}

/*********************************************************************************************
  Notes:
    The empirical relationships between VIs and LAI are described using suitable
    linear or nonlinear regression models. Users can modify the following functions.
  Parameters:
    LAI: Leaf-area index.
    VIs: Vegetation indexes.
    a,b: Regression coefficients.
  Functions: (?) can be 0/1.
    err_fit_? :
	  Specify the error of the regression model and
	  is called in RSCM(...) and GetCoef(...).
    err_fit_reparam_? and reparam_?:
	  Specify the error of the regression model and
	  is called in RSCM(...) and GetLAI(...).
	Re-parameterization is used in GetLAI(...) to
	  avoid that the LAI value implied from VIs becomes negative.
*********************************************************************************************/

double sqr(double x) { return x*x; }

/* Returning the error of log-log linear regression. */
double err_fit_0(double a, double b, double LAI, double VI){
  return log(VI)-a-b*log(LAI);
}

double err_fit_reparam_0(double a, double b, double x, double VI){
  return log(VI)-a-b*x;
}

double reparam_0(double x){
  return exp(x);
}

double err_fit_1(double a, double b, double LAI, double VI){
  return VI-a*tanh(b*LAI);
//  return VI-a*pow(LAI,b);
}

double err_fit_reparam_1(double a, double b, double x, double VI){
  return VI-a*tanh(b*x);
//  return VI-a*exp(b*x);
}

double reparam_1(double x){
//  return x;
  return exp(x);
}


/*********************************************************************************************
  RSCM: Estimate the parameters (a,b,c,L0,rGDD) in the Remote Sensing and Crop Model (RSCM)
        from a data of LAI or data of VIs

  Inputs (p refers to pointer):

    Crop-associated parameters:
    Tbase : base temperature                                     
    k     : Light extinction coefficient                             
    RUE   : radiation use efficiency                               
    SLA   : specific leaf area                                     
    beta1 : ratio of SR to PAR                                   
    eGDD  : GDD at plant emergence                                
    pd    : parameter d

    fmLAI    : factor of max LAI (if the implied LAI exceeds fmLAI, program set LAI to fmLAI)
    start    : starting date of planting
    nrecords : maximum number of days allowed in simulation since the starting date
    npixels  : the number of pixels

    oOpt: 0 = Least square using LAI data
               1 = Bayesian using LAI data
               2 = Least square using VIs data
               3 = Bayesian using VIs data
    plantingDateOpt: To specify the way to input the starting date
                   0 means to use a common starting date (day of year) for all pixels
                   1 means to input the starting dates pixel by pixel from a file
    regressionOpt: To specify the regression model of LAI against VIs, 0/1
    impliedLAIOpt: If VIs are inputted,
	                 0 means to imply LAI from the regression model and fit the RSCM based on the implied LAI
	                 1 means to fit the RSCM based on the VIs

    n_VI     : Number of vegetation indexes (VIs)
    nObs     : number of days with observed LAI/VIs data
    obs_d_vec    : array storing the observed LAI/VIs data

    n_wcol   : Number of columns in the weather data
    ndoy     : number of days in the year
    wx_data_vec    : array storing the daily weather data

    n_unknown: Number of unknowns, 5 should be used for (a,b,c,L0,rGDD)
    para0vec : Initial guesses
    prior_inv_cov_vec: The inverse of covariance matrix of the prior under Bayesian approach
    prior_mean_vec: The mean of the prior under Bayesian approach
    coef_vec: Regression coefficients when regressing VIs against LAI
    reginvSigmavec: Inverse of covariance matrix of the errors when regressing VIs against LAI
    
  paraouts:
    paraout    : estimated parameters (a,b,c,L0,rGDD)
    LAIobs : LAI (oOpt=0,1) or implied LAI (oOpt=2,3)

*********************************************************************************************/

void RSCM_Multiple(double *pTbase, double *pk, double *pRUE, double *pSLA, double *pbeta1, double *peGDD,
             double *pd, double *pfmLAI, int *poOpt, int *pn_unknown, int *pn_VI, int *pn_wcol, int *pplantingDateOpt, int *pregressionOpt, int *pimpliedLAIOpt,
             int *pstart, int *pndoy, int *pnrecords, int *pnObs, int *pnpixels,
             double *para0vec, double *prior_inv_cov_vec, double *prior_mean_vec, double *coef_vec, double *reginvSigmavec,
             double *wx_data_vec, double *obs_d_vec, int *psub_paddyFields, int *psub_plantingDate,
             double *paraout, double *LAIobs){

  double Tbase,k,RUE,SLA,beta1,eGDD,d,fmLAI;
  int oOpt,n_unknown,n_VI,n_wcol,start,ndoy,nrecords,nObs,npixels,plantingDateOpt,regressionOpt,impliedLAIOpt;

  double *para0, **prior_inv_cov, *prior_mean, **coef, **reg_inv_Sigma, **param_all, **impliedLAI;
  double ***wxData, ***rs_data;

  int bComplete, nNonEmptyObs, *ODOYarr, *Indexarr, OneDate;
  double dGDD, *LAI, *GDDarr, *RADarr;
  double *TempRes;

  double **h,**hLAI,tolder,*param,paramLAI[2];
  int mon,ifail,nevals,iter;
  double yval;
  int count,current,i,j;

  // Generate LAI given the parameters (a,b,c,L0,rGDD).
  void generate_LAI(double a, double b, double c, double L0, double rGDD){
    int i;
    double APAR, dM, PF, dLAI, AGDM = 0.;

    LAI[0] = L0;
    for(i=0;i<=nrecords-2;i++){
      APAR = beta1*RADarr[i+start]*(1-exp(-k*LAI[i]));
      dM = RUE*APAR;                       //Biomass conversion based on the Monteith's law
      PF = 1.0/(1.0+a*exp(b*(GDDarr[i+1]-eGDD)));     // Leaf partitioning function
      dLAI = SLA * PF * dM;
      if((PF == 0.0)||(GDDarr[i+1]>=rGDD)){
        dLAI = - dLAI * c * (AGDM/d);
      }
      LAI[i+1] = LAI[i] + dLAI;
      if(LAI[i] <= 0.0) LAI[i] = 0.0;
      AGDM = AGDM + dM;
    }
  }

  // Sum of squared error comparing the generated LAI and the inputted LAI.
  // Optimization is further used to obtain the unknown parameters (a,b,c,L0,rGDD).
  // Inputs: the parameters (a,b,c,L0,rGDD)
  void SS(int n_unknown, double *vpar, double *obj){
    int i,j,kk,iNonEmpty,iObs,ODOY,sDOY;
    double a,b,c,L0,rGDD,temp,temp2,TempVI,**VIarr,*impliedLAIarr;

    a = 1/(1+exp(-vpar[1]));
    b = 1/(1+exp(-vpar[2]));
    c = 1/(1+exp(-vpar[3]));
    L0 = 1/(1+exp(-vpar[4]));
    rGDD = exp(vpar[5]);
    generate_LAI(a, b, c, L0, rGDD);

    iNonEmpty = 0;
    iObs = Indexarr[iNonEmpty];
    ODOY = ODOYarr[iNonEmpty];

    temp = 0;
    VIarr = rs_data[current];
    impliedLAIarr = impliedLAI[current];
    for(i=0;i<nrecords-2;i++){
      sDOY = floor(wxData[current][i+start][0]);
      if(sDOY==ODOY){
        if((impliedLAIOpt==0)&&(oOpt>1)){
          for(j=0;j<n_VI;j++){
            if(regressionOpt==0)
              TempRes[j] = err_fit_0(coef[j][0],coef[j][1],LAI[i+1],VIarr[iObs][j+1]);
            else if(regressionOpt==1)
              TempRes[j] = err_fit_1(coef[0][j],coef[1][j],LAI[i+1],VIarr[iObs][j+1]);
          }
          for(j=0;j<n_VI;j++){
            TempVI = 0;
            for(kk=0;kk<n_VI;kk++)
              TempVI += reg_inv_Sigma[j][kk]*TempRes[kk];
            temp += TempVI*TempRes[j];
          }
        }else{
          temp += sqr(log(impliedLAIarr[iObs])-log(LAI[i+1]));
        }

        if(iNonEmpty<nNonEmptyObs-1){
          iNonEmpty ++;
          iObs = Indexarr[iNonEmpty];
          ODOY = ODOYarr[iNonEmpty];
        }
      }

    }
    *obj = nNonEmptyObs*log(temp/nNonEmptyObs);
    if((oOpt==1)||(oOpt==3)){
      temp2=0;
      for(i=0;i<n_unknown;i++){
        TempVI = 0;
        for(j=0;j<n_unknown;j++)
          TempVI += prior_inv_cov[i][j]*(vpar[j+1]-prior_mean[j]);
        temp2 += TempVI*(vpar[i+1]-prior_mean[i]);
      }
      *obj += temp2;
    }
  }

  // Sum of squared error comparing the observing VIs and a given LAI value.
  // Optimization is further used to guess the unobserved LAI from the observed VIs
  // Input: LAI
  void SS_LAI(int n_unknown, double *vpar, double *obj){
    int j,kk;
    double x, temp,TempVI;

    x = vpar[1];
    temp = 0;
    for(j=0;j<n_VI;j++){
      if(regressionOpt==0)
        TempRes[j] = err_fit_reparam_0(coef[j][0],coef[j][1],x,rs_data[current][i][j+1]);
      else if(regressionOpt==1)
        TempRes[j] = err_fit_reparam_1(coef[j][0],coef[j][1],x,rs_data[current][i][j+1]);
    }
    for(j=0;j<n_VI;j++){
      TempVI = 0;
      for(kk=0;kk<n_VI;kk++)
        TempVI += reg_inv_Sigma[j][kk]*TempRes[kk];
      temp += TempVI*TempRes[j];
    }
    *obj = temp;
  }

  Tbase = *pTbase;
  k = *pk;
  RUE = *pRUE;
  SLA = *pSLA;
  beta1 = *pbeta1;
  eGDD = *peGDD;
  d = *pd;
  fmLAI = *pfmLAI;
  oOpt = *poOpt;
  n_unknown = *pn_unknown;
  n_VI = oOpt > 1 ? *pn_VI : 1;
  plantingDateOpt = *pplantingDateOpt;
  regressionOpt = *pregressionOpt;
  impliedLAIOpt = *pimpliedLAIOpt;
  ndoy = *pndoy;
  nrecords = *pnrecords;
  nObs = *pnObs;
  n_wcol = *pn_wcol;
  npixels = *pnpixels;

  para0 = para0vec;
  prior_inv_cov = ddmatrix(prior_inv_cov_vec,n_unknown,n_unknown);
  prior_mean = prior_mean_vec;
  coef = ddmatrix(coef_vec,n_VI,2);
  reg_inv_Sigma = ddmatrix(reginvSigmavec,n_VI,n_VI);

  wxData = d3array(wx_data_vec,npixels,ndoy,n_wcol);
  rs_data = d3array(obs_d_vec,npixels,nObs,n_VI+1);

  LAI = (double *)malloc((nrecords)*sizeof(double));
  GDDarr = (double *)malloc((nrecords)*sizeof(double));
  RADarr = (double *)malloc((ndoy)*sizeof(double));
  ODOYarr = (int *)malloc((nObs)*sizeof(int));
  Indexarr = (int *)malloc((nObs)*sizeof(int));
  TempRes = (double *)malloc((n_VI)*sizeof(double));
  param = (double *)malloc((n_unknown+1)*sizeof(double));
  h=ddmatrix((double *)malloc((n_unknown+1)*(n_unknown+1)*sizeof(double)),n_unknown+1,n_unknown+1);  // zero index not used by min_powell
  hLAI=ddmatrix((double *)malloc(2*2*sizeof(double)),2,2);  // zero index not used by min_powell
  param_all = ddmatrix(paraout,npixels,n_unknown);
  impliedLAI = ddmatrix(LAIobs,npixels,nObs);

  /* Quasi-Newton:
     mon=1 for monitor QN iterations
     nevals = # function evaluations for min_powell
     tolder = step size for numerical derivative if no analytic deriv
     h = Hessian matrix
     ifail=0 if QN converged OK
  */
  /* Powell:
     mon=1 for monitor Powell iterations
     nevals = # function evaluations for min_powell
     tolder = step size for bracketing a local minimum
     h = Matrix containing n directions
     ifail=0 if QN converged OK
  */

  nevals=1000;
  mon=0; 
  tolder=0.01;  // Choose tolder=0.01 for Powell and tolder=1e-5 for Quasi-Newton

  count = 0;
  for(current=0;current<npixels;current++){
    if(psub_paddyFields[current]==1){
      start = plantingDateOpt ? psub_plantingDate[current] : *pstart;
      nNonEmptyObs = 0;
      for(i=0;i<nObs;i++){
        bComplete = 1;
        OneDate = floor(rs_data[current][i][0]);
        if(OneDate<start) bComplete = 0;
        for(j=1;j<=n_VI;j++){
          if(rs_data[current][i][j]<=0)
            bComplete = 0;
        }
        if(bComplete){
          Indexarr[nNonEmptyObs] = i;
          ODOYarr[nNonEmptyObs] = OneDate;
          nNonEmptyObs ++;

          if(oOpt>1){
            paramLAI[1] = 0.5;
            min_powell(1,paramLAI,hLAI,&yval,&iter,nevals,&ifail,mon,SS_LAI,tolder);
            if(regressionOpt==0)
              impliedLAI[current][i] = reparam_0(paramLAI[1])<fmLAI ? reparam_0(paramLAI[1]) : fmLAI;
            else if(regressionOpt==1)
              impliedLAI[current][i] = reparam_1(paramLAI[1])<fmLAI ? reparam_1(paramLAI[1]) : fmLAI;
          }else{
            impliedLAI[current][i] = rs_data[current][i][1];
          }
        }
      }
      if(nNonEmptyObs != 0){
        GDDarr[0] = 0;
        for(i=0;i<=nrecords-2;i++){
          dGDD = (wxData[current][i+start][2]+wxData[current][i+start][3])/2-Tbase;
          GDDarr[i+1] = GDDarr[i] + (dGDD > 0 ? dGDD : 0);
          RADarr[i+start] = wxData[current][i+start][1];
        }
        
        for(i=0;i<n_unknown-1;i++)
          param[i+1] = -log(1/para0[i]-1);
        param[n_unknown] = log(para0[n_unknown-1]);
        min_powell(n_unknown,param,h,&yval,&iter,nevals,&ifail,mon,SS,tolder);
        if(!ifail){
#ifdef MONITOR
          printf("%d\n",iter);
#endif
          for(i=1;i<=n_unknown-1;i++){
            param_all[current][i-1] = 1/(1+exp(-param[i]));
          }
          param_all[current][n_unknown-1] = exp(param[n_unknown]);
        }
        count ++;
      }
    }
#ifdef MONITOR
    printf("%d pixels are scanned.\n",current+1);
#endif
  }

  // Free the memory.
  free(LAI);
  free(GDDarr);
  free(RADarr);
  free(Indexarr);
  free(ODOYarr);
  free(TempRes);
  free(param);
  free(h[0]); free(h);
  free(hLAI[0]); free(hLAI);

  free(param_all); free(impliedLAI); free(prior_inv_cov); free(coef); free(reg_inv_Sigma);
  free(wxData[0]);  free(wxData);
  free(rs_data[0]);  free(rs_data);
}

/*********************************************************************************************
  GetCoef: To obtain regression coefficients from a dataset containing both LAI and VIs.

  Inputs:
    regressionOpt: To specify the regression model of LAI against VIs, 0/1
    n_VI     : Number of vegetation indexes (VIs)
    nObs     : number of days with observed LAI/VIs data
    datavec  : The arry storing the data containing both LAI and VIs
  
  Outputs:
    coef_vec: Regression coefficients when regressing VIs against LAI
    Sigmavec: Inverse of covariance matrix of the errors when regressing VIs against LAI
*********************************************************************************************/

void GetCoef(double *datavec, int *pnObs, int *pn_VI, int *pregressionOpt, double *coef_vec, double *Sigmavec){
  int nObs, n_VI, n_unknown, regressionOpt;
  double **data;

  double **h,tolder,*param,**Sigma,**coef;
  int mon,ifail,nevals,iter;
  double yval;

  int i,j,k;
  
  nObs = *pnObs;
  n_VI = *pn_VI;
  regressionOpt = *pregressionOpt;
  n_unknown = 2;
  data = ddmatrix(datavec,nObs,n_VI+2);
  Sigma = ddmatrix(Sigmavec,n_VI, n_VI);
  coef = ddmatrix(coef_vec,n_VI,2);

  void SS(int n_unknown, double *vpar, double *obj){
    double a,b,temp;
    int i;
   
    a = vpar[1];
    b = vpar[2];
    temp = 0.;
    for(i=0;i<nObs;i++){
      if(regressionOpt==1)
        temp += sqr(err_fit_1(a,b,data[i][1],data[i][k]));
      else if(regressionOpt==0)
        temp += sqr(err_fit_0(a,b,data[i][1],data[i][k]));
    }
    *obj = temp;
  }
  
  nevals=1000;
  mon=0;
  tolder=0.01;
  param = (double *)malloc((n_unknown+1)*sizeof(double));
  h=ddmatrix((double *)malloc((n_unknown+1)*(n_unknown+1)*sizeof(double)),n_unknown+1,n_unknown+1);  // zero index not used by min_powell
  for(k=2;k<2+n_VI;k++){
    param[1] = 1;
    param[2] = 1;
    min_powell(n_unknown,param,h,&yval,&iter,nevals,&ifail,mon,SS,tolder);
    for(i=1;i<=n_unknown;i++)
      coef[k-2][i-1] = param[i];
  }
  for(k=0;k<n_VI;k++){
    for(j=0;j<n_VI;j++){
      Sigma[k][j] = 0;
      for(i=0;i<nObs;i++){
        if(regressionOpt==1)
          Sigma[k][j] += err_fit_1(coef[k][0],coef[k][1],data[i][1],data[i][k+2])
                          * err_fit_1(coef[j][0],coef[j][1],data[i][1],data[i][j+2])  ;
        else if(regressionOpt==0)
          Sigma[k][j] += err_fit_0(coef[k][0],coef[k][1],data[i][1],data[i][k+2])
                          * err_fit_0(coef[j][0],coef[j][1],data[i][1],data[i][j+2])  ;
      }
      Sigma[k][j] = Sigma[k][j]/(nObs-2);
    }
  }
  free(param);
  free(h[0]); free(h);
  free(data);
  free(Sigma);
  free(coef);
}

/*********************************************************************************************
  GetLAI: Imply LAI from VIs given the regression coefficients.

  Inputs (p refers to pointer):
    fmLAI    : factor of max LAI (if the implied LAI exceeds fmLAI, program set LAI to fmLAI)
    regressionOpt: To specify the regression model of LAI against VIs, 0/1

    n_VI     : Number of vegetation indexes (VIs)
    nObs     : number of days with observed LAI/VIs data
    obs_d_vec    : array storing the observed VIs data

    coef_vec: Regression coefficients when regressing VIs against LAI
    reginvSigmavec: Inverse of covariance matrix of the errors when regressing VIs against LAI
    
  Outputs:
    impliedLAI : implied LAI

*********************************************************************************************/

void GetLAI(double *pfmLAI, int *pn_VI, int *pregressionOpt,
             int *pnObs,
             double *coef_vec, double *reginvSigmavec,
             double *obs_d_vec,
             double *impliedLAI){

  double fmLAI;
  int n_VI,nObs,regressionOpt;
  double **coef, **reg_inv_Sigma;
  double **rs_data;

  double *TempRes;

  double **hLAI,tolder,paramLAI[2];
  int mon,ifail,nevals,iter;
  double yval;
  int i;


  // Sum of squared error comparing the observing VIs and a given LAI value.
  // Optimization is further used to guess the unobserved LAI from the observed VIs
  // Input: LAI
  void SS_LAI(int n_unknown, double *vpar, double *obj){
    int j,kk;
    double x, temp,TempVI;

    x = vpar[1];
    temp = 0;
    for(j=0;j<n_VI;j++){
      if(regressionOpt==0)
        TempRes[j] = err_fit_reparam_0(coef[j][0],coef[j][1],x,rs_data[i][j+1]);
      else if(regressionOpt==1)
        TempRes[j] = err_fit_reparam_1(coef[j][0],coef[j][1],x,rs_data[i][j+1]);
    }
    for(j=0;j<n_VI;j++){
      TempVI = 0;
      for(kk=0;kk<n_VI;kk++)
        TempVI += reg_inv_Sigma[j][kk]*TempRes[kk];
      temp += TempVI*TempRes[j];
    }
    *obj = temp;
  }

  fmLAI = *pfmLAI;
  n_VI = *pn_VI;
  regressionOpt = *pregressionOpt;
  nObs = *pnObs;

  coef = ddmatrix(coef_vec,n_VI,2);
  reg_inv_Sigma = ddmatrix(reginvSigmavec,n_VI,n_VI);
  rs_data = ddmatrix(obs_d_vec,nObs,n_VI+1);

  TempRes = (double *)malloc((n_VI)*sizeof(double));
  hLAI=ddmatrix((double *)malloc(2*2*sizeof(double)),2,2);  // zero index not used by min_powell

  /* mon=1 for monitor QN iterations
     nevals = # function evaluations for min_powell
     tolder = step size for numerical derivative if no analytic deriv
     h = Hessian matrix
     ifail=0 if QN converged OK
  */
  nevals=1000;
  mon=0; 
  tolder=0.01;

  for(i=0;i<nObs;i++){
    paramLAI[1] = 0.5;
    min_powell(1,paramLAI,hLAI,&yval,&iter,nevals,&ifail,mon,SS_LAI,tolder);
    if(regressionOpt==0)
      impliedLAI[i] = reparam_0(paramLAI[1])<fmLAI ? reparam_0(paramLAI[1]) : fmLAI;
    else if(regressionOpt==1)
      impliedLAI[i] = reparam_1(paramLAI[1])<fmLAI ? reparam_1(paramLAI[1]) : fmLAI;
  }

  // Free the memory.
  free(TempRes);
  free(hLAI[0]); free(hLAI);
  free(coef); free(reg_inv_Sigma); free(rs_data);
}
