#include <stdio.h>
#include <malloc.h>
#include <math.h>

/*
==================================================================================
* Reference: https://github.com/lzhengchun/optimization/blob/master/powell.cpp#L33
==================================================================================
*/

/*
==================================================================================
* function name: fitness_direction
* an interface to user defined fitness function, for one direction search
*    
* @para par: start x
* @para dir: search direction
* @para lam: the lambda (ratio, scaler)
* return values: 
*  fintness function value
==================================================================================
*/
void fitness_direction(int np, double *par, double *dir, double lam, void (*funct1)(int, double *, double *),
               double *par_new, double *yval_new){
    int i;
	for (i = 1; i <= np; ++i){
        par_new[i] = par[i] + lam * dir[i];
    }
    funct1(np,par_new,yval_new);
}

/*
==================================================================================
* function name: bracket
* bracket the interval of minimum point
*    
* @para par: start x
* @para dir: search direction
* @para lam1, start point
* @para h: initial search increment used in 'bracket'
* return values: 
*  the interval, upper and lower 
==================================================================================
*/

void bracket(int np, double *par, double *dir, double lam1, double h, void (*funct1)(int, double *, double *),
             double *par_new, double *a, double *b){
    double c = 1.618033989;
    double lam2, lam3;
    double f1, f2, f3;
    int i;

    lam2 = h + lam1;
    fitness_direction(np, par, dir, lam1, funct1, par_new, &f1);
    fitness_direction(np, par, dir, lam2, funct1, par_new, &f2);

    if(f2 > f1){
        h = -h;
        lam2 = lam1 + h;
        fitness_direction(np, par, dir, lam2, funct1, par_new, &f2);
        if(f2 > f1){
        	*a = lam2;
        	*b = lam1-h;
            return;
        }
    }

    for (i = 0; i < 100; ++i){
        h = c * h;
        lam3 = lam2 + h;
        fitness_direction(np, par, dir, lam3, funct1, par_new, &f3);
        if(f3 > f2){
        	*a = lam1;
        	*b = lam3;
        	return;
        }
        lam1 = lam2;
        lam2 = lam3;
        f1 = f2;
        f2 = f3;
    }
//    cout << "Bracket did not find a mimimum" << endl;
    *a = 0.0;
    *b = 0.0;
    return;
}
/*
==================================================================================
* function name: golden_section_search
* The golden section search is a technique for finding the extremum 
* (minimum or maximum) of a strictly unimodal function by successively narrowing 
* the range of values inside which the extremum is known to exist. 
* i.e., find lambda to min f(x+lam*v)
*    
* @para par: start x
* @para dir: search direction
* @para a, b: search interval [a,b]
* @para tol: tolerente of variables, i.e., the final interval width
* return values: 
*   lambda and minimum value found 
==================================================================================
*/
void golden_section_search(int np, double *par, double *dir, double a, double b, double tol,
                           void (*funct1)(int, double *, double *),
						   double *par_new, double *lam, double *fval){
    // compute the number of telescoping operations required to 
    // reduce h from |b − a| to an error tolerance
    int nIter = ceil(-2.078087*(log(tol)-log(fabs(b-a))));
    int i;
    double R = 0.618033989;   // golden ratio
    double C = 1.0 - R;
    // First telescoping
    double lam1 = R * a + C * b;
    double lam2 = C * a + R * b;
    double f1, f2;
    fitness_direction(np, par, dir, lam1, funct1, par_new, &f1);
    fitness_direction(np, par, dir, lam2, funct1, par_new, &f2);
    for (i = 0; i < nIter; ++i){
        if (f1 > f2){
            a = lam1;
            lam1 = lam2;
            f1 = f2;
            lam2 = C * a + R * b;
            fitness_direction(np, par, dir, lam2, funct1, par_new, &f2);
        }
        else{
            b = lam2;
            lam2 = lam1;
            f2 = f1;
            lam1 = R * a + C * b;
            fitness_direction(np, par, dir, lam1, funct1, par_new, &f1);
        }
    }
    if(f1 < f2){
    	*lam = lam1;
    	*fval = f1;
        return;
    }
    else{
    	*lam = lam2;
    	*fval = f2;
        return;
    }
}

/*
==================================================================================
* function name: mse
* calculating Mean squared error
*    
* @para v1: the first vector
* @para v2: the second vector
* return values: 
*   Mean squared error 
==================================================================================
*/
double mse(int np, double *dir1, double *dir2){
    double len = 0.0;
    double temp;
    int i;
    for (i = 1; i <= np; ++i){
    	temp = dir1[i] - dir2[i];
        len += temp * temp;
    }
    return sqrt(len / np);
}
/*
==================================================================================
* function name: min_powell
* Powell's method of minimizing user-supplied function
* without calculating its derivatives
*    
* @para x: starting point 
* @para h: initial search increment used in 'bracket'
* @para tolerate:
* @para maxit: maximum iterations
* return values: 
*   a set of parameters which will carry out the minimum (local)    
==================================================================================
*/
void min_powell(int np, double *par, double **u, double *yval, int *iter, int maxit, 
                int *ifail, int mon, void (*funct1)(int, double *, double *), double h)
{
    int i,j,kk,i_max;
    double tolerate = 1.0e-6;
    double fitness_old, a, b, s, temp;
    double *fitness_dir_min, *df, *par_new, *par_old, *dir, *dir_new;

    /* allocate space */
    par_new = (double *) malloc((np+1) * sizeof(double));
    par_old = (double *) malloc((np+1) * sizeof(double));
    dir_new = (double *) malloc((np+1) * sizeof(double));
    fitness_dir_min = (double *) malloc((np+1) * sizeof(double));
    df = (double *) malloc((np+1) * sizeof(double));

    *ifail = 0;

    // direction vectors v stored here by rows
    // set direction vectors
    for (i = 1; i <= np; ++i){
        for (j = 1; j <= np; ++j)
          u[i][j] = 0.0;
        u[i][i] = 1.0;
    }

    // main iteration loop
    for (j = 0; j < maxit; j++) { 
        for(i=1; i<=np; ++i){
        	par_old[i] = par[i];
		}
		funct1(np,par_old,&fitness_old);
		fitness_dir_min[0] = fitness_old;
        for (i = 1; i <= np; ++i){
            dir = u[i];
            bracket(np, par, dir, 0.0, h, funct1, par_new, &a, &b);
            golden_section_search(np, par, dir, a, b, 1.0e-9, funct1, par_new, &s, &temp);
            fitness_dir_min[i] = temp;
            for (kk = 1; kk <= np; ++kk){
                par[kk] = par_new[kk];
            }
       	if(mon==1){
            for(kk=1;kk<=np;kk++){
    	       printf("%f ",par[kk]);
	        }
	        printf("| %f %f %f %f \n",a,b,s,temp);
        }
       }
        for (i = 0; i < np; ++i){
            df[i+1] = fitness_dir_min[i] - fitness_dir_min[i+1];
        }
        // Last line gloden section search in the cycle    
        for (i = 1; i <= np; ++i){
            dir_new[i] = par[i] - par_old[i];
        }
        bracket(np, par, dir_new, 0.0, h, funct1, par_new, &a, &b);
        golden_section_search(np, par, dir_new, a, b, 1.0e-9, funct1, par_new, &s, &temp);
        // dependence among search directions
        for (i = 1; i <= np; ++i){
            par[i] = par_new[i];
        }

      	if(mon==1){
            for(kk=1;kk<=np;kk++){
    	       printf("%f ",par[kk]);
	        }
	        printf("| %f %f %f %f \n",a,b,s,temp);
        }
        // Check for convergence
        if(mse(np,par, par_old) < tolerate){
        	*iter = j+1;
        	funct1(np,par,yval);
        	if(mon==1){
              printf("found minimize value at %d step with value: %f\n", *iter, *yval);
			}
			free(par_new); free(par_old); free(fitness_dir_min); free(dir_new); free(df);
            return;
        }
        i_max = 1;
        for (i = 2; i <= np; ++i){
            if(df[i] > df[i_max]){
                i_max = i;
            }
        }
        for (i = i_max; i <= np-1; ++i){
        	for(kk = 1;kk<=np;++kk){
              u[i][kk] = u[i+1][kk];
			}
        }
       	for(kk = 1;kk<=np;++kk){
            u[np][kk] = dir_new[kk];
        }

    }
    *ifail = 1;
    if(mon==1){
      printf("Powell did not converge\n");
    }
	free(par_new); free(par_old); free(fitness_dir_min); free(dir_new); free(df);
    return;
}


/*=======================================================================
 * Test the program

void mytestfunc(int np, double *par, double *yval){
    double x, y, z;
    x = par[1];
    y = par[2];
    z = par[3];
    double f = 1.0 / (1.0 + (x-y)*(x-y)) + sin(0.5 * y * z) + exp(-((x+z)/y-2)*((x+z)/y-2));
//    *yval = (x-2)*(x-2)+(y-4)*(y-4)+(z+3)*(z+3);
    *yval = -f;
}

double **ddmatrix(double *avec, int nrow, int ncol)
{ int i;
  double **amat;
  amat=(double **)malloc((unsigned) nrow*sizeof(double*));
  for(i=0;i<nrow;i++) amat[i]=avec+i*ncol;
  return amat;
}

int main(int argc, char const *argv[])
{
    int n_unknown, mon, iter, ifail, nevals;
    double yval,h;
	double *par;
	double **u;
	
	mon = 1;
	nevals = 1000;
	h = 0.1;
	n_unknown = 3;
    par = (double *) malloc((n_unknown+1) * sizeof(double));
    u=ddmatrix((double *)malloc((n_unknown+1)*(n_unknown+1)*sizeof(double)),n_unknown+1,n_unknown+1);  

    par[1] = 0.11; par[2] = 0.18; par[3] = 0.1;

    min_powell(n_unknown,par,u,&yval,&iter,nevals,&ifail,mon,mytestfunc,h);
	
    for(int i=1;i<=n_unknown;i++){
    	printf("%f ",par[i]);
	}
	printf("\n");
    free(par); free(u[0]); free(u);
    return 0;
}

*/

