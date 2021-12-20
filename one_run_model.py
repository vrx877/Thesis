# PACKAGES
from types import SimpleNamespace
import numpy as np
from scipy import optimize
from scipy import interpolate
import warnings

# Technical settings
n_gridpoints = 121 # This gives even grid spacing of 100.000 DKK

# MAIN MODEL ESTIMATION FUNCTION
def estimate(par,hh):
    decisions_5 = solve_period(5,par,hh,future=0)
    u_5 = interpolate.interp1d(decisions_5[0], decisions_5[1][:,0], bounds_error=False,fill_value='extrapolate')

    decisions_4 = solve_period(4,par,hh,future=u_5)
    u_4 = interpolate.interp1d(decisions_4[0], decisions_4[1][:,0], bounds_error=False,fill_value='extrapolate')

    decisions_3 = solve_period(3,par,hh,future=u_4)
    u_3 = interpolate.interp1d(decisions_3[0], decisions_3[1][:,0], bounds_error=False,fill_value='extrapolate')

    decisions_2 = solve_period(2,par,hh,future=u_3)
    u_2 = interpolate.interp1d(decisions_2[0], decisions_2[1][:,0], bounds_error=False,fill_value='extrapolate')

    decisions_1 = solve_period(1,par,hh,future=u_2)
    u_1 = interpolate.interp1d(decisions_1[0], decisions_1[1][:,0], bounds_error=False,fill_value='extrapolate')

    # Interpolate all functions
    u = [interpolate.interp1d(decisions_1[0], decisions_1[1][:,0], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_2[0], decisions_2[1][:,0], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_3[0], decisions_3[1][:,0], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_4[0], decisions_4[1][:,0], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_5[0], decisions_5[1][:,0], bounds_error=False,fill_value=None)]
    
    c = [interpolate.interp1d(decisions_1[0], decisions_1[1][:,1], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_2[0], decisions_2[1][:,1], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_3[0], decisions_3[1][:,1], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_4[0], decisions_4[1][:,1], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_5[0], decisions_5[1][:,1], bounds_error=False,fill_value=None)]

    h = [interpolate.interp1d(decisions_1[0], decisions_1[1][:,2], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_2[0], decisions_2[1][:,2], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_3[0], decisions_3[1][:,2], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_4[0], decisions_4[1][:,2], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_5[0], decisions_5[1][:,2], bounds_error=False,fill_value=None)]
    
    s = [interpolate.interp1d(decisions_1[0], decisions_1[1][:,3], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_2[0], decisions_2[1][:,3], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_3[0], decisions_3[1][:,3], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_4[0], decisions_4[1][:,3], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_5[0], decisions_5[1][:,3], bounds_error=False,fill_value=None)]
    
    wplus = [interpolate.interp1d(decisions_1[0], decisions_1[1][:,4], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_2[0], decisions_2[1][:,4], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_3[0], decisions_3[1][:,4], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_4[0], decisions_4[1][:,4], bounds_error=False,fill_value=None),
         interpolate.interp1d(decisions_5[0], decisions_5[1][:,4], bounds_error=False,fill_value=None)]
    
    own = [interpolate.interp1d(decisions_1[0], decisions_1[2][:,0], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_2[0], decisions_2[2][:,0], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_3[0], decisions_3[2][:,0], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_4[0], decisions_4[2][:,0], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_5[0], decisions_5[2][:,0], bounds_error=False,fill_value=None,kind='nearest')]
    
    LTV = [interpolate.interp1d(decisions_1[0], decisions_1[2][:,1], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_2[0], decisions_2[2][:,1], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_3[0], decisions_3[2][:,1], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_4[0], decisions_4[2][:,1], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_5[0], decisions_5[2][:,1], bounds_error=False,fill_value=None,kind='nearest')]

    LTI = [interpolate.interp1d(decisions_1[0], decisions_1[2][:,2], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_2[0], decisions_2[2][:,2], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_3[0], decisions_3[2][:,2], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_4[0], decisions_4[2][:,2], bounds_error=False,fill_value=None,kind='nearest'),
         interpolate.interp1d(decisions_5[0], decisions_5[2][:,2], bounds_error=False,fill_value=None,kind='nearest')]
    
    return u,c,h,s,wplus,own,LTV,LTI

def create_paths(w0,u,c,h,s,wplus,own,LTV,LTI):
    
    # Determine first path for the state variable, wealth, then iterate forward along the wplus function
    
    wealth_path = [w0]                                  # Set period one wealth as initial, exogenous wealth
    wealth_path.append(wplus[0]([wealth_path[-1]])[0])  # Then, for each period, append end-of-period-wealth as the next beginning-of-period wealth, 
    wealth_path.append(wplus[1]([wealth_path[-1]])[0])  # given beginning-of-period-t-wealth (which is the latest entry in the list, hence wealth_path[-1])
    wealth_path.append(wplus[2]([wealth_path[-1]])[0])  # This gives the five beginning-of-period wealth levels
    wealth_path.append(wplus[3]([wealth_path[-1]])[0])
    
    # Then, all other paths
    u_path = [u[x](wealth_path[x])[()] for x in range(len(wealth_path))] # For each beginnig-of-period-wealth, calculate choices from interpolated functions 
    c_path = [c[x](wealth_path[x])[()] for x in range(len(wealth_path))]
    h_path = [h[x](wealth_path[x])[()] for x in range(len(wealth_path))]
    s_path = [s[x](wealth_path[x])[()] for x in range(len(wealth_path))]
    own_path = [own[x](wealth_path[x])[()] for x in range(len(wealth_path))]
    LTV_path = [LTV[x](wealth_path[x])[()] for x in range(len(wealth_path))]
    LTI_path = [LTI[x](wealth_path[x])[()] for x in range(len(wealth_path))]
    
    keys = ["wealth_path","u_path","c_path","h_path","s_path","own_path","LTV_path","LTI_path"]
    elements = [wealth_path,u_path,c_path,h_path,s_path,own_path,LTV_path,LTI_path]
    
    output = SimpleNamespace(**dict(zip(keys,elements)))
    
    return output


### PART 1: Define the problem

def utility(c,h,alpha,eta,theta=1):
    return (((c**(alpha)*(theta*h)**(1-alpha))**(1-eta)-1)/(1-eta))

def bequest(w_plus,eta,chi):
    return (chi*(((w_plus)**(1-eta)-1)/(1-eta)))

# The below are value functions reduced in dimensionality using first order conditions
# (for renter and the owner's inner solution) or knowledge about net deposits (for LTI, LTV)

# CASE A - Renter
def value_function_rent(e,omega,t,par,hh,future=0):
     
    c = e*hh.alpha
    h = e*(1-hh.alpha)/par.rent
    u = e*(hh.alpha**hh.alpha*(1-hh.alpha)**(1-hh.alpha)*par.rent**(hh.alpha-1))
    s = omega - e
    wplus = s*(1+par.rho)
    
    if t == 5: 
        return ((u**(1-hh.eta)-1)/(1-hh.eta) + hh.chi*(((wplus)**(1-hh.eta)-1)/(1-hh.eta))),c,h,s,wplus
    else:
        return ((u**(1-hh.eta)-1)/(1-hh.eta) + hh.beta*future(wplus)),c,h,s,wplus

# CASE B1a and B1b - inner solution, no binding credit constraint 
# (rho is interest rate for deposits, r is interest rate for credit)
def value_function_own_inner(c,omega,t,par,hh,future=0,use_rho=False):
    
    r = par.r if use_rho == False else par.rho
    
    h = c/(hh.alpha/(1-hh.alpha)*((1+r)*(par.p+par.uc_sqm+par.p*par.uc_value)-par.p)/(1+r))
    
    s = omega - c - h*(par.uc_sqm+par.p*(1+par.uc_value)) # period savings
    wplus = s*(1+r)+par.p*h
    
    if t == 5: return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + bequest((wplus),hh.eta,hh.chi)),c,h,s,wplus
    else: return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + hh.beta*future(wplus)),c,h,s,wplus

# CASE B2 - Binding LTI Constraint
def value_function_own_LTI(c,omega,t,par,hh,future=0):
    
    y = hh.wage_path[t-1]
    h = (omega+par.LTI*y-c)/(par.uc_sqm+par.p*(1+par.uc_value))
    s = omega - c - h*(par.uc_sqm+par.p*(1+par.uc_value))
    wplus = s*(1+par.r)+par.p*h
    
    if t == 5: return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + bequest((wplus),hh.eta,hh.chi)),c,h,s,wplus
    else: return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + hh.beta*future(wplus)),c,h,s,wplus
    
# CASE B3 - Binding LTV Constraint   
def value_function_own_LTV(c,omega,t,par,hh,future=0):
    
    h = (omega-c)/(par.uc_sqm+par.p*(1-par.LTV+par.uc_value))
    s = omega - c - h*(par.uc_sqm+par.p*(1+par.uc_value))
    wplus = s*(1+par.r)+par.p*h
    
    if t == 5: return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + bequest((wplus),hh.eta,hh.chi)),c,h,s,wplus
    else: return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + hh.beta*future(wplus)),c,h,s,wplus

# Fallback case - two dimensional optimization of the owner's problem
def value_function_own(choices,omega,t,par,hh,future=0):
    
    c = choices[0]
    h = choices[1]
    
    s = omega - c - par.uc_sqm*h - par.uc_value*par.p*h - par.p*h # period savings
    wplus = s*(1+par.rho)+par.p*h if s>0 else s*(1+par.r)+par.p*h # end-of-period-wealth    
    if t == 5: 
        return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + bequest((wplus),hh.eta,hh.chi)),c,h,s,wplus
    
    else:
        return (utility(c,h,hh.alpha,hh.eta,hh.theta[t-1]) + hh.beta*future(wplus)),c,h,s,wplus
    
# Constraints for two-dimensional fallback optimization
def LTI_constraint(x,omega,y,par):
    s = omega - x[0] - par.uc_sqm*x[1] - par.uc_value*par.p*x[1] - par.p*x[1] # period savings
    return y*par.LTI + s # violated if negative

def LTV_constraint(x,omega,par):
    s = omega - x[0] - par.uc_sqm*x[1] - par.uc_value*par.p*x[1] - par.p*x[1] # period savings
    return par.p*x[1]*par.LTV + s # violated if negative    

def debt_constraint_rent(x,omega,t,par):
    s = omega - x # period savings
    if t == 5:
        return s # require non-negative s in last period
    else:
        return par.phi + s # require s larger than -phi in other periods

### PART 2: Solve the problem
def solve_period(t,par,hh,future=0):
    
    # Create grids
    w_vec = np.linspace(-2,10,n_gridpoints)
    output = np.empty(shape=(n_gridpoints,5))
    status = np.empty(shape=(n_gridpoints,3))
    
    # Loop over possible wealth levels, optimize for each of these
    for i,w in enumerate(w_vec):
        
        fallback = False # dummy for defaulting to two-dimensional optimization
        LTV_binds = False # initial value. Will change below, if it indeed binds
        LTI_binds = False # See just above

        omega = w+hh.wage_path[t-1]
        rent_debt = {'type': 'ineq', 'fun': debt_constraint_rent, 'args': (omega,t,par)}
        
        # Case A: Choosing to rent
        x0_rent = 0.6*omega
        obj_rent = lambda x: -value_function_rent(x,omega,t,par,hh,future)[0]
        result_rent = optimize.minimize(obj_rent,x0_rent,method='SLSQP',constraints=[rent_debt],bounds=((1e-8,np.inf),))
        output_rent = value_function_rent(result_rent.x,omega,t,par,hh,future)
        
        # Case B1a: Neither credit constraint bind, thus use inner solution 
        # We expect negative net deposits, so use r as interest rate
        x0_own = 0.5*omega
        obj_own_inner = lambda x: -value_function_own_inner(x,omega,t,par,hh,future)[0]
        result_own_inner = optimize.minimize(obj_own_inner,x0_own,method='SLSQP',bounds=((1e-8,np.inf),)) # [SQM,LTI,LTV]
        
        # Save output for inner function    
        output_own = value_function_own_inner(result_own_inner.x,omega,t,par,hh,future)     
        
        # ensure that failed optimization revokes fallback
        if result_own_inner.success == False: fallback = True; output_own = 0,0,0,0,0
        
        # CASE B1b: Check if net deposits is indeed negative; if not, wrong interest rate was used
        # and we instead use rho as the interest rate
        if output_own[3] > 0:
            x0_own = 0.5*omega
            obj_own_inner = lambda x: -value_function_own_inner(x,omega,t,par,hh,future,use_rho=True)[0]
            result_own_inner = optimize.minimize(obj_own_inner,x0_own,method='SLSQP',bounds=((1e-8,np.inf),))
            
            # Save output for inner function    
            output_own = value_function_own_inner(result_own_inner.x,omega,t,par,hh,future)     
            
            # if this now gives us negative net deposits, wrong interest rate is again used
            # thus the reduced dimension optimization is unfruitful, and we revoke the fallback
            # (and we also revoke fallback if optimization process failed)
            if output_own[3] < 0 or result_own_inner.success == False: fallback = True; output_own = 0,0,0,0,0; #print("Period:",t,"wealth:",round(w,2),", rho/r problem. Solved by two-dimensional fallback optimization.") 
        
        # Now check if inner solution breaks any of the credit constraints
        LTV_bind = -par.LTV*output_own[2]*par.p    
        LTI_bind = -par.LTI*hh.wage_path[t-1]
        s = output_own[3]
        
        x0_own = output_own[1] # Update guess to result from inner function (the constrained result should be close)

        # Case B2: Check whether LTV binds, if yes, calculate that
        if ((not s == max(s,LTV_bind,LTI_bind)) and (LTV_bind == max(LTV_bind,LTI_bind))): 
            LTV_binds = True
            obj_own_LTV = lambda x: -value_function_own_LTV(x,omega,t,par,hh,future)[0]
            result_own_LTV = optimize.minimize(obj_own_LTV,x0_own,method='SLSQP',bounds=((1e-8,np.inf),))        
            # Update output for binding LTV  
            output_own = value_function_own_LTV(result_own_LTV.x,omega,t,par,hh,future)     
            
            # ensure that failed optimization revokes fallback
            if result_own_LTV.success == False: fallback = True; output_own = 0,0,0,0,0
        
        # Case B3: Check whether LTI binds, if yes, calculate that    
        if ((not s == max(s,LTV_bind,LTI_bind)) and (LTI_bind == max(LTV_bind,LTI_bind))):
            
            LTI_binds = True
            obj_own_LTI = lambda x: -value_function_own_LTI(x,omega,t,par,hh,future)[0]
            result_own_LTI = optimize.minimize(obj_own_LTI,x0_own,method='SLSQP',bounds=((1e-8,np.inf),))
            output_own = value_function_own_LTI(result_own_LTI.x,omega,t,par,hh,future)
                        
            # ensure that failed optimization revokes fallback
            if result_own_LTI.success == False: fallback = True; output_own = 0,0,0,0,0
            
        # FALLBACK CASE - IF SOMETHING AS GONE WRONG ABOVE OR THE RESULT OF ABOVE FAILS ANY CONSTRAINTS OR W<0, GO TO TWO-DIMENSIONAL OPTIMIZATION
        if (fallback or output_own[1] < 0 or output_own[2] < 0 or output_own[3] < min(-par.LTV*output_own[2]*par.p,-par.LTI*hh.wage_path[t-1]) or w<0):
            
            LTI = {'type': 'ineq', 'fun': LTI_constraint, 'args': (omega,hh.wage_path[t-1],par)} 
            LTV = {'type': 'ineq', 'fun': LTV_constraint, 'args': (omega,par)}
            
            obj_own = lambda x: -value_function_own(x,omega,t,par,hh,future)[0]
            x0_own = [0.2*omega,0.8*omega]
            result_own = optimize.minimize(obj_own,x0_own,method='SLSQP',constraints=[LTV,LTI])
            output_own = value_function_own(result_own.x,omega,t,par,hh,future) 
            
            # if net deposits within 1% band on either side of LTI or LTV bind, mark that this is binding
            if LTV_bind*1.01<output_own[3]<LTV_bind*0.99: LTV_binds = True
            if LTI_bind*1.01<output_own[3]<LTI_bind*0.99: LTI_binds = True
        
        # Clean up insensible solutions that arise due to numeric approximation around 0 or unsolvable problems
        if output_own[0] < -20: output_own = np.nan,np.nan,np.nan,np.nan,np.nan
        if output_rent[0] < -20: output_rent = np.nan,np.nan,np.nan,np.nan,np.nan
        if output_own[0] > 20: output_own = np.nan,np.nan,np.nan,np.nan,np.nan
        if output_rent[0] > 20: output_rent = np.nan,np.nan,np.nan,np.nan,np.nan
        if t == 5 and output_rent[3] < 0: output_rent = np.nan,np.nan,np.nan,np.nan,np.nan
        
        # Choose between renting or owning
        own = True if output_own[0]>output_rent[0] else False    
        output[i] = output_own if own else output_rent
        status[i] = own,LTV_binds,LTI_binds    
        
    return  w_vec,output,status
