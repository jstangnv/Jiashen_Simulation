import numpy as np
import sympy

class pulsemodref():
    def __init__(self, str_modfunc=None):
        
        #
        # Dictionary with string that matches to :
        # 1. the correct modulation function and 
        # 2. a default list of arguments to feed the modulation function
        #
        self.dict_master = {'RECT': [self.f_rectangle, {'amp':1}],
                            'GAUS': [self.f_gaussian, {'center':0, 'sd':1, 'maxamp':1, 'norm':True}],
                            'SINC': [self.f_sinc, {'center':0, 'peaktozero':1, 'maxamp':1, 'norm':True}],
                            'HRM2': [self.f_hermite_poly2, {'center':0, 'sd_gauss':1, 'coeff':1, 'maxamp':1}]}
        
        # check is string is given upon object initialization       
        if str_modfunc:
            self.reset_defaults(str_modfunc)
        else:
            self.modfunc = {}
            self.moddict = {}
            self.modstr = None
        
    def reset_defaults(self, str_modfunc):
        #
        # Input is a string with first 4 letters matching one of 
        # the keys in self.dict_master
        #
        key2match = str_modfunc[:4].upper()
        if key2match in self.dict_master:
            self.modfunc = self.dict_master[key2match][0]
            self.moddict = self.dict_master[key2match][1]
            self.modstr = key2match
        else:
            raise ValueError('Specified key is not assigned to a function within puslemodfuncs')
        
    def f_gaussian(self, x, center=0, sd=1, maxamp=1, norm=False):
        #
        # To output a normalized Gaussian distribution
        # Set maxamp to either: 0 or None
        #

        if not maxamp:
            maxamp = 1/(sd*((2*np.pi)**0.5))
        elif norm:
            maxamp *= 1/(sd*((2*np.pi)**0.5)) 
        return np.exp(-(1/2)*((x-center)/sd)**2)*maxamp

    def f_sinc(self, x, center=0, peaktozero=1, maxamp=1, norm=True):
        #
        # To output a normalized distribution
        # Set maxamp to either: 0 or None
        #
        if not maxamp:
            maxamp = 1
        elif norm:
            maxamp /= peaktozero
        return np.sinc((x-center)/peaktozero)*maxamp #note np.sinc(x) = sin(pi*x)/(pi*x)

    
    def f_hermite_poly2(self, x, center=0, sd_gauss=1, coeff=1, maxamp=1):
        #
        # To output a normalized distribution
        # Set maxamp to either: 0 or None
        #
        # Note: For now, code is only ready for polynomials up to order 2!
        #
        
        expr1 = np.exp(-(((x-center)/sd_gauss)**2)/2)*(1-coeff*(((x-center)/sd_gauss)**2)/2)
        normfactor = -((np.pi/2)**0.5)*sd_gauss*(coeff-2)
        if not maxamp:
            maxamp = 1
                
        return expr1*maxamp/normfactor
    
    
    def f_rectangle(self, x, amp):
        return amp*np.ones(np.asarray(x).size)
    

    
class chirpfuncs():
    def __init__(self, str_modfunc=None):
        
        #
        # Dictionary with string that matches to :
        # 1. the correct modulation function and 
        # 2. a default list of arguments to feed the modulation function
        #
        self.dict_sympy = {}
        self.dict_sympy['RAMP'] = [self.f_ramp, {'slope':1, 'intercept':0}]
        
        self.dict_master = {}
        for key in self.dict_sympy:
            modfunc, dict_args = self.dict_sympy[key]
            self.dict_master[key] = {'expr': sympy.integrate(modfunc(dict_args), sympy.symbols('t'))}
        
        self.modfunc = self.eval_expr
        
        # check is string is given upon object initialization       
        if str_modfunc:
            self.reset_defaults(str_modfunc)
        else:
            self.moddict = {}
            self.modfunc_args = None
            self.modfunc_sympy = None
        
    def reset_defaults(self, str_modfunc):
        #
        # Input is a string with first 4 letters matching one of 
        # the keys in self.dict_master
        #
        key2match = str_modfunc[:4].upper()
        if key2match in self.dict_master:
            self.modstr = key2match
            self.modfunc_sympy = self.dict_sympy[key2match][0]
            self.modfunc_args = self.dict_sympy[key2match][1]
            self.moddict = self.dict_master[key2match]
        else:
            raise ValueError('Specified key is not assigned to a function within pulsemodfuncs')
    
    def eval_expr(self, expr, tval):
        return expr.subs(sympy.symbols('t'), tval)
    
    def f_ramp(self, dict_args):
        slope = dict_args['slope']
        intercept = dict_args['intercept']
        return slope*sympy.symbols('t')+intercept    
  
    

