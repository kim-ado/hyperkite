import torch
import numpy as np
import warnings
from scipy.ndimage import uniform_filter
eps = 1e-10

#Cross-correlation matrix
def cross_correlation(H_fuse, H_ref):
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating mean value
    mean_fuse = torch.mean(H_fuse_reshaped, 1).unsqueeze(1)
    mean_ref = torch.mean(H_ref_reshaped, 1).unsqueeze(1)

    CC = torch.sum((H_fuse_reshaped- mean_fuse)*(H_ref_reshaped-mean_ref), 1)/torch.sqrt(torch.sum((H_fuse_reshaped- mean_fuse)**2, 1)*torch.sum((H_ref_reshaped-mean_ref)**2, 1))

    CC = torch.mean(CC)
    return CC

# Spectral-Angle-Mapper (SAM)
def SAM(H_fuse, H_ref):
    #Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped  = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]
    
    # Calculating inner product
    inner_prod  = torch.nansum(H_fuse_reshaped*H_ref_reshaped, 0)
    fuse_norm   = torch.nansum(H_fuse_reshaped**2, dim=0).sqrt()
    ref_norm    = torch.nansum(H_ref_reshaped**2, dim=0).sqrt()

    # Calculating SAM
    SAM = torch.rad2deg(torch.nansum( torch.acos(inner_prod/(fuse_norm*ref_norm)) )/N_pixels)
    return SAM

# Root-Mean-Squared Error (RMSE)
def RMSE(H_fuse, H_ref):
    # Rehsaping fused and reference data
    H_fuse_reshaped = H_fuse.view(-1)
    H_ref_reshaped = H_ref.view(-1)

    # Calculating RMSE
    RMSE = torch.sqrt(torch.nansum((H_ref_reshaped - H_fuse_reshaped)**2)/H_fuse_reshaped.shape[0])
    return RMSE

# Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
def ERGAS(H_fuse, H_ref, beta):
    
    #Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)
    N_pixels = H_fuse_reshaped.shape[1]

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.nansum((H_ref_reshaped-H_fuse_reshaped)**2, dim=1)/N_pixels)
    mu_ref = torch.mean(H_ref_reshaped, dim=1)

    # Calculating Erreur Relative Globale Adimensionnelle De Synthèse (ERGAS)
    ERGAS = 100*(1/beta**2)*torch.sqrt(torch.nansum(torch.div(rmse, mu_ref)**2)/N_spectral)
    return ERGAS


# Peak SNR (PSNR)
def PSNR(H_fuse, H_ref):
    #Compute number of spectral bands
    N_spectral = H_fuse.shape[1]

    # Reshaping images
    H_fuse_reshaped = H_fuse.view(N_spectral, -1)
    H_ref_reshaped = H_ref.view(N_spectral, -1)

    # Calculating RMSE of each band
    rmse = torch.sqrt(torch.sum((H_ref_reshaped-H_fuse_reshaped)**2, dim=1)/H_fuse_reshaped.shape[1])

    # Calculating max of H_ref for each band
    max_H_ref, _ = torch.max(H_ref_reshaped, dim=1)

    # Calculating PSNR
    PSNR = torch.nansum(10*torch.log10(torch.div(max_H_ref, rmse)**2))/N_spectral

    return PSNR

def _uqi_single(GT,P,ws):
	N = ws**2
	window = np.ones((ws,ws))

	GT_sq = GT*GT
	P_sq = P*P
	GT_P = GT*P

	GT_sum = uniform_filter(GT, ws)    
	P_sum =  uniform_filter(P, ws)     
	GT_sq_sum = uniform_filter(GT_sq, ws)  
	P_sq_sum = uniform_filter(P_sq, ws)  
	GT_P_sum = uniform_filter(GT_P, ws)

	GT_P_sum_mul = GT_sum*P_sum
	GT_P_sum_sq_sum_mul = GT_sum*GT_sum + P_sum*P_sum
	numerator = 4*(N*GT_P_sum - GT_P_sum_mul)*GT_P_sum_mul
	denominator1 = N*(GT_sq_sum + P_sq_sum) - GT_P_sum_sq_sum_mul
	denominator = denominator1*GT_P_sum_sq_sum_mul

	q_map = np.ones(denominator.shape)
	index = np.logical_and((denominator1 == 0) , (GT_P_sum_sq_sum_mul != 0))
	q_map[index] = 2*GT_P_sum_mul[index]/GT_P_sum_sq_sum_mul[index]
	index = (denominator != 0)
	q_map[index] = numerator[index]/denominator[index]

	s = int(np.round(ws/2))
	return np.mean(q_map[s:-s,s:-s])

def uqi (GT,P,ws=2):
	"""calculates universal image quality index (uqi).

	:param GT: first (original) input image.
	:param P: second (deformed) input image.
	:param ws: sliding window size (default = 8).

	:returns:  float -- uqi value.
	"""
	GT,P = _initial_check(GT,P)
	return np.mean([_uqi_single(GT[:,:,i],P[:,:,i],ws) for i in range(GT.shape[2])])

def _initial_check(GT,P):
	assert GT.shape == P.shape, "Supplied images have different sizes " + \
	str(GT.shape) + " and " + str(P.shape)
	if GT.dtype != P.dtype:
		msg = "Supplied images have different dtypes " + \
			str(GT.dtype) + " and " + str(P.dtype)
		warnings.warn(msg)
	

	if len(GT.shape) == 2:
		GT = GT[:,:,np.newaxis]
		P = P[:,:,np.newaxis]

	return GT,P

def d_lambda (ms,fused,p=1):
	"""calculates Spectral Distortion Index (D_lambda).

	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param p: parameter to emphasize large spectral differences (default = 1).

	:returns:  float -- D_lambda.
	"""
	L = ms.shape[2]

	M1 = np.ones((L,L))
	M2 = np.ones((L,L))

	for l in range(L):
		for r in range(l+1,L):
			M1[l,r] = M1[r,l] = uqi(fused[:,:,l],fused[:,:,r])
			M2[l,r] = M2[r,l] = uqi(ms[:,:,l],ms[:,:,r])

	diff = np.abs(M1 - M2)**p
	return (1./(L*(L-1)) * np.sum(diff))**(1./p)

def d_s_r(I_F, I_PAN, S=32):
    _, _, CD, _, _ = lsr(I_F, I_PAN, S)
    
    Ds_R_index = 1 - CD
    
    return Ds_R_index

def lsr(I_F, I_PAN, S=32):
    # Vectorization of PAN and fused MS
    IHc = I_PAN.flatten()
    ILRc = I_F.reshape(-1, I_F.shape[2])

    # Multivariate linear regression
    w, _, _, _ = np.linalg.lstsq(ILRc, IHc, rcond=None)
    alpha = np.expand_dims(np.expand_dims(w, axis=0), axis=0)

    # Fitted Least squares intensity
    I_R = np.sum(I_F * np.tile(alpha, (I_F.shape[0], I_F.shape[1], 1)), axis=2)

    # Space-varying least squares error
    err_reg = I_PAN.flatten() - I_R.flatten()

    # Regression residual value
    res = np.sqrt(np.mean(err_reg**2))

    # Coefficient of determination
    cd = 1 - (np.var(err_reg) / np.var(I_PAN.flatten()))

    # Variance of PAN
    var_pan = np.var(I_PAN.flatten())

    return w, res, cd, var_pan, err_reg


def qnr (pan,ms,fused,alpha=1,beta=1,p=1,q=1):
	"""calculates Quality with No Reference (QNR).

	:param pan: high resolution panchromatic image.
	:param ms: low resolution multispectral image.
	:param fused: high resolution fused image.
	:param alpha: emphasizes relevance of spectral distortions to the overall.
	:param beta: emphasizes relevance of spatial distortions to the overall.
	:param p: parameter to emphasize large spectral differences (default = 1).
	:param q: parameter to emphasize large spatial differences (default = 1).
	:param r: ratio of high resolution to low resolution (default=4).
	:param ws: sliding window size (default = 7).

	:returns:  float -- QNR.
	"""

	lam = d_lambda(ms,fused,p=p)
	a = (1-lam)**alpha
	dsr = d_s_r(fused, pan)
	b = (1-dsr)**beta
	return a*b, lam, dsr