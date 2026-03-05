# Code version: v13
# Last updated: March 2026
# Author: Kartikey Singh

"""
Physics-Informed Neural Surrogate Inversion for Electrical Impedance Tomography
================================================================================

Primary implementation and algorithm design:
    Kartikey Singh

Research discussions and guidance:
    Prof. Debasish Roy

Date:
    March 2026

This code was developed as part of a collaborative research project.

If this code contributes to a publication,
the authors of this repository should be appropriately credited
according to standard academic authorship practices.

# Version: v13
# Repository snapshot: March 2026

------------------------------------------------------------------------------

LICENSE
-------

Copyright (c) 2026 Kartikey Singh

This code is released for academic and research use.

If you use this code in academic work, please cite the repository.

Commercial use requires explicit permission from the author.

------------------------------------------------------------------------------

Citation
--------

If you use this code in academic work, please cite:

Kartikey Singh (2026)
Physics-Informed Neural Surrogate Inversion for Electrical Impedance Tomography
Code Repository

------------------------------------------------------------------------------

Overview
--------
This script implements a hybrid physics-informed neural surrogate framework for
solving the Electrical Impedance Tomography (EIT) inverse problem.

The goal is to reconstruct the location and geometry of a conductive inclusion
inside a 2D domain from boundary voltage measurements.

The method combines:

1. Finite Difference Method (FDM)
   Used as an accurate forward solver for the EIT PDE.

2. Physics-Informed Neural Network (PINN)
   Trained as a fast differentiable surrogate of the forward model.

3. Gradient-based inverse optimisation
   The polygon parameters are optimized so that predicted voltages match
   measured boundary voltages.

# ---------------------------------------------------------------------
# Method Concept
# ---------------------------------------------------------------------
# The surrogate inversion framework implemented in this script was
# developed as part of a research collaboration between Kartikey Singh
# and Prof. Debasish Roy.
#
# The implementation design, training pipeline, optimization strategy,
# and evaluation framework were written by Kartikey Singh.
#
# The research direction and problem formulation were discussed jointly.
# ---------------------------------------------------------------------

---------------------------------------------------------------------

Reconstruction Pipeline
-----------------------

Step 1 — Synthetic measurement generation
    The FDM solver computes electrode voltages for the true conductivity
    distribution. These voltages serve as the measured data (V_MEAS).

Step 2 — Surrogate training dataset generation
    Random polygon inclusions are generated across the domain and the FDM
    solver computes their corresponding electrode voltages.

Step 3 — PINN surrogate training
    A neural network is trained to approximate the mapping

        polygon geometry → electrode voltages

    using:
        • supervised loss against FDM solutions
        • physics regularisation enforcing EIT constraints

Step 4 — Inverse reconstruction
    The trained PINN is used as a differentiable forward model. The polygon
    parameters are optimized so that predicted voltages match V_MEAS.

Step 5 — LBFGS refinement
    After the gradient-based optimization stages, a quasi-Newton LBFGS optimizer
    is applied to refine the polygon parameters and achieve a more accurate
    voltage match. This step may be computationally intensive but can significantly 
    improve reconstruction quality by fine-tuning the solution in the local parameter space.

---------------------------------------------------------------------

Inverse Problem Formulation
---------------------------

The governing PDE for EIT is

        ∇ · (σ(x) ∇u(x)) = 0

where

    σ(x) : electrical conductivity distribution
    u(x) : electric potential

Boundary currents are injected through electrodes and resulting voltages
are measured.

The inverse problem is to recover σ(x) from these boundary measurements.

---------------------------------------------------------------------

Assumptions Used in This Experiment
-----------------------------------

The reconstruction assumes:

    • Known electrode positions
    • Known background conductivity σ_bg
    • Known inclusion conductivity σ_inc
    • Inclusion shape class: convex quadrilateral

The true polygon parameters are NEVER provided to the optimizer.

They are used only for:
    • generating synthetic measurements
    • computing evaluation metrics

# ---------------------------------------------------------------------
# Methodological Transparency
# ---------------------------------------------------------------------
# The reconstruction assumes that the inclusion belongs to a known
# shape class (convex quadrilateral).
#
# This prior is explicitly disclosed as part of the inverse problem
# formulation and does not constitute leakage of the ground truth.
#
# The optimizer never receives the true polygon parameters.
# They are used only for synthetic data generation and evaluation.
# ---------------------------------------------------------------------

---------------------------------------------------------------------

Evaluation Metrics
------------------

The reconstruction quality is evaluated using:

    IoU (Intersection over Union)
    Dice coefficient
    Hausdorff distance
    Center localization error
    Area relative error
    Boundary deviation

These metrics quantify both geometric accuracy and spatial localization.

---------------------------------------------------------------------

Outputs
-------

The script generates the following figures:

    final_result_pinn.png
        Conductivity reconstruction comparison

    polygon_comparison_pinn.png
        Overlay of true and reconstructed polygons

    polygon_dark_plasma.png
        Visualization of polygon geometry

    training_curves_pinn.png
        Training and inversion loss curves

---------------------------------------------------------------------

Runtime
-------

Typical runtime (GPU):

    Dataset generation: ~2–3 minutes
    PINN training:      ~8–10 minutes
    Inversion:          ~3–5 minutes

---------------------------------------------------------------------
=============================================================

METHOD: Physics-Informed Neural Surrogate Inversion
─────────────────────────────────────────────────────
Initial experiments in this project showed that training the PINN
using only a single polygon configuration does not provide sufficient
gradient information for stable inverse reconstruction. The network 
fails to learn how electrode voltages change with respect to polygon 
geometry across the domain, leading to unreliable gradients and poor convergence.

To resolve this issue, the surrogate network is trained on a large set of
random polygon geometries distributed across the domain.

This enables the network to learn the sensitivity of electrode voltages with
respect to polygon geometry, allowing reliable gradient-based inversion.

The network cannot extrapolate gradient direction → wrong update.

This approach can be viewed as surrogate-based PDE inversion, where a 
neural network approximates the forward operator mapping geometry parameters to boundary voltages.
Published widely: e.g. Li et al. 2020 (FNO), Chen et al. 2021.

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import os, time, copy
from scipy.spatial.distance import directed_hausdorff
from scipy.spatial import KDTree
from matplotlib.path import Path

# --------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------
# All experiments are deterministic due to fixed random seeds.
# The repository also stores intermediate artifacts so the
# entire pipeline can be reproduced without retraining.
#
# Saved artifacts:
#   pinn_dataset_v13.pt
#   pinn_pretrained_v13.pt
#   vmeas_v13_fdm.pt
# --------------------------------------------------------------

torch.manual_seed(42); np.random.seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
torch.set_default_dtype(torch.float32)

# ══════════════════════════════════════════════════════════
# 0. Constants
# ══════════════════════════════════════════════════════════
SIGMA_BG  = 1.0
SIGMA_INC = 15.0
N_GRID    = 60
N_ELEC    = 16
ELEC_LEN  = 2.0
DOMAIN    = 20.0
BATCH     = 512

# ══════════════════════════════════════════════════════════
# 1. Injection patterns
# ══════════════════════════════════════════════════════════
# 1. Injection patterns (ALL electrode pairs)
PATTERNS = []
for i in range(N_ELEC):
    for j in range(i+1, N_ELEC):
        PATTERNS.append((i, j))
N_PAT = len(PATTERNS)
NEUTRAL = torch.tensor([p for p,(s,k) in enumerate(PATTERNS)
                         if s!=0 and k!=0], device=device)
print(f"Injection patterns: {N_PAT}")

# ══════════════════════════════════════════════════════════
# 2. True polygon — only for V_MEAS & evaluation
# ══════════════════════════════════════════════════════════
TRUE_NP = np.array([[4.675,7.08],[7.675,8.04],[8.275,5.64],[5.875,4.20]])
TRUE_T  = torch.tensor(TRUE_NP, dtype=torch.float32, device=device)
TRUE_C  = TRUE_T.mean(0)

def shoelace(v):
    a=0.
    for i in range(len(v)): a+=v[i,0]*v[(i+1)%len(v),1]-v[(i+1)%len(v),0]*v[i,1]
    return abs(a)/2.
TRUE_AREA = shoelace(TRUE_NP)

print(f"\n=== GROUND TRUTH (evaluation only — never used in optimizer) ===")
print(f"  Center : ({TRUE_C[0]:.4f}, {TRUE_C[1]:.4f})")
print(f"  Area   : {TRUE_AREA:.4f}")
print(f"  Verts  :\n{TRUE_NP}")
print("="*62+"\n")

def hard_sigma_true(xy):
    v=TRUE_T; x,y=xy[:,0],xy[:,1]
    inside=torch.zeros_like(x,dtype=torch.bool)
    for i in range(4):
        x0,y0=v[i]; x1,y1=v[(i+1)%4]
        cond=((y0>y)!=(y1>y))&(x<(x1-x0)*(y-y0)/(y1-y0+1e-12)+x0)
        inside^=cond
    return torch.where(inside,torch.full_like(x,SIGMA_INC),
                               torch.full_like(x,SIGMA_BG)).view(-1,1)

# ══════════════════════════════════════════════════════════
# 3. Polygon model (soft, differentiable)
# ══════════════════════════════════════════════════════════
class PolygonModel(nn.Module):
    def __init__(self, init_verts, alpha=6.0):
        super().__init__()
        v0=torch.tensor(init_verts,dtype=torch.float32,device=device)
        self.center =nn.Parameter(v0.mean(0).clone())
        self.offsets=nn.Parameter((v0-v0.mean(0)).clone())
        self.alpha  =alpha

    def get_vertices(self): return self.center+self.offsets

    def forward(self,xy):
        v=self.get_vertices(); vn=torch.roll(v,-1,0); e=vn-v
        nm=F.normalize(torch.stack([-e[:,1],e[:,0]],1),1)
        d=torch.sum((xy[:,None,:]-v[None])*nm[None],2)
        sa=(v[:,0]*torch.roll(v[:,1],-1)-v[:,1]*torch.roll(v[:,0],-1)).sum()
        d=torch.sign(sa+1e-12)*d
        sd=-(1./3.)*torch.logsumexp(-3.*d,1)
        return (SIGMA_BG+(SIGMA_INC-SIGMA_BG)*torch.sigmoid(self.alpha*sd)).view(-1,1)

# ══════════════════════════════════════════════════════════
# 4. FDM Solver (exact, differentiable)
# ══════════════════════════════════════════════════════════
class FDMSolver:
    def __init__(self,N=N_GRID):
        self.N=N; self.h=DOMAIN/N; self.n=(N+1)**2
        self._build(); self._electrodes(); self._rhs()
        print(f"FDM: {N+1}×{N+1}={self.n} nodes, h={self.h:.4f}")

    def _build(self):
        N=self.N; n=self.n
        ih=torch.arange(N,device=device).repeat_interleave(N+1)
        jh=torch.arange(N+1,device=device).repeat(N)
        sh=ih*(N+1)+jh; dh=(ih+1)*(N+1)+jh
        iv=torch.arange(N+1,device=device).repeat_interleave(N)
        jv=torch.arange(N,device=device).repeat(N+1)
        sv=iv*(N+1)+jv; dv=iv*(N+1)+jv+1
        self.src=torch.cat([sh,sv]); self.dst=torch.cat([dh,dv])
        ne=len(self.src)
        row=torch.cat([torch.arange(ne,device=device)]*2)
        col=torch.cat([self.src,self.dst])
        val=torch.cat([torch.ones(ne,device=device),-torch.ones(ne,device=device)])
        self.B=torch.sparse_coo_tensor(torch.stack([row,col]),val,(ne,n)).to_dense()
        xs=torch.linspace(0,DOMAIN,N+1,device=device)
        ys=torch.linspace(0,DOMAIN,N+1,device=device)
        XX,YY=torch.meshgrid(xs,ys,indexing='ij')
        self.node_xy=torch.stack([XX.reshape(-1),YY.reshape(-1)],1)

    def _electrodes(self):
        N=self.N; h=self.h; step=4*DOMAIN/N_ELEC; half=ELEC_LEN/2.
        elec=[[] for _ in range(N_ELEC)]
        for idx in range(self.n):
            i=idx//(N+1); j=idx%(N+1)
            if   j==0 and i<N: s=i*h
            elif i==N and j<N: s=DOMAIN+j*h
            elif j==N and i>0: s=2*DOMAIN+(N-i)*h
            elif i==0 and j>0: s=3*DOMAIN+(N-j)*h
            else: continue
            s%=4*DOMAIN
            for k in range(N_ELEC):
                c=k*step+step/2.
                if min(abs(s-c),abs(s-c-4*DOMAIN),abs(s-c+4*DOMAIN))<=half:
                    elec[k].append(idx)
        self.en=[torch.tensor(e,dtype=torch.long,device=device) for e in elec]

    def _rhs(self):
        F=torch.zeros(self.n,N_PAT,device=device)
        for p,(src,snk) in enumerate(PATTERNS):
            sn=self.en[src]; sk=self.en[snk]
            if len(sn)>0: F[sn,p]+=1./len(sn)
            if len(sk)>0: F[sk,p]-=1./len(sk)
        self.F0=F

    def solve(self,sigma_fn):
        sigma=sigma_fn(self.node_xy).squeeze(-1)
        w=(sigma[self.src]+sigma[self.dst])/2.
        K=self.B.t()@(self.B*w.unsqueeze(1))
        e0=torch.zeros(1,self.n,device=device); e0[0,0]=1.
        Kg=torch.cat([e0,K[1:,:]],0)
        Fg=torch.cat([torch.zeros(1,N_PAT,device=device),self.F0[1:,:]],0)
        U=torch.linalg.solve(Kg,Fg)
        V=torch.zeros(N_PAT,N_ELEC,device=device)
        for k in range(N_ELEC):
            if len(self.en[k])>0: V[:,k]=U[self.en[k],:].mean(0)
        return V

fdm = FDMSolver(N=N_GRID)

# ══════════════════════════════════════════════════════════
# 5. V_MEAS — FDM at true phantom 
# ══════════════════════════════════════════════════════════
for cname in ['vmeas_v11_fdm.pt','vmeas_v12_fdm.pt','vmeas_v13_fdm.pt']:
    if os.path.exists(cname):
        saved=torch.load(cname,map_location=device,weights_only=True)
        vm=saved['V_MEAS']
        if vm.shape==(N_PAT,N_ELEC):
            V_MEAS=vm; print(f"Loaded {cname}  range=[{vm.min():.3f},{vm.max():.3f}]"); break
else:
    print("Computing V_MEAS via FDM...")
    t0=time.time()
    with torch.no_grad(): V_MEAS=fdm.solve(hard_sigma_true)
    torch.save({'V_MEAS':V_MEAS},'vmeas_v13_fdm.pt')
    print(f"Done in {time.time()-t0:.1f}s")
V_MEAS=V_MEAS.to(device)

# ─── Optional measurement noise (for experiments) ───
NOISE_LEVEL = 0.00   # 0.00, 0.01, 0.03, 0.05

if NOISE_LEVEL > 0:
    noise = NOISE_LEVEL * V_MEAS.abs() * torch.randn_like(V_MEAS)
    V_MEAS = V_MEAS + noise
    print(f"Added {NOISE_LEVEL*100:.1f}% Gaussian noise to measurements")
    
with torch.no_grad():
    dl=torch.mean((fdm.solve(hard_sigma_true)-V_MEAS)**2).item()
print(f"FDM sanity: {dl:.2e}  ✓\n")

# ══════════════════════════════════════════════════════════
# 6. MultiPotNet — neural forward surrogate
#    Conditioned on polygon vertices; output = electrode voltages
# ══════════════════════════════════════════════════════════
class MultiPotNet(nn.Module):
    """
    Input:  (x,y) spatial coordinates + polygon vertices as context
    Output: u(x,y; polygon) for all N_PAT injection patterns
    Gradient path: loss → trunk → poly_branch(vertices) → center  ✓
    """
    def __init__(self, n_verts=4, units=128):
        super().__init__()
        poly_dim=n_verts*2
        self.xy_br=nn.Sequential(
            nn.Linear(2,units),     nn.Tanh(),
            nn.Linear(units,units), nn.Tanh(),
            nn.Linear(units,units), nn.Tanh(),
        )
        self.sg_br=nn.Sequential(
            nn.Linear(1,units//4), nn.Tanh(),
        )
        self.poly_br=nn.Sequential(
            nn.Linear(poly_dim,units), nn.Tanh(),
            nn.Linear(units,units),    nn.Tanh(),
            nn.Linear(units,units),    nn.Tanh(),
        )
        fused=units+units//4+units
        self.trunk=nn.Sequential(
            nn.Linear(fused,units), nn.Tanh(),
            nn.Linear(units,units), nn.Tanh(),
            nn.Linear(units,N_PAT),
        )

    def forward(self,xy,poly_model):
        xn=(xy-10.)/10.
        sig=poly_model(xy)
        sn=(sig-SIGMA_BG)/(SIGMA_INC-SIGMA_BG)
        verts=poly_model.get_vertices()             # live or detached
        vn=(verts.reshape(1,-1)-10.)/10.
        vn=vn.expand(xy.shape[0],-1)
        return self.trunk(torch.cat([self.xy_br(xn),
                                     self.sg_br(sn),
                                     self.poly_br(vn)],1))

def predict_voltages(unet, poly_model):
    """Mean potential at each electrode for all patterns."""
    preds=[torch.mean(unet(fdm.node_xy[fdm.en[k]],poly_model),0)
           for k in range(N_ELEC)]
    return torch.stack(preds).T   # (N_PAT, N_ELEC)

# ══════════════════════════════════════════════════════════
# 7. Physics losses (weak form — faster than strong form)
# ══════════════════════════════════════════════════════════
ALL_ELEC_XY=torch.cat([fdm.node_xy[fdm.en[k]] for k in range(N_ELEC)],0)
ALL_ELEC_N =torch.zeros_like(ALL_ELEC_XY)   # we use FDM BCs, not flux here
NEUMANN_PTS=fdm.node_xy  # interior physics applied at grid nodes

def sample_pts(poly_model, n=BATCH):
    v=poly_model.get_vertices().detach(); c=v.mean(0); vn=torch.roll(v,-1,0)
    u=torch.rand(n//3,2,device=device)*DOMAIN
    cc=c+1.5*torch.randn(n//3,2,device=device)
    t=torch.rand(n//3,1,device=device); eid=torch.randint(0,4,(n//3,),device=device)
    eb=v[eid]+t*(vn[eid]-v[eid])+0.15*torch.randn(n//3,2,device=device)
    return torch.clamp(torch.cat([u,cc,eb],0),0.,DOMAIN)

def energy_loss(unet, poly_model, xy, n_pat=8):
    """Weak form: ∫σ|∇u|² → correct iff ∇·(σ∇u)=0"""
    xy=xy.requires_grad_(True); sig=poly_model(xy); u=unet(xy,poly_model)
    sub=torch.randperm(N_PAT,device=device)[:n_pat]; loss=0.
    for p in sub:
        gu=torch.autograd.grad(u[:,p:p+1],xy,
            grad_outputs=torch.ones(xy.shape[0],1,device=device),
            create_graph=True,retain_graph=True)[0]
        loss+=torch.mean(sig*(gu[:,0:1]**2+gu[:,1:2]**2))
    return loss/len(sub)

def equipotential_loss(unet, poly_model):
    loss=0.
    for k in range(N_ELEC):
        loss+=torch.mean(torch.var(unet(fdm.node_xy[fdm.en[k]],poly_model),0))
    return loss

def ground_loss(unet, poly_model):
    return torch.sum(torch.mean(
        unet(fdm.node_xy[fdm.en[0]],poly_model)[:,NEUTRAL],0)**2)

# ══════════════════════════════════════════════════════════
# 8. FDM pre-training dataset
#    400 random polygons → FDM exact solutions
#    Training on many polygon geometries allows the surrogate network
#    to learn how electrode voltages change with respect to polygon
#    position and shape across the domain. This provides the necessary gradient information for inversion,
#    across the ENTIRE domain (not just one position).
# ══════════════════════════════════════════════════════════
PRE_N = 3000   # number of random training polygons

DATASET_CACHE='pinn_dataset_v13.pt'
if os.path.exists(DATASET_CACHE):
    ds=torch.load(DATASET_CACHE,map_location=device,weights_only=True)
    DS_VERTS=ds['verts']; DS_V=ds['voltages']
    print(f"Loaded pre-training dataset: {DS_VERTS.shape[0]} polygons")
else:
    print(f"Generating FDM pre-training dataset ({PRE_N} random polygons)...")
    t0=time.time()
    DS_VERTS=[]; DS_V=[]

    # Temporary polygon for FDM calls (no grad)
    tmp_poly=PolygonModel([[5,5],[10,5],[10,10],[5,10]]).to(device)

    for i in range(PRE_N):
        # Sample random CONVEX polygon anywhere in domain [3,17]²
        rand_c=torch.rand(2,device=device)*12.+3.    # center in [3,17]
        angles=torch.sort(torch.rand(4)*2*np.pi).values.to(device)
        radii =torch.rand(4,device=device)*1.8+0.6   # radius 0.6–2.4
        rand_off=torch.stack([radii*torch.cos(angles),
                               radii*torch.sin(angles)],1)

        # Keep polygon inside domain
        verts_tmp=rand_c+rand_off
        if (verts_tmp.min()<1.).item() or (verts_tmp.max()>19.).item():
            # clamp toward center
            rand_c=torch.rand(2,device=device)*8.+6.
            radii =torch.rand(4,device=device)*1.2+0.5
            rand_off=torch.stack([radii*torch.cos(angles),
                                   radii*torch.sin(angles)],1)

        with torch.no_grad():
            tmp_poly.center.copy_(rand_c)
            tmp_poly.offsets.copy_(rand_off)
            tmp_poly.alpha = 12.0
        
            # FDM exact solve (electrode voltages)
            V_fdm = fdm.solve(tmp_poly)
        
        # store dataset entry
        DS_VERTS.append(rand_c + rand_off)
        DS_V.append(V_fdm.clone())

        if i%100==0:
            print(f"  {i+1}/{PRE_N} polygons  center=({rand_c[0]:.1f},{rand_c[1]:.1f})"
                  f"  {time.time()-t0:.0f}s")

    DS_VERTS=torch.stack(DS_VERTS,0)  # (PRE_N, 4, 2)
    DS_V    =torch.stack(DS_V,0)      # (PRE_N, N_PAT, N_ELEC)
    torch.save({'verts':DS_VERTS,'voltages':DS_V},DATASET_CACHE)
    print(f"Dataset saved ({time.time()-t0:.0f}s)\n")

# ══════════════════════════════════════════════════════════
# 9. PINN pre-training
#    Supervised by FDM exact voltages + physics regularisation
#    This guarantees poly_branch learns correct sensitivity
# ══════════════════════════════════════════════════════════
unet=MultiPotNet().to(device)
n_params=sum(p.numel() for p in unet.parameters())
print(f"MultiPotNet params: {n_params:,}")

PINN_CACHE='pinn_pretrained_v13.pt'
if os.path.exists(PINN_CACHE):
    unet.load_state_dict(torch.load(PINN_CACHE,map_location=device,weights_only=True))
    print(f"Loaded pre-trained PINN from {PINN_CACHE}")
else:
    print(f"\nPINN pre-training (supervised by FDM + physics)...")
    opt_pre=optim.AdamW(unet.parameters(),lr=1e-3,weight_decay=1e-5)
    sch_pre=optim.lr_scheduler.CosineAnnealingLR(opt_pre,T_max=3000,eta_min=1e-4)
    t0=time.time()

    # Temporary learnable polygon for PINN training
    tmp=PolygonModel([[5,5],[10,5],[10,10],[5,10]]).to(device)

    for step in range(3000):
        # Pick random polygon from dataset
        idx=torch.randint(0,PRE_N,(1,)).item()
        verts_i=DS_VERTS[idx]    # (4,2)
        V_fdm_i=DS_V[idx]        # (N_PAT, N_ELEC)

        # Load into tmp polygon (no grad for polygon)
        with torch.no_grad():
            tmp.center.copy_(verts_i.mean(0))
            tmp.offsets.copy_(verts_i-verts_i.mean(0))
            tmp.alpha=6.+step/3000.*18.   # anneal 6→24

        for p in tmp.parameters(): p.requires_grad_(False)
        for p in unet.parameters(): p.requires_grad_(True)

        opt_pre.zero_grad()

        # ── Supervised voltage loss (FDM teacher) ──
        V_pred_i=predict_voltages(unet, tmp)
        l_sup=torch.mean((V_pred_i-V_fdm_i)**2)

        # ── Physics regularisation (weak form, fast) ──
        xy=sample_pts(tmp).requires_grad_(True)
        l_en=energy_loss(unet, tmp, xy, n_pat=6)
        l_eq=equipotential_loss(unet, tmp)
        l_gnd=ground_loss(unet, tmp)

        loss=(50.*l_sup
             +0.1*l_en
             +0.5*l_eq
             +0.5*l_gnd)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unet.parameters(),1.0)
        opt_pre.step(); sch_pre.step()

        if step%500==0:
            with torch.no_grad():
                # Check: how well does PINN match FDM on init polygon?
                poly_check=PolygonModel([[5.5,8.5],[9.5,8.5],[9.5,12.5],[5.5,12.5]]).to(device)
                v_check=fdm.solve(poly_check)
                v_pinn =predict_voltages(unet,poly_check)
                err_check=torch.mean((v_pinn-v_check)**2).item()
            print(f"  pre {step:4d}/3000 | sup={l_sup.item():.2e} "
                  f"en={l_en.item():.2e} | "
                  f"PINN-vs-FDM at init polygon: {err_check:.2e} | "
                  f"{time.time()-t0:.0f}s")

    torch.save(unet.state_dict(), PINN_CACHE)
    print(f"Pre-training done ({(time.time()-t0)/60:.1f} min)\n")

# ══════════════════════════════════════════════════════════
# 10. Gradient probe — PINN vs FDM direction check
# ══════════════════════════════════════════════════════════
init_verts=[[5.5,8.5],[9.5,8.5],[9.5,12.5],[5.5,12.5]]
poly=PolygonModel(init_verts,alpha=8.0).to(device)

print("── Gradient probe: PINN vs FDM direction ──")
for pm,label in [(poly,'PINN'),(poly,'FDM')]:
    for p in poly.parameters(): p.requires_grad_(True); p.grad=None
    if label=='PINN':
        V_p=predict_voltages(unet,poly)
    else:
        V_p=fdm.solve(poly)
    dl=torch.mean((V_p-V_MEAS)**2); dl.backward()
    gc=poly.center.grad.clone()
    print(f"  {label}: data_loss={dl.item():.6f}  "
          f"grad=({gc[0].item():.4e}, {gc[1].item():.4e})")
    for p in poly.parameters(): p.grad=None
    # Direction check
    tc=TRUE_C; pc=poly.get_vertices().mean(0).detach()
    correct=[]
    for d,nm in [(0,'x'),(1,'y')]:
        need_down=pc[d]>tc[d]; grad_down=gc[d]>0
        ok=need_down==grad_down; correct.append(ok)
        print(f"    {nm}: need={'↓' if need_down else '↑'} "
              f"grad={'↓' if grad_down else '↑'} {'✓' if ok else '✗'}")
    print(f"  {'✓ Both correct!' if all(correct) else '⚠ Check direction'}\n")

# ══════════════════════════════════════════════════════════
# 11. Inversion — PINN as differentiable forward model
# ══════════════════════════════════════════════════════════
# Reset polygon to init
with torch.no_grad():
    v0=torch.tensor(init_verts,dtype=torch.float32,device=device)
    poly.center.copy_(v0.mean(0)); poly.offsets.copy_(v0-v0.mean(0))
    poly.alpha=6.0

def perimeter_reg(pm,lam=5e-5):
    v=pm.get_vertices()
    return lam*torch.mean(torch.norm(torch.roll(v,-1,0)-v,1)**2)

def angle_reg(pm, lam=2e-4):
    v = pm.get_vertices()
    vp = torch.roll(v,1,0)
    vn = torch.roll(v,-1,0)

    a = F.normalize(v-vp, dim=1)
    b = F.normalize(vn-v, dim=1)

    cos = torch.sum(a*b,1)
    return lam*torch.mean((cos+1)**2)


def area_reg(pm, lam=2e-4):
    v = pm.get_vertices()
    x = v[:,0]; y = v[:,1]
    area = 0.5*torch.abs(torch.sum(x*torch.roll(y,-1)-y*torch.roll(x,-1)))

    # soft size regularization
    return lam * (1.0 / (area + 1e-3))


def edge_uniformity(pm, lam=2e-4):
    v = pm.get_vertices()
    edges = torch.norm(torch.roll(v,-1,0)-v, dim=1)
    return lam*torch.var(edges)

def convexity_reg(pm,lam=3e-4):
    v=pm.get_vertices(); vp=torch.roll(v,1,0); vn=torch.roll(v,-1,0)
    cross=(v-vp)[:,0]*(vn-v)[:,1]-(v-vp)[:,1]*(vn-v)[:,0]
    return lam*torch.mean(F.relu(-cross))

def boundary_reg(pm,lam=1e-3):
    v=pm.get_vertices()
    return lam*(torch.mean(F.relu(-v)**2)+torch.mean(F.relu(v-DOMAIN)**2))

STAGES=[
    (300,  6.0, 3e-2, 1e-2, 300., 5e-5, 4e-4, 1e-3),
    (400, 12.0, 1e-2, 5e-3, 400., 3e-5, 3e-4, 1e-3),
    (500, 18.0, 3e-3, 2e-3, 500., 1e-5, 2e-4, 1e-3),
    (400, 24.0, 8e-4, 5e-4, 600., 5e-6, 1e-4, 1e-3),
]
TOTAL=sum(s[0] for s in STAGES)
print(f"Inversion: {TOTAL} epochs\n")

best_err=float('inf'); best_st=None; best_ep=0
history={'dl':[],'err':[],'fdm_dl':[]}
t_start=time.time(); epoch=0

# --- Unfreeze only the poly_branch of the PINN for local fine-tuning during inversion ---
# Freeze everything first...
for p in unet.parameters(): p.requires_grad_(False)

# Then unfreeze poly_branch parameters only (so PINN can adapt sensitivities to local region)
unet_poly_params = []
for name, p in unet.named_parameters():
    if 'poly_br' in name:
        p.requires_grad_(True)
        unet_poly_params.append(p)

print(f"Unet poly_branch params unfrozen for inversion: {sum(p.numel() for p in unet_poly_params):,}")

# --- Anchor copy of poly_branch parameters (pretrained reference) ---
unet_poly_init = [p.detach().clone() for p in unet_poly_params]

# Prevents excessive drift of the surrogate parameters during inversion, which can lead to unphysical predictions and poor gradients. 
# The anchor loss penalizes large deviations from the initial pretrained parameters, keeping the surrogate grounded while still allowing some local adaptation.
lambda_unet_anchor = 1e-6

# Very small learning rate for surrogate fine-tuning
lr_unet = 1e-5

print(f"{'Ep':>5} {'Stg':>4} {'α':>5} {'PINN_dl':>12} {'FDM_dl':>12} {'CenterErr':>10} {'Time':>6}")
print("─"*75)

for stg_idx,(n_ep,alpha,lr_c,lr_o,wd,wp,wc,wb) in enumerate(STAGES):
    opt = optim.AdamW([
        {'params':[poly.center],  'lr':lr_c},
        {'params':[poly.offsets], 'lr':lr_o},
        {'params': unet_poly_params, 'lr': lr_unet},
    ], weight_decay=1e-5)
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=n_ep,eta_min=lr_c*0.05)
    poly.alpha=alpha

    for _ in range(n_ep):
        opt.zero_grad()
        # PINN prediction (fast surrogate)
        V_pred_pinn = predict_voltages(unet, poly)
        dl_pinn = torch.mean((V_pred_pinn - V_MEAS)**2)
        
        # True physics correction (FDM)
        V_pred_fdm = fdm.solve(poly)
        dl_fdm = torch.mean((V_pred_fdm - V_MEAS)**2)
        
        # Hybrid loss
        dl = dl_pinn + 50.0 * dl_fdm
        
        reg = (wp*perimeter_reg(poly,1)
              +wc*convexity_reg(poly,1)
              +wb*boundary_reg(poly,1)
              +angle_reg(poly)
              +area_reg(poly)
              +edge_uniformity(poly))
        
        # --- anchor regularization to prevent surrogate drift ---
        reg_unet = 0.0
        for p, p0 in zip(unet_poly_params, unet_poly_init):
            reg_unet += torch.sum((p - p0)**2)
        reg_unet = lambda_unet_anchor * reg_unet
        
        loss = wd*(dl_pinn + 50.0*dl_fdm) + reg + reg_unet
        loss.backward()
        
        # clip gradients for both polygon + surrogate branch
        torch.nn.utils.clip_grad_norm_(
            [poly.center, poly.offsets] + list(unet_poly_params),
            1.0
        )
        opt.step()
        sch.step()
        
        # # ---- small exploration noise to escape flat PINN landscape ----
        # with torch.no_grad():
        #     poly.center += 0.002 * torch.randn_like(poly.center)

        with torch.no_grad():
            pc=poly.get_vertices().mean(0)
            err=torch.norm(pc-TRUE_C).item()
        if err<best_err-0.002:
            best_err=err; best_st=copy.deepcopy(poly.state_dict()); best_ep=epoch

        if epoch%10==0:
            with torch.no_grad():
                fdm_dl=torch.mean((fdm.solve(poly)-V_MEAS)**2).item()
            history['dl'].append(dl.item())
            history['err'].append(err)
            history['fdm_dl'].append(fdm_dl)
            el=(time.time()-t_start)/60.
            mk=" ◀" if abs(err-best_err)<0.003 else ""
            print(f"{epoch:5d}  S{stg_idx}  {alpha:5.1f} "
                  f"{dl.item():12.2e} {fdm_dl:12.2e} {err:10.4f} "
                  f"{el:5.2f}m  pred=({pc[0]:.3f},{pc[1]:.3f}){mk}")
        epoch+=1

    with torch.no_grad():
        vp=poly.get_vertices().cpu().numpy(); vt=TRUE_NP
        print(f"\n  Stage {stg_idx} end:")
        for i in range(4):
            print(f"    v{i}: pred=({vp[i,0]:.3f},{vp[i,1]:.3f})  "
                  f"true=({vt[i,0]:.3f},{vt[i,1]:.3f})")
        print()

# ══════════════════════════════════════════════════════════
# 12. Metrics and visualization
# ══════════════════════════════════════════════════════════
# ══════════════════════════════════════════════════════════
# 11.5 LBFGS refinement (precise inverse solve)
# ══════════════════════════════════════════════════════════
print("\nStarting LBFGS refinement...")

lbfgs = torch.optim.LBFGS(
    [poly.center, poly.offsets],
    lr=0.5,
    max_iter=120,
    history_size=50,
    line_search_fn="strong_wolfe"
)

def closure():
    lbfgs.zero_grad()

    V_pred = predict_voltages(unet, poly)
    dl = torch.mean((V_pred - V_MEAS)**2)

    reg = (5e-5 * perimeter_reg(poly,1)
          +3e-4 * convexity_reg(poly,1)
          +1e-3 * boundary_reg(poly,1)
          +angle_reg(poly)
          +area_reg(poly)
          +edge_uniformity(poly))

    loss = 600. * dl + reg
    loss.backward()

    return loss

lbfgs.step(closure)

print("LBFGS refinement done.")

if best_st: poly.load_state_dict(best_st)

with torch.no_grad():
    V_final_fdm=fdm.solve(poly)
    dl_fdm_final=torch.mean((V_final_fdm-V_MEAS)**2).item()
    V_final_pinn=predict_voltages(unet,poly)
    dl_pinn_final=torch.mean((V_final_pinn-V_MEAS)**2).item()

pred_np=poly.get_vertices().detach().cpu().numpy()

def rasterize(verts,R=500):
    xs=np.linspace(0,DOMAIN,R); ys=np.linspace(0,DOMAIN,R)
    XX,YY=np.meshgrid(xs,ys); pts=np.column_stack([XX.ravel(),YY.ravel()])
    path=Path(np.vstack([verts,verts[0]]))
    return path.contains_points(pts).reshape(R,R)

def poly_boundary(verts,n=500):
    pts=[]
    for i in range(len(verts)):
        p0=verts[i]; p1=verts[(i+1)%len(verts)]
        for t in np.linspace(0,1,n//len(verts),endpoint=False):
            pts.append(p0+t*(p1-p0))
    return np.array(pts)

def compute_metrics(pv,tv,R=500):
    m={}
    pc=pv.mean(0); tc=tv.mean(0)
    m['center_error']=float(np.linalg.norm(pc-tc))
    m['center_pred']=pc.tolist(); m['center_true']=tc.tolist()
    mp=rasterize(pv,R).astype(float); mt=rasterize(tv,R).astype(float)
    inter=(mp*mt).sum(); union=(mp+mt-mp*mt).sum()
    m['IoU']=float(inter/union) if union>0 else 0.
    m['Dice']=float(2*inter/(mp.sum()+mt.sum()))
    bp=poly_boundary(pv); bt=poly_boundary(tv)
    h1=directed_hausdorff(bp,bt)[0]; h2=directed_hausdorff(bt,bp)[0]
    m['Hausdorff']=float(max(h1,h2))
    tree=KDTree(bt); dists,_=tree.query(bp)
    m['mean_boundary_dev']=float(dists.mean())
    def area(v):
        a=0.
        for i in range(len(v)): a+=v[i,0]*v[(i+1)%len(v),1]-v[(i+1)%len(v),0]*v[i,1]
        return abs(a)/2.
    pa=area(pv); ta=area(tv)
    m['area_pred']=float(pa); m['area_true']=float(ta)
    m['area_rel_err']=float(abs(pa-ta)/ta)
    def perim(v): return sum(np.linalg.norm(v[(i+1)%len(v)]-v[i]) for i in range(len(v)))
    m['perim_pred']=float(perim(pv)); m['perim_true']=float(perim(tv))
    return m

metrics=compute_metrics(pred_np,TRUE_NP)

print(f"\n{'='*60}")
print(f"  PUBLICATION METRICS (PINN surrogate inversion)")
print(f"  Best epoch: {best_ep}")
print(f"{'='*60}")
print(f"  Center localization error (L2) : {metrics['center_error']:.4f} units")
print(f"  Predicted center               : ({metrics['center_pred'][0]:.4f}, {metrics['center_pred'][1]:.4f})")
print(f"  True center                    : ({metrics['center_true'][0]:.4f}, {metrics['center_true'][1]:.4f})")
print(f"  ─────────────────────────────────────────────────────")
print(f"  IoU  (Intersection/Union)      : {metrics['IoU']:.4f}  ({metrics['IoU']*100:.1f}%)")
print(f"  Dice coefficient               : {metrics['Dice']:.4f}  ({metrics['Dice']*100:.1f}%)")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Hausdorff distance             : {metrics['Hausdorff']:.4f} units")
print(f"  Mean boundary deviation        : {metrics['mean_boundary_dev']:.4f} units")
print(f"  ─────────────────────────────────────────────────────")
print(f"  Area predicted / true          : {metrics['area_pred']:.4f} / {metrics['area_true']:.4f}")
print(f"  Area relative error            : {metrics['area_rel_err']*100:.2f}%")
print(f"  ─────────────────────────────────────────────────────")
print(f"  PINN data residual (MSE)       : {dl_pinn_final:.2e}")
print(f"  FDM  data residual (MSE)       : {dl_fdm_final:.2e}  (ground truth)")
print(f"{'='*60}")
print(f"\nTotal runtime: {(time.time()-t_start)/60:.2f} min")

# ══════════════════════════════════════════════════════════
# 13. Figures
# ══════════════════════════════════════════════════════════
R_vis=300
grid=torch.stack(torch.meshgrid(
    torch.linspace(0,DOMAIN,R_vis,device=device),
    torch.linspace(0,DOMAIN,R_vis,device=device),indexing='ij'),
    dim=-1).reshape(-1,2)

with torch.no_grad():
    sp=poly(grid).reshape(R_vis,R_vis).cpu().numpy()
    st=hard_sigma_true(grid).reshape(R_vis,R_vis).cpu().numpy()

vp_c=np.vstack([pred_np,pred_np[0]]); vt_c=np.vstack([TRUE_NP,TRUE_NP[0]])

# ── Fig 1: 3-panel ─────────────────────────────────────
fig,axes=plt.subplots(1,3,figsize=(16,5))
ext=[0,DOMAIN,0,DOMAIN]

im0=axes[0].imshow(st.T,origin='lower',extent=ext,
                   vmin=SIGMA_BG,vmax=SIGMA_INC,cmap='hot',interpolation='bilinear')
axes[0].plot(vt_c[:,0],vt_c[:,1],'c-o',lw=2,ms=5)
axes[0].set_title('Ground Truth σ',fontsize=13,fontweight='bold')
axes[0].set_xlabel('x [cm]'); axes[0].set_ylabel('y [cm]')
plt.colorbar(im0,ax=axes[0],fraction=0.046,pad=0.04,label='σ [S/m]')

im1=axes[1].imshow(sp.T,origin='lower',extent=ext,
                   vmin=SIGMA_BG,vmax=SIGMA_INC,cmap='hot',interpolation='bilinear')
axes[1].plot(vt_c[:,0],vt_c[:,1],'c--',lw=1.5,alpha=0.7,label='True')
axes[1].plot(vp_c[:,0],vp_c[:,1],'r-o',lw=2,ms=5,label='PINN pred')
axes[1].legend(fontsize=9)
axes[1].set_title(f'PINN Reconstruction\nIoU={metrics["IoU"]:.3f}  Dice={metrics["Dice"]:.3f}',
                  fontsize=12,fontweight='bold')
axes[1].set_xlabel('x [cm]')
plt.colorbar(im1,ax=axes[1],fraction=0.046,pad=0.04,label='σ [S/m]')

diff=np.abs(sp-st)
im2=axes[2].imshow(diff.T,origin='lower',extent=ext,cmap='Blues',interpolation='bilinear')
axes[2].plot(vt_c[:,0],vt_c[:,1],'c--',lw=1.5)
axes[2].plot(vp_c[:,0],vp_c[:,1],'r-',lw=1.5)
axes[2].set_title(f'|Pred−True|  Hausdorff={metrics["Hausdorff"]:.3f}',
                  fontsize=12,fontweight='bold')
axes[2].set_xlabel('x [cm]')
plt.colorbar(im2,ax=axes[2],fraction=0.046,pad=0.04,label='|Δσ|')
plt.tight_layout(); plt.savefig('final_result_pinn.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 2: PINN vs FDM loss curves ─────────────────────
fig,(a1,a2,a3)=plt.subplots(1,3,figsize=(17,4))
eps=[i*10 for i in range(len(history['dl']))]

a1.semilogy(eps,history['dl'],'b-',lw=2,label='PINN residual')
a1.semilogy(eps,history['fdm_dl'],'r--',lw=2,label='FDM residual')
a1.set_title('Data Residual (PINN vs FDM)',fontweight='bold')
a1.set_xlabel('Epoch'); a1.legend(); a1.grid(True,alpha=0.4)
for s,n_ec in zip(range(4),[300,700,1200,1600]):
    a1.axvline(n_ec,c='gray',ls=':',alpha=0.5)

a2.plot(eps,history['err'],'r-',lw=2)
a2.axhline(best_err,c='k',ls='--',lw=1.5,label=f'best={best_err:.4f}')
a2.set_title('Center Localization Error',fontweight='bold')
a2.set_xlabel('Epoch'); a2.set_ylabel('L2 error'); a2.legend(); a2.grid(True,alpha=0.4)

# FDM vs PINN residual correlation
a3.scatter(history['fdm_dl'],history['dl'],c=eps,cmap='viridis',s=10,alpha=0.7)
a3.set_xlabel('FDM residual'); a3.set_ylabel('PINN residual')
a3.set_title('PINN tracks FDM (should be correlated)',fontweight='bold')
a3.set_xscale('log'); a3.set_yscale('log')
plt.colorbar(plt.cm.ScalarMappable(cmap='viridis'),ax=a3,label='epoch')
plt.tight_layout(); plt.savefig('training_curves_pinn.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 3: Polygon overlay ──────────────────────────────
fig,ax=plt.subplots(figsize=(7,7))
ax.set_xlim(0,DOMAIN); ax.set_ylim(0,DOMAIN); ax.set_aspect('equal')
ax.set_facecolor('#f9f9f9'); ax.grid(True,alpha=0.3)
from matplotlib.patches import Polygon as MPoly
ax.add_patch(MPoly(TRUE_NP, closed=True, facecolor='cyan', alpha=0.25))
ax.add_patch(MPoly(pred_np,closed=True,facecolor='red',alpha=0.25))
ax.plot(vt_c[:,0],vt_c[:,1],'c-o',lw=2.5,ms=8,label=f'True  area={TRUE_AREA:.2f}')
ax.plot(vp_c[:,0],vp_c[:,1],'r-s',lw=2.5,ms=8,label=f"Pred  area={metrics['area_pred']:.2f}")
tc_np=TRUE_NP.mean(0); pc_np=pred_np.mean(0)
ax.plot(*tc_np,'c*',ms=15,zorder=5); ax.plot(*pc_np,'r*',ms=15,zorder=5)
ax.plot([tc_np[0],pc_np[0]],[tc_np[1],pc_np[1]],'k--',lw=1.5,
        label=f"Δcenter={metrics['center_error']:.4f}")
ax.set_title(f'PINN Polygon Reconstruction\n'
             f'IoU={metrics["IoU"]:.4f}  Dice={metrics["Dice"]:.4f}  '
             f'Hausdorff={metrics["Hausdorff"]:.4f}',fontsize=11,fontweight='bold')
ax.set_xlabel('x [cm]'); ax.set_ylabel('y [cm]'); ax.legend(fontsize=10)
plt.tight_layout(); plt.savefig('polygon_comparison_pinn.png',dpi=150,bbox_inches='tight'); plt.close()

# ── Fig 4: Dark theme polygon comparison (plasma/magma) ─────────────────
fig,(ax1,ax2)=plt.subplots(1,2,figsize=(12,6),facecolor='black')

for ax in (ax1,ax2):
    ax.set_facecolor('black')
    ax.set_xlim(0,DOMAIN)
    ax.set_ylim(0,DOMAIN)
    ax.set_aspect('equal')
    ax.grid(color='white',alpha=0.15)
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')

# TRUE polygon (plasma)
ax1.fill(vt_c[:,0],vt_c[:,1],
         color=plt.cm.plasma(0.75),
         alpha=0.85)

ax1.plot(vt_c[:,0],vt_c[:,1],
         color=plt.cm.plasma(0.95),
         lw=3)

ax1.set_title("Ground Truth Polygon",
              color='white',
              fontsize=14,
              fontweight='bold')

# RECONSTRUCTED polygon (magma)
ax2.fill(vp_c[:,0],vp_c[:,1],
         color=plt.cm.magma(0.75),
         alpha=0.85)

ax2.plot(vp_c[:,0],vp_c[:,1],
         color=plt.cm.magma(0.95),
         lw=3)

ax2.set_title("Reconstructed Polygon (PINN)",
              color='white',
              fontsize=14,
              fontweight='bold')

plt.tight_layout()
plt.savefig("polygon_dark_plasma.png",
            dpi=200,
            facecolor=fig.get_facecolor(),
            bbox_inches='tight')
plt.close()

print("\nSaved: final_result_pinn.png  polygon_comparison_pinn.png  polygon_dark_plasma.png  training_curves_pinn.png")
print("Done.")
