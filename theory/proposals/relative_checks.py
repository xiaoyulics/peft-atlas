"""Numerical sanity checks for theory/proposals/relative.md (Jacobian-rank image dimensions, Adam/SGD hyperparameter equivalences, Noether charge). Run: python3 relative_checks.py"""
# ---- Part 1: image dimensions ----
import torch, math
torch.set_default_dtype(torch.float64)
torch.manual_seed(0)
from torch.autograd.functional import jacobian

def jrank(f, q, tol=1e-8):
    J = jacobian(f, q)
    J = J.reshape(-1, q.numel())
    s = torch.linalg.svdvals(J)
    return int((s > tol * s[0]).sum())

m, n, r = 7, 5, 2
W0 = torch.randn(m, n)

# LoRA generic and zero-init
def lora(q):
    B = q[:m*r].reshape(m, r); A = q[m*r:].reshape(r, n)
    return W0 + B @ A
qg = torch.randn(m*r + r*n)
q0 = torch.cat([torch.zeros(m*r), torch.randn(r*n)])
print("LoRA generic rank", jrank(lora, qg), "expected", r*(m+n-r))
print("LoRA zero-init S1", jrank(lora, q0), "expected m*r", m*r)
qB = torch.cat([torch.randn(m*r), torch.zeros(r*n)])
print("LoRA Init[B] S1", jrank(lora, qB), "expected n*r", n*r)

# VeRA
Bf = torch.randn(m, r); Af = torch.randn(r, n)
def vera(q):
    b = q[:m]; d = q[m:]
    return W0 + torch.diag(b) @ Bf @ torch.diag(d) @ Af
print("VeRA generic rank", jrank(vera, torch.randn(m+r)), "expected m+r-1", m+r-1)

# LoHa (r1=r2=r)
def loha(q):
    k = 0
    B1 = q[k:k+m*r].reshape(m, r); k += m*r
    A1 = q[k:k+r*n].reshape(r, n); k += r*n
    B2 = q[k:k+m*r].reshape(m, r); k += m*r
    A2 = q[k:k+r*n].reshape(r, n)
    return W0 + (B1 @ A1) * (B2 @ A2)
nl = 2*(m*r + r*n)
print("LoHa generic rank", jrank(loha, torch.randn(nl)), "upper bound 2r(m+n-r)-(m+n-1)", 2*r*(m+n-r)-(m+n-1), "ambient", m*n)
m2, n2, rr = 8, 8, 1
W0b = torch.randn(m2, n2)
def loha2(q):
    k=0
    B1=q[k:k+m2*rr].reshape(m2,rr);k+=m2*rr
    A1=q[k:k+rr*n2].reshape(rr,n2);k+=rr*n2
    B2=q[k:k+m2*rr].reshape(m2,rr);k+=m2*rr
    A2=q[k:k+rr*n2].reshape(rr,n2)
    return W0b + (B1@A1)*(B2@A2)
print("LoHa r1=r2=1 on 8x8 rank", jrank(loha2, torch.randn(2*(m2+n2))), "bound 2(m+n-1)-(m+n-1)=", (m2+n2-1))

# HiRA
W0nz = torch.randn(m, n)
def hira(q):
    B = q[:m*r].reshape(m, r); A = q[m*r:].reshape(r, n)
    return W0nz + W0nz * (B @ A)
print("HiRA generic rank", jrank(hira, torch.randn(m*r+r*n)), "expected", r*(m+n-r))
Bh = torch.randn(m, r); Ah = torch.randn(r, n)
print("rank of W0 ⊙ BA:", torch.linalg.matrix_rank(W0nz * (Bh@Ah)).item(), "rank BA", r)

# OFT full via Cayley on right (input side), W0 m x n with m<=n full row rank
mo, no = 3, 5
W0o = torch.randn(mo, no)
iu = torch.triu_indices(no, no, 1)
def oft(q):
    S = torch.zeros(no, no); S[iu[0], iu[1]] = q; S = S - S.T
    I = torch.eye(no)
    R = (I + S) @ torch.linalg.inv(I - S)
    return W0o @ R
print("OFT full rank", jrank(oft, 0.3*torch.randn(iu.shape[1])), "expected mn - m(m+1)/2", mo*no - mo*(mo+1)//2)

# HRA: right multiplication by product of r Householders; W0 injective (m>=n)
mh, nh, rh = 7, 5, 2
W0h = torch.randn(mh, nh)
def hra(q):
    U = q.reshape(rh, nh)
    R = torch.eye(nh)
    for i in range(rh):
        u = U[i]
        R = R @ (torch.eye(nh) - 2*torch.outer(u,u)/(u@u))
    return W0h @ R
print("HRA rank", jrank(hra, torch.randn(rh*nh)), "expected rn - r(r+1)/2", rh*nh - rh*(rh+1)//2)
U = torch.randn(rh, nh); R = torch.eye(nh)
for i in range(rh):
    u=U[i]; R = R @ (torch.eye(nh) - 2*torch.outer(u,u)/(u@u))
print("rank(W0(R-I))", torch.linalg.matrix_rank(W0h @ (R - torch.eye(nh))).item())

# Converse: random rotation R with rank(R-I)=2 (rotation in random plane) -> is product of 2 reflections? check rank and det
Qm,_ = torch.linalg.qr(torch.randn(nh, nh))
th = 0.7
Rot = torch.eye(nh); Rot[:2,:2] = torch.tensor([[math.cos(th), -math.sin(th)],[math.sin(th), math.cos(th)]])
Rg = Qm @ Rot @ Qm.T
print("rotation in a plane: rank(R-I)", torch.linalg.matrix_rank(Rg - torch.eye(nh)).item(), "det", torch.det(Rg).item())

# DoRA (row magnitudes, HF convention): W = diag(mag) V / ||V||_row, V = W0 + BA
def dora(q):
    mag = q[:m]; B = q[m:m+m*r].reshape(m, r); A = q[m+m*r:].reshape(r, n)
    V = W0 + B @ A
    return torch.diag(mag) @ (V / V.norm(dim=1, keepdim=True))
qd = torch.cat([W0.norm(dim=1) + 0.1*torch.randn(m), torch.randn(m*r), torch.randn(r*n)])
print("DoRA generic rank", jrank(dora, qd), "LoRA", r*(m+n-r), "+m =", r*(m+n-r)+m, "ambient", m*n)

# Transversality OFT tangent vs row-scaling tangent at W0 (m<=n full row rank)
I = torch.eye(no)
Tof = []
for k in range(iu.shape[1]):
    S = torch.zeros(no,no); S[iu[0][k], iu[1][k]] = 1; S = S - S.T
    Tof.append((W0o @ S).flatten())
Tia = [ (torch.diag(torch.eye(mo)[i]) @ W0o).flatten() for i in range(mo)]
M1 = torch.stack(Tof); M2 = torch.stack(Tia)
r1 = torch.linalg.matrix_rank(M1).item(); r2 = torch.linalg.matrix_rank(M2).item(); r12 = torch.linalg.matrix_rank(torch.cat([M1,M2])).item()
print("OFT tangent", r1, "IA3 tangent", r2, "sum span", r12, "(transverse iff = sum)")

# LoKr = rank-1 LoRA in Van Loan coordinates check
m1,n1,m2_,n2_ = 2,3,4,2
C = torch.randn(m1,n1); D = torch.randn(m2_,n2_)
K = torch.kron(C, D)
# Van Loan rearrangement
Rr = K.reshape(m1, m2_, n1, n2_).permute(0,2,1,3).reshape(m1*n1, m2_*n2_)
print("rank of kron", torch.linalg.matrix_rank(K).item(), "rank of rearranged", torch.linalg.matrix_rank(Rr).item())

# ---- Part 2: LoHa dimension formula ----

torch.set_default_dtype(torch.float64); torch.manual_seed(3)
from torch.autograd.functional import jacobian
def jrank(f,q,tol=1e-9):
    J=jacobian(f,q).reshape(-1,q.numel()); s=torch.linalg.svdvals(J); return int((s>tol*s[0]).sum())
for (m,n,r1,r2) in [(9,8,2,2),(12,10,2,3),(16,16,2,2),(10,10,3,3),(20,18,2,2)]:
    def f(q):
        k=0
        B1=q[k:k+m*r1].reshape(m,r1);k+=m*r1
        A1=q[k:k+r1*n].reshape(r1,n);k+=r1*n
        B2=q[k:k+m*r2].reshape(m,r2);k+=m*r2
        A2=q[k:k+r2*n].reshape(r2,n)
        return (B1@A1)*(B2@A2)
    nq=(m+n)*(r1+r2)
    d=jrank(f,torch.randn(nq))
    pred=min(m*n, r1*(m+n-r1)+r2*(m+n-r2)-(m+n-1))
    print((m,n,r1,r2),"dim",d,"pred",pred,"params",nq,"rank bound",r1*r2)

# ---- Part 3: dynamics ----
import torch, math
torch.set_default_dtype(torch.float64)
torch.manual_seed(1)
m, n, r, N = 12, 10, 3, 40
W0 = torch.randn(m, n); Wst = W0 + 0.5*torch.randn(m, n)
X = torch.randn(n, N)
def loss(W): return ((W - Wst) @ X).pow(2).sum() / N
A0 = torch.randn(r, n) / math.sqrt(n)

def run(s, etaA, etaB, A_init, steps=300, opt='adam', eps=0.0, wd=0.0):
    B = torch.zeros(m, r, requires_grad=True); A = A_init.clone().requires_grad_(True)
    if opt == 'adam':
        o = torch.optim.Adam([{'params':[B],'lr':etaB},{'params':[A],'lr':etaA}], betas=(0.9,0.999), eps=eps, weight_decay=0)
    else:
        o = torch.optim.SGD([{'params':[B],'lr':etaB},{'params':[A],'lr':etaA}])
    Ws = []
    for t in range(steps):
        o.zero_grad()
        W = W0 + s * B @ A
        L = loss(W); L.backward()
        # convention 0/0=0 for eps=0
        if opt=='adam' and eps==0.0:
            for p in (A,B):
                if p.grad is not None and p.grad.abs().max()==0: p.grad += 0.0
        o.step()
        if opt=='adam' and eps==0.0:
            for p in (A,B):
                if torch.isnan(p).any(): raise RuntimeError('nan')
        Ws.append((W0 + s*B@A).detach().clone())
    return torch.stack(Ws)

# Adam with eps tiny (eps=0 gives 0/0 at step 0 for A); use eps=1e-300
eps = 1e-300
s, eta, lam = 2.0, 1e-3, 16.0
T1 = run(s, eta, lam*eta, A0, eps=eps)
T2 = run(lam*s, eta, eta, A0, eps=eps)
print("Adam: LoRA+(s,eta,lam*eta) vs LoRA(lam*s,eta,eta) max traj diff:", (T1-T2).abs().max().item(), " traj scale", (T1-W0).abs().max().item())
T3 = run(s, eta, eta, A0, eps=eps)
print("Adam: vs plain LoRA(s) diff (should be large):", (T1-T3).abs().max().item())
# two invariants: (s*etaA*etaB, sigmaA/etaA)
t=3.0
T4 = run(s, t*eta, lam*eta/t, t*A0, eps=eps)
print("Adam: gauge-transformed hyperparams diff:", (T1-T4).abs().max().item())
# with eps=1e-8 realistic
T1e = run(s, eta, lam*eta, A0, eps=1e-8); T2e = run(lam*s, eta, eta, A0, eps=1e-8)
print("Adam eps=1e-8 diff:", (T1e-T2e).abs().max().item())
# SGD: LoRA+(s,eta,lam eta) vs LoRA(sqrt(lam) s, eta, eta)
eta_s = 2e-3
S1 = run(s, eta_s, lam*eta_s, A0, opt='sgd'); S2 = run(math.sqrt(lam)*s, eta_s, eta_s, A0, opt='sgd')
print("SGD: LoRA+ vs LoRA(sqrt(lam) s) diff:", (S1-S2).abs().max().item(), "scale", (S1-W0).abs().max().item())

# Noether: small-step GD, charge B^T B/etaB - A A^T/etaA
def charge_run(etaA, etaB, steps=4000, h=1e-4):
    B = torch.zeros(m, r, requires_grad=True); A = A0.clone().requires_grad_(True)
    c0 = None; drift = 0
    for t in range(steps):
        W = W0 + B @ A; L = loss(W)
        gB, gA = torch.autograd.grad(L, [B, A])
        with torch.no_grad():
            B -= h*etaB*gB; A -= h*etaA*gA
            c = B.T@B/etaB - A@A.T/etaA
            if c0 is None: c0 = -(A0@A0.T)/etaA
            drift = max(drift, (c-c0).abs().max().item())
    with torch.no_grad():
        ev_min = torch.linalg.eigvalsh(A@A.T - A0@A0.T).min().item()
    return drift, (B.norm().item()), ev_min, loss(W0+B@A).item()
print("Noether (etaA=etaB=1): max drift, |B|, min eig(AA^T - A0A0^T), loss", charge_run(1.0, 1.0))
print("Noether (etaA=1, etaB=16):", charge_run(1.0, 16.0))
