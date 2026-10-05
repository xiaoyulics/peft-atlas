"""Numerical checks for para-lens.md (Thm 3.6, 3.7, Cor 3.5, Thm 3.10, 3.14, 3.15). Run: python3 para-lens-checks.py"""
# ---- Part 1: Theorem 3.7 closed form + Theorem 3.6 charge with unequal learning rates ----
import numpy as np
from scipy.integrate import solve_ivp
rng = np.random.default_rng(0)
dout, din, r = 7, 9, 3
X = rng.normal(size=(40, din)); Wstar = rng.normal(size=(dout, din))
Y = X @ Wstar.T
W0 = rng.normal(size=(dout, din))*0.3
def gradW(W):  # L = 1/(2N) ||X W^T - Y||^2
    return ((X @ W.T - Y).T @ X)/X.shape[0]
s, etaA, etaB, c = 0.7, 0.5, 2.0, 0.8
# A0 with A0 A0^T = c I
Q,_ = np.linalg.qr(rng.normal(size=(din, r))); A0 = np.sqrt(c)*Q.T
B0 = np.zeros((dout, r))
def rhs(t, y):
    B = y[:dout*r].reshape(dout, r); A = y[dout*r:].reshape(r, din)
    G = gradW(W0 + s*B@A)
    dB = -etaB*s*G@A.T; dA = -etaA*s*B.T@G
    return np.concatenate([dB.ravel(), dA.ravel()])
sol = solve_ivp(rhs, [0, 3], np.concatenate([B0.ravel(), A0.ravel()]), rtol=1e-11, atol=1e-12, dense_output=True)
def fk(x,k): return (np.sqrt(k*k+4*x)-k)/2
def gk(x,k): return (np.sqrt(k*k+4*x)+k)/2
def matfun(M, f):
    w,V = np.linalg.eigh((M+M.T)/2); w = np.clip(w,0,None); return V@np.diag(f(w))@V.T
kappa = c/etaA; m = s*np.sqrt(etaA*etaB)
for t in [0.05, 0.5, 1.5, 3.0]:
    y = sol.sol(t); B = y[:dout*r].reshape(dout, r); A = y[dout*r:].reshape(r, din)
    Phi = B.T@B/etaB - A@A.T/etaA
    dW = s*B@A; G = gradW(W0+dW)
    true = -s*s*(etaB*G@A.T@A + etaA*B@B.T@G)
    # formula: right side acts on row space of dW; use SVD of dW restricted to rank r
    U,S,Vt = np.linalg.svd(dW); U=U[:,:r]; S=S[:r]; V=Vt[:r].T
    N2 = (S/m)**2
    right = V@np.diag(gk(N2,kappa))@V.T; left = U@np.diag(fk(N2,kappa))@U.T
    pred = -m*m*(G@right + left@G)
    print(f"t={t}: |Phi+kappa I|={np.linalg.norm(Phi+kappa*np.eye(r)):.2e}, rel err formula={np.linalg.norm(pred-true)/np.linalg.norm(true):.2e}, svals dW={np.round(S,3)}, crossover s c sqrt(lam)={s*c*np.sqrt(etaB/etaA):.3f}")

# ---- Part 2: weight decay, gauge equivariance, GaLore equivalence, rearrangement, HRA ----
import numpy as np
from scipy.integrate import solve_ivp
rng = np.random.default_rng(1)
m, n, r = 6, 8, 2
X = rng.normal(size=(30, n)); Y = X @ rng.normal(size=(m, n)).T
W0 = 0.3*rng.normal(size=(m, n))
gradW = lambda W: ((X @ W.T - Y).T @ X)/X.shape[0]
# (a) weight decay: equal LR eta, decay lam -> Phi(t) = exp(-2 eta lam t) Phi0
eta, lam = 1.0, 0.3
B0 = 0.2*rng.normal(size=(m, r)); A0 = 0.5*rng.normal(size=(r, n))
def rhs(t, y):
    B = y[:m*r].reshape(m, r); A = y[m*r:].reshape(r, n); G = gradW(W0 + B@A)
    return np.concatenate([(-eta*(G@A.T + lam*B)).ravel(), (-eta*(B.T@G + lam*A)).ravel()])
sol = solve_ivp(rhs, [0, 2], np.concatenate([B0.ravel(), A0.ravel()]), rtol=1e-11, atol=1e-12)
B = sol.y[:m*r, -1].reshape(m, r); A = sol.y[m*r:, -1].reshape(r, n)
Phi0 = B0.T@B0 - A0@A0.T; Phi = B.T@B - A@A.T
print("(a) weight decay charge decay err:", np.linalg.norm(Phi - np.exp(-2*eta*lam*2)*Phi0)/np.linalg.norm(Phi0))
# (b) Zhang-Pilanci preconditioned update is GL_r equivariant and induced dW depends only on W
G = gradW(W0 + B0@A0)
def zp(B, A, G):
    dB = -G@A.T@np.linalg.inv(A@A.T); dA = -np.linalg.inv(B.T@B)@B.T@G; return dB, dA
g = rng.normal(size=(r, r)) + 2*np.eye(r)
dB, dA = zp(B0, A0, G); dB2, dA2 = zp(B0@np.linalg.inv(g), g@A0, G)
print("(b) equivariance err:", np.linalg.norm(dB2 - dB@np.linalg.inv(g)), np.linalg.norm(dA2 - g@dA))
dW1 = dB@A0 + B0@dA; dW2 = dB2@(g@A0) + (B0@np.linalg.inv(g))@dA2
U,_,_ = np.linalg.svd(B0, full_matrices=False); V,_,_ = np.linalg.svd(A0.T, full_matrices=False)
print("    induced dW = -(P_U G + G P_V):", np.linalg.norm(dW1 + (U@U.T@G + G@V@V.T)), np.linalg.norm(dW1-dW2))
# plain GD not equivariant
dWp1 = -(G@A0.T@A0 + B0@B0.T@G); Bg, Ag = B0@np.linalg.inv(g), g@A0; dWp2 = -(G@Ag.T@Ag + Bg@Bg.T@G)
print("    plain GD induced dW gauge-dependence:", np.linalg.norm(dWp1 - dWp2)/np.linalg.norm(dWp1))
# (c) GaLore-Adam (frozen left projector P, one period) == one-sided LoRA-Adam rho(Bq) = W_k + P Bq
P,_ = np.linalg.qr(rng.normal(size=(m, r)))
def adam_run(step_grad, x0, T=50, lr=1e-2, b1=0.9, b2=0.999, eps=1e-8):
    x = x0.copy(); mo = np.zeros_like(x); v = np.zeros_like(x); traj = []
    for t in range(1, T+1):
        g_ = step_grad(x); mo = b1*mo + (1-b1)*g_; v = b2*v + (1-b2)*g_**2
        x = x - lr*(mo/(1-b1**t))/(np.sqrt(v/(1-b2**t)) + eps); traj.append(x.copy())
    return traj
# GaLore: state W, Adam on R = P^T G, update W -= lr * P * adam(R)
W = W0.copy(); mo = np.zeros((r, n)); v = np.zeros((r, n))
for t in range(1, 51):
    R = P.T@gradW(W); mo = 0.9*mo + 0.1*R; v = 0.999*v + 0.001*R**2
    W = W - 1e-2*P@((mo/(1-0.9**t))/(np.sqrt(v/(1-0.999**t)) + 1e-8))
Bq = adam_run(lambda Bq: P.T@gradW(W0 + P@Bq), np.zeros((r, n)))[-1]
print("(c) GaLore vs one-sided LoRA (Adam, 50 steps):", np.linalg.norm(W - (W0 + P@Bq)))
# (d) Van Loan-Pitsianis rearrangement: R(A kron B) = vec(A) vec(B)^T
d1, n1, d2, n2 = 2, 3, 4, 2
Ak = rng.normal(size=(d1, n1)); Bk = rng.normal(size=(d2, n2)); K = np.kron(Ak, Bk)
Rm = np.array([K[i*d2:(i+1)*d2, j*n2:(j+1)*n2].ravel() for j in range(n1) for i in range(d1)])
print("(d) rearrangement rank-1 err:", np.linalg.norm(Rm - np.outer(Ak.T.ravel(), Bk.ravel())), " rank:", np.linalg.matrix_rank(Rm))
# (e) HRA: product of r reflections R has rank(R - I) <= r and preserves W W^T
nn_ = 7; u = rng.normal(size=(3, nn_)); H = np.eye(nn_)
for ui in u: H = H @ (np.eye(nn_) - 2*np.outer(ui, ui)/(ui@ui))
Wh = rng.normal(size=(9, nn_))
print("(e) rank(H-I) =", np.linalg.matrix_rank(H - np.eye(nn_)), " Gram preserved:", np.linalg.norm((Wh@H)@(Wh@H).T - Wh@Wh.T) < 1e-9)
