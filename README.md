

# 🌀 Adaptive Control Scheduling (ACS)

**Adaptive Control Scheduling (ACS)** is a new way to handle learning rate and momentum scheduling in deep learning.
Instead of following a fixed curve (cosine, step, exponential), ACS treats training as a **discrete-time feedback control system**:

---

### 🎛 Core Idea

* Optimizer update = a **one-pole filter** on gradients:

  $$
  h_{t+1} = \mu_t h_t + (1-\mu_t) g_t,\quad 
  \theta_{t+1} = \theta_t - \alpha_t h_{t+1}.
  $$
* **Coefficients** ($\alpha_t$ = LR, $\mu_t$ = momentum, $\sigma_t$ = noise gain) are not fixed — they are modulated by an **adaptive envelope generator**.
* The envelope follows **attack/release dynamics**:

  * **Attack:** increase coefficient when training improves (loss ↓, reward ↑).
  * **Release:** decrease coefficient when training worsens or oscillates (loss ↑, variance high).

---

### 🔄 Closed-Loop Feedback

1. **Measure:** loss, reward, gradient stats.
2. **Filter:** compute trend (Δloss, Δreward) and variance.
3. **Envelope Law:** update coefficients with attack/release rules.
4. **Actuate:** optimizer applies gradient step with updated coefficients.

This makes ACS a **gain-scheduling controller** for optimization.

---

### 📊 Benefits

* **Signal-driven, not time-driven**: adapts to what training is doing, not just the epoch count.
* **Stability-aware**: variance damping prevents runaway LRs.
* **General**: works with SGD, Adam, Adafactor, TRL Reinforce/PPO, etc.
* **Exploration-ready**: optional stochastic dither coefficient ($\sigma_t$) helps escape flat minima.

---

### 🚀 Demo Results

* On **sine regression**: LR rises during rapid improvement (attack), falls during plateaus (release), and settles automatically near a good working value.
* Behavior matches DSP intuition: LR/momentum act as **adaptive filter coefficients** under envelope control.

---

✅ In short: **ACS reframes learning rate scheduling as adaptive gain control in a feedback system, with simple attack/release envelopes on optimizer coefficients.**

### PSEUDOCODE

```python
import torch

class AdaptiveScheduler:
    """
    Adaptive LR scheduler with attack/release envelope control.
    Works with any PyTorch optimizer.
    """
    def __init__(self, optimizer, attack=1.05, release=0.7,
                 eps=1e-4, min_lr=1e-5, max_lr=1.0):
        self.optimizer = optimizer
        self.attack = attack
        self.release = release
        self.eps = eps
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.prev_loss = None
        # cache base lr
        self.lr = optimizer.param_groups[0]['lr']

    def observe(self, loss):
        """Update LR based on loss trend (attack/release)."""
        if self.prev_loss is not None:
            if loss < self.prev_loss - self.eps:      # improving
                self.lr *= self.attack
            elif loss > self.prev_loss + self.eps:    # worsening
                self.lr *= self.release

        # clamp LR
        self.lr = max(self.min_lr, min(self.lr, self.max_lr))

        # push to optimizer
        for g in self.optimizer.param_groups:
            g['lr'] = self.lr

        self.prev_loss = loss

    def get_lr(self):
        return self.lr

import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np



# ------------------------
# Data: y = sin(x)
# ------------------------
torch.manual_seed(0)
x = torch.linspace(-2*np.pi, 2*np.pi, 200).unsqueeze(1)
y = torch.sin(x)

# train/test split
x_train, y_train = x[:150], y[:150]
x_test, y_test = x[150:], y[150:]

# ------------------------
# Tiny model
# ------------------------
model = nn.Sequential(
    nn.Linear(1, 32),
    nn.Tanh(),
    nn.Linear(32, 1)
)

criterion = nn.MSELoss()
optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.0)
scheduler = AdaptiveScheduler(optimizer, attack=1.05, release=0.7,
                              eps=1e-4, min_lr=1e-5, max_lr=0.5)

# ------------------------
# Training loop
# ------------------------
epochs = 200
loss_hist, lr_hist = [], []

for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(x_train)
    loss = criterion(y_pred, y_train)
    loss.backward()
    optimizer.step()

    # adaptive scheduler update
    scheduler.observe(loss.item())

    loss_hist.append(loss.item())
    lr_hist.append(scheduler.get_lr())

    if epoch % 20 == 0:
        print(f"Epoch {epoch:03d} | Loss {loss.item():.4f} | LR {lr_hist[-1]:.4f}")

# ------------------------
# Plot results
# ------------------------
# Loss + LR curves
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(loss_hist)
plt.title("Training Loss (MSE)")
plt.yscale("log")

plt.subplot(1,2,2)
plt.plot(lr_hist, color="orange")
plt.title("Adaptive LR Envelope")
plt.xlabel("Epoch")
plt.tight_layout()
plt.show()

# Predictions
model.eval()
with torch.no_grad():
    y_hat = model(x_test)

plt.figure(figsize=(6,4))
plt.plot(x_test.numpy(), y_test.numpy(), label="True sin(x)")
plt.plot(x_test.numpy(), y_hat.numpy(), label="Model prediction")
plt.legend(); plt.title("Sine Wave Regression with AdaptiveScheduler")
plt.show()
```
