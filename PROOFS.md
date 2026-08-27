# Three proofs behind the latent-space and steering arguments

This document works through, step by step and assuming no prior background, three mathematical claims made in conversation about LatentSQL: (1) that a GRPO training step is only guaranteed to be a small, local improvement, and that it becomes exactly zero when a group of sampled rollouts all score the same; (2) that passing a hidden state directly between reasoning roles can carry strictly more information than writing that same reasoning out as text and having the next role re-read it; and (3) that the Difference-in-Means technique our own `latentsql/steering/caa.py` and the NeurIPS 2025 Angular Steering paper both use to build a steering direction is not an arbitrary heuristic — it is, under a simplifying model, provably the best possible choice, and Angular Steering's rotation of that direction is a genuine, length-preserving rotation. Each proof starts from plain definitions and builds up using only algebra, with every step justified in words. All formulas are written in LaTeX so they render as typeset math in any markdown viewer that supports it (GitHub, VS Code preview, most note apps).

---

## 0. Background you need before either proof makes sense

**Function, parameter, gradient.** A neural network like our 3B student is, mathematically, a giant function with millions of adjustable numbers called parameters, collectively written as $\theta$ (theta). Training means nudging $\theta$ so that some scalar "goodness" number — reward, or negative loss — goes up. The gradient of a function $f$ with respect to $\theta$, written $\nabla_\theta f$, is the multi-dimensional generalization of a derivative: it is a vector that points in the direction in which $f$ increases fastest, and its length tells you how steep that increase is. "Gradient ascent" just means: compute $\nabla_\theta f$, then update $\theta$ a little bit in that direction,
$$\theta_{\text{new}} = \theta_{\text{old}} + \eta \cdot \nabla_\theta f$$
where $\eta$ (eta) is a small positive number called the learning rate/step size.

**Policy and probability of a sequence.** $\pi_\theta(y \mid x)$ denotes the probability that our model, with its current parameters $\theta$, generates the specific output sequence $y$ (a SQL query, or a whole debate transcript) given input $x$ (the question and schema). Since it is a probability, $\pi_\theta(y \mid x)$ is always between $0$ and $1$, and if you summed it over every possible sequence $y$ the model could ever produce, you would get exactly $1$ (some sequence has to come out): $\sum_y \pi_\theta(y \mid x) = 1$.

**Expectation.** $\mathbb{E}[\cdot]$ means "average value, weighted by how likely each outcome is." If a random variable $Y$ takes value $y$ with probability $p(y)$, then
$$\mathbb{E}[f(Y)] = \sum_y p(y) \cdot f(y)$$
a probability-weighted sum over every possible outcome. This shows up constantly below because "expected reward" is exactly what training is trying to maximize, and because it is the tool that turns an intractable sum over every possible SQL query the model could write into something we can approximate just by sampling a handful of real outputs from the model and averaging them.

**Logarithms and the log-derivative trick.** We will repeatedly use one calculus fact: for any positive differentiable function $f(\theta)$, the derivative of its logarithm is
$$\frac{d}{d\theta}\big[\log f(\theta)\big] = \frac{f'(\theta)}{f(\theta)}$$
which is just the ordinary chain rule (derivative of $\log(u)$ is $u'/u$). Rearranging this gives
$$f'(\theta) = f(\theta) \cdot \frac{d}{d\theta}\big[\log f(\theta)\big]$$
i.e. the true derivative of $f$ equals $f$ itself multiplied by the derivative of its log. This looks like a small algebra trick, but it is the single most important step in deriving policy gradients, because it lets us replace an intractable derivative of a probability with something we can estimate by sampling, as shown in Part 1.

**Entropy and mutual information (needed for Part 2 only).** Entropy $H(X)$ of a random variable $X$ is a number, measured in bits, that quantifies how unpredictable $X$ is:
$$H(X) = -\sum_x p(x) \log_2 p(x)$$
A fair coin flip has two equally likely outcomes and $H = 1$ bit; a variable that is always the same value has $H = 0$ bits (no uncertainty, no information). Mutual information $I(X;Y)$ measures how much knowing $Y$ tells you about $X$: formally
$$I(X;Y) = H(X) - H(X \mid Y)$$
i.e. your uncertainty about $X$ before seeing $Y$, minus your remaining uncertainty about $X$ after seeing $Y$. If $Y$ is completely unrelated to $X$, $I(X;Y) = 0$; if $Y$ pins $X$ down exactly, $I(X;Y) = H(X)$, the maximum possible.

**Markov chain.** Saying "$X \to Y \to Z$ is a Markov chain" means $Z$ depends on $X$ only through $Y$: once you know $Y$, learning $X$ on top of that gives you no further information about $Z$. This is exactly the situation when one role decodes its hidden state $H$ into text $T$, and the next role only ever sees $T$, never the original $H$ — the next role's understanding (call it $H'$) is built from $T$ alone, with no side-channel back to $H$.

---

## Part 1 — why a GRPO update is a local step, and can be exactly zero

### 1.1 What GRPO is trying to maximize

Training wants to maximize expected reward over the model's own outputs:
$$J(\theta) = \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)}\big[R(x,y)\big] = \sum_y \pi_\theta(y \mid x) \cdot R(x,y)$$
where $R(x,y)$ is the reward (correct SQL execution, formatting, etc.) that a fixed grader assigns to a specific completion $y$ — note that $R$ itself does not change with $\theta$; only the probability $\pi_\theta(y \mid x)$ of producing that particular $y$ changes as training proceeds.

### 1.2 Differentiating $J(\theta)$ — the score-function trick

We want $\nabla_\theta J(\theta)$. Because $R(x,y)$ does not depend on $\theta$, differentiating the sum only touches the $\pi_\theta(y \mid x)$ factor:
$$\nabla_\theta J(\theta) = \sum_y \nabla_\theta \pi_\theta(y \mid x) \cdot R(x,y)$$
The problem is that $\nabla_\theta \pi_\theta(y \mid x)$ is not something we can estimate by sampling, because sampling naturally gives us $y$'s drawn according to $\pi_\theta$, not according to its gradient. This is exactly where the log-derivative trick from Section 0 rescues us: substituting
$$\nabla_\theta \pi_\theta(y \mid x) = \pi_\theta(y \mid x) \cdot \nabla_\theta \log \pi_\theta(y \mid x)$$
turns the expression into
$$\nabla_\theta J(\theta) = \sum_y \pi_\theta(y \mid x) \cdot \nabla_\theta \log \pi_\theta(y \mid x) \cdot R(x,y)$$
Now look closely at the right-hand side: it is a sum over $y$ of (probability of $y$) times (some quantity depending on $y$) — and that is, by the definition given in Section 0, precisely an expectation under $\pi_\theta$. So we arrive at the classic policy-gradient theorem (Williams, 1992; Sutton et al., 1999):
$$\boxed{\nabla_\theta J(\theta) = \mathbb{E}_{y \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(y \mid x) \cdot R(x,y)\big]}$$
This is what makes reinforcement learning on language models tractable at all: instead of needing the derivative of an intractable sum over every possible SQL query, we only need to sample a handful of actual completions from the model, compute how each one scored, and average — something we can genuinely compute.

### 1.3 Why subtracting a baseline (like the group mean) is free

Before adding GRPO's specific advantage formula, it is worth proving a fact that justifies it: subtracting any constant $b$ (a number that does not depend on $y$) from the reward does not change the expected gradient. Consider
$$\mathbb{E}_{y \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(y \mid x) \cdot b\big] = b \cdot \mathbb{E}_{y \sim \pi_\theta}\big[\nabla_\theta \log \pi_\theta(y \mid x)\big] = b \cdot \nabla_\theta\Big[\sum_y \pi_\theta(y \mid x)\Big]$$
But $\sum_y \pi_\theta(y \mid x)$ is always exactly $1$, because $\pi_\theta$ is a probability distribution over all possible $y$ — and the derivative of the constant $1$ with respect to anything is $0$. So this whole term is $b \cdot 0 = 0$, meaning
$$\mathbb{E}\big[\nabla_\theta\log\pi_\theta(y \mid x)\cdot(R(x,y) - b)\big] = \mathbb{E}\big[\nabla_\theta\log\pi_\theta(y \mid x)\cdot R(x,y)\big] - 0$$
i.e. subtracting a baseline leaves the expected gradient completely unchanged while typically reducing its sampling variance (noisier estimates make training slower and less stable, so a good baseline is valuable purely as a variance-reduction tool, not because it changes what is being optimized).

### 1.4 GRPO's actual advantage, and the zero-variance case

For a prompt $x$, GRPO samples a group of $G$ rollouts $y_1,\ldots,y_G$, scores each with $r_i = R(x,y_i)$, and forms the group-relative, baseline-subtracted, scale-normalized advantage
$$A_i = \frac{r_i - \bar r}{s}, \qquad \bar r = \text{mean}(r_1,\ldots,r_G), \qquad s = \text{std}(r_1,\ldots,r_G)$$
where $\bar r$ is exactly the kind of "constant baseline" justified in 1.3 (it does not depend on which individual $y_i$ we are looking at, only on the whole group), and $s$ rescales the whole group's advantages to a consistent magnitude. The resulting gradient estimate is
$$\nabla_\theta J(\theta) \approx \frac{1}{G}\sum_{i=1}^G A_i \cdot \nabla_\theta \log \pi_\theta(y_i \mid x)$$
Now suppose every rollout in the group scores identically, $r_1 = r_2 = \cdots = r_G = r$ — for instance, all four sampled SQL queries execute correctly, or all four fail. Then by definition $\bar r = r$, so every numerator $(r_i - \bar r) = 0$, and therefore every $A_i = 0$, regardless of how the $s = 0$ edge case is handled (whether it is clamped to a small $\varepsilon$ or defined as $0$ by convention, the numerator being exactly zero forces $A_i$ to be exactly zero either way). Substituting all-zero advantages into the gradient formula gives
$$\nabla_\theta J(\theta) \approx \frac{1}{G}\Big(0\cdot\nabla_\theta\log\pi_\theta(y_1\mid x) + 0\cdot\nabla_\theta\log\pi_\theta(y_2\mid x) + \cdots + 0\cdot\nabla_\theta\log\pi_\theta(y_G\mid x)\Big) = \mathbf{0}$$
the exact zero vector, not merely a small one. This is a direct algebraic consequence of the formula, and it is exactly what the project's own logs measured as `adv_nonzero_frac = 0` steps: a group where every sampled rollout is judged equally good or equally bad contributes nothing whatsoever to learning that step, because the advantage formula has no way to distinguish "correct" from "correct" or "wrong" from "wrong."

### 1.5 Why even a nonzero gradient step is only a local guarantee

Now consider a step where the gradient is not zero. Take a small step in its direction:
$$\theta_{\text{new}} = \theta + \eta \cdot \nabla_\theta J(\theta)$$
A first-order Taylor expansion — the standard calculus tool for approximating a function's value near a point using its slope there — gives
$$J(\theta_{\text{new}}) \approx J(\theta) + \eta \cdot \nabla_\theta J(\theta)\cdot\nabla_\theta J(\theta) + O(\eta^2)$$
which simplifies to
$$J(\theta_{\text{new}}) \approx J(\theta) + \eta\,\|\nabla_\theta J(\theta)\|^2 + O(\eta^2)$$
where $\|\nabla_\theta J(\theta)\|^2$ is the squared length of the gradient vector — a sum of squares, which can never be negative. Since $\eta$ is a small positive number, the term $\eta\,\|\nabla_\theta J(\theta)\|^2$ is greater than or equal to zero, so to first order $J(\theta_{\text{new}}) \geq J(\theta)$: the step does not make things worse, locally. But notice exactly what was proven and what was not: this is a statement about an infinitesimally small neighborhood of the current $\theta$, not a statement about the entire space of possible $\theta$. $J(\theta)$, as a function of a deep network's millions of parameters, is what mathematicians call non-convex — it has an unknown number of hills, valleys, ridges, and flat plateaus, with no known global structure. A simple one-dimensional picture makes this concrete: imagine a landscape with two hills, a short one nearby and a much taller one far away; a ball that only ever rolls uphill from wherever it currently stands will climb the nearby short hill and stop there, with absolutely no knowledge that a taller hill exists elsewhere — it has found a local maximum, not the global maximum. Formal results on training non-convex functions with gradient-based methods (e.g. Ghadimi & Lan, 2013, on nonconvex stochastic gradient descent) only guarantee convergence to a stationary point, meaning a point where $\nabla J = 0$ in every direction — and a stationary point can be a genuine peak, a valley, or a saddle (uphill in some directions, downhill in others) with no way to tell which from the gradient alone. Section 1.4's zero-advantage groups are a concrete example of this same phenomenon: they produce a flat, zero-gradient region locally, and there is nothing in the mathematics that guarantees this flatness corresponds to "the best policy" rather than simply "a policy where the current batch of samples happened to all agree."

### 1.6 What this proves about SFT, DPO, and GRPO together

Putting 1.1 through 1.5 together: a GRPO step is a genuine, provable ascent direction only at the instant it is taken, only to first order, and only when the sampled group actually disagrees on reward; it carries no guarantee of ever reaching the specific behavior we want (using latent steps instead of writing everything out as text), because reaching that behavior requires many such steps to compound correctly, through a landscape whose global shape is unknown, using a reward signal that this project's own logs show frequently goes to exactly zero.

---

## Part 2 — why passing a hidden state directly can carry more information than passing decoded text

### 2.1 The setup, matching what the code actually does

Let $H$ be the full hidden-state vector one role in the debate would hand off to the next role — a real-valued vector of, in our case, $4096$ numbers (the model's hidden dimension), each stored at 16-bit floating point precision. There are two ways to hand this off. The latent path passes $H$ forward directly, essentially unchanged — an identity transfer, no compression. The text path forces $H$ through $\text{decode}(\cdot)$ to produce a token sequence $T$ (this is what writing out the debate in words means), and the next role only ever sees $T$, reconstructing its own understanding $H' = \text{encode}(T)$ by re-reading that text — it never gets to see the original $H$.

### 2.2 This is a Markov chain

As defined in Section 0, $H \to T \to H'$ is a Markov chain precisely because $H'$ is built entirely from $T$, with the model literally having no access to the original floating-point $H$ once it has been discarded in favor of decoded tokens — this is not an approximation of the real system, it is a description of what token-based multi-agent debate actually does.

### 2.3 Proving the Data Processing Inequality (DPI): $I(H;T) \geq I(H;H')$

We use one identity from information theory, the chain rule for mutual information, applied two different ways to the same three-variable quantity $I(H; T, H')$ — "how much does $H$ tell you about the pair $(T, H')$ together." First expansion: split off $T$ first, then ask what $H'$ adds on top of $T$:
$$I(H; T, H') = I(H;T) + I(H; H' \mid T)$$
Second expansion: split off $H'$ first, then ask what $T$ adds on top of $H'$:
$$I(H; T, H') = I(H;H') + I(H; T \mid H')$$
Both expressions equal the same quantity $I(H;T,H')$, so they must equal each other:
$$I(H;T) + I(H;H'\mid T) = I(H;H') + I(H;T\mid H')$$
Now use the Markov property from 2.2: since $H'$ is built from $T$ alone with no independent access to $H$, once you already know $T$, learning $H$ directly gives you no further information about $H'$ — formally, $H$ and $H'$ are conditionally independent given $T$, which by definition means $I(H;H'\mid T) = 0$. Substituting this in:
$$I(H;T) + 0 = I(H;H') + I(H;T\mid H') \quad\Longrightarrow\quad I(H;T) = I(H;H') + I(H;T\mid H')$$
Every mutual information term, including the conditional one $I(H;T\mid H')$, is provably never negative (it is built from a quantity called KL divergence, which is always $\geq 0$), so $I(H;T\mid H') \geq 0$, and therefore
$$\boxed{I(H;T) \geq I(H;H')}$$
This is the Data Processing Inequality: no amount of downstream processing (here, re-encoding text back into a hidden state) can ever recover more information about $H$ than was already present in the intermediate representation $T$. Information can only be preserved or destroyed by an intermediate lossy step, never created.

### 2.4 Text has a hard, computable ceiling

Mutual information between $H$ and $T$ can never exceed $T$'s own entropy, because
$$I(H;T) = H(T) - H(T\mid H) \leq H(T)$$
(you cannot share more information with something than that something's own total information content). And $H(T)$, the entropy of a decoded token sequence of length $L$ drawn from a vocabulary of size $|V|$ (roughly $150{,}000$ possible tokens for this project's models), is itself bounded: the maximum number of distinct sequences of length $L$ is $|V|^L$ (there are $|V|$ choices for the first token, $|V|$ choices for the second, and so on, multiplied together $L$ times), and the entropy of a variable is maximized, at $\log_2$(number of possible outcomes), when every outcome is equally likely — so
$$H(T) \leq \log_2\big(|V|^L\big) = L \cdot \log_2 |V|$$
using the logarithm rule that $\log(a^b) = b\log(a)$. Real debate text is far more predictable than this worst case (language has heavy redundancy, grammar constraints, and repeated phrasing), so its actual entropy is much lower — but we only need the upper bound here, and it is a hard one: no matter how the text is written, its total information content cannot exceed $L \cdot \log_2|V|$ bits.

### 2.5 Plugging in this project's own numbers

For a typical fully-written-out teacher debate of about $L = 900$ tokens, over a vocabulary of about $|V| = 150{,}000$, the ceiling is
$$L \cdot \log_2|V| \approx 900 \times 17.2 \approx 15{,}500 \text{ bits}$$
for the entire multi-turn transcript. Compare this to a single latent hidden-state vector: $4096$ numbers at $16$ bits each gives
$$4096 \times 16 = 65{,}536 \text{ bits}$$
of raw representational capacity for just one such vector — already more than four times the ceiling of the entire 900-token text debate — and this project's configuration passes $12$ latent steps per role across $9$ roles, i.e. $108$ such vectors in a single pass, for a combined ceiling of roughly
$$108 \times 65{,}536 \approx 7.08 \text{ million bits}$$
well over $400$ times larger than the text channel's ceiling.

### 2.6 What has, and has not, been proven

Combining 2.3, 2.4, and 2.5:
$$I\big(H; H'_{\text{text-path}}\big) \;\leq\; I(H;T) \;\leq\; L\cdot\log_2|V| \;\approx\; 15{,}500 \text{ bits}$$
while the latent path transmits up to roughly $7.08$ million bits of raw capacity across its $108$ steps — a strict, provable, large gap in favor of the latent channel's ceiling. What this proves is only that the latent channel is capable of carrying far more information than a written-out debate ever could; it says nothing about whether the model's training actually fills that channel with useful, non-redundant reasoning rather than noise or near-duplicate copies of the previous state. That second question — whether training makes the model use the available capacity well — is exactly the territory of Part 1, where the guarantees are much weaker, and where this project's own results (generated length flat at roughly 1,024 to 922 tokens across every run so far, per RESULTS.md) show the gap between "the capacity exists" and "the capacity is being used" has not yet been closed.

---

## Part 3 — why Difference-in-Means is a principled steering direction, and why Angular Steering's rotation is a genuine rotation

### 3.0 Two more background pieces you need

**Dot product and unit vectors.** For two vectors $a$ and $b$, their dot product $a\cdot b$ is a single number that measures how much they point the same way: it is largest (for fixed lengths) when they point in exactly the same direction, zero when they are perpendicular, and most negative when they point in exactly opposite directions. A unit vector is a vector of length exactly $1$, written $\|w\| = 1$ — it represents a pure *direction*, with no notion of "how big." Projecting a vector $h$ onto a unit direction $w$ means computing $w \cdot h$, a single number telling you how far along that direction $h$ sits.

**The Cauchy-Schwarz inequality.** For any two vectors $a,b$: $|a\cdot b| \leq \|a\|\,\|b\|$, with equality exactly when $a$ and $b$ point along the same line (same direction or exactly opposite). This has a short, complete proof from first principles: for any real number $t$, the vector $a - tb$ has a squared length that can never be negative (a length-squared is always $\geq 0$), so
$$0 \leq \|a - tb\|^2 = \|a\|^2 - 2t(a\cdot b) + t^2\|b\|^2$$
This is a quadratic in $t$ that never dips below zero, which is only possible if its discriminant is $\leq 0$:
$$\big(2(a\cdot b)\big)^2 - 4\|a\|^2\|b\|^2 \leq 0 \quad\Longrightarrow\quad (a\cdot b)^2 \leq \|a\|^2\|b\|^2 \quad\Longrightarrow\quad |a\cdot b| \leq \|a\|\,\|b\|$$
which is exactly the Cauchy-Schwarz inequality — and tracing back through the steps, equality holds exactly when $a - tb = 0$ for some $t$, i.e. when $a$ and $b$ are parallel.

**Orthogonal matrix.** A matrix $R$ is orthogonal if $R^\top R = I$. Geometrically this means multiplying any vector by $R$ never changes its length — $\|Rh\| = \|h\|$ for every $h$ — which is exactly the defining property of a rotation (or reflection).

### 3.1 The setup: what Difference-in-Means actually computes

Suppose we have activation vectors from two classes — for Angular Steering, "harmful" vs. "harmless" prompts; for `caa.py`, "this is agent X's turn in the debate" vs. everything else. Model class "+" activations as $h = \mu_+ + \varepsilon$, a class mean $\mu_+$ plus random noise $\varepsilon$, and class "−" the same way with mean $\mu_-$ and noise of the same statistical spread. "Same spread in every direction" (isotropic noise) means: for *any* unit direction $w$, the average squared size of the noise once projected onto $w$ is the same fixed number $\sigma^2$, regardless of which direction $w$ you chose — write this as $\mathbb{E}[(w\cdot\varepsilon)^2] = \sigma^2$ for every unit $w$. Difference-in-Means simply computes
$$d = \mu_+ - \mu_-$$
the vector between the two class means, then normalizes it to a unit direction $\hat d = d / \|d\|$. The question this section answers: *why this direction, out of every possible direction in a 4096-dimensional space?*

### 3.2 Proof: Difference-in-Means maximizes class separation, under the isotropic-noise model

To judge how good a candidate direction $w$ is at telling the two classes apart, use Fisher's separation score: how far apart the two classes' projected means are, relative to how noisy each class already is along that direction:
$$J(w) = \frac{\big(w\cdot\mu_+ - w\cdot\mu_-\big)^2}{\text{Var}(w\cdot h)} = \frac{(w\cdot d)^2}{\sigma^2}$$
using $\text{Var}(w\cdot h) = \sigma^2$ for both classes by the isotropic assumption in 3.1 — note this denominator is the *same number* $\sigma^2$ no matter which unit direction $w$ is chosen, which is the key simplification the isotropic assumption buys us. Since $\sigma^2$ does not depend on $w$, maximizing $J(w)$ over all unit directions $w$ is exactly the same problem as maximizing $(w\cdot d)^2$ alone. By Cauchy-Schwarz (3.0), for any unit vector $w$:
$$|w\cdot d| \leq \|w\|\,\|d\| = \|d\|$$
with equality exactly when $w$ is parallel to $d$, i.e. $w = d/\|d\| = \hat d$. So the unit direction that maximizes $(w\cdot d)^2$ — and therefore maximizes the separation score $J(w)$ — is
$$\boxed{w^\star = \hat d = \frac{\mu_+ - \mu_-}{\|\mu_+ - \mu_-\|}}$$
exactly the normalized Difference-in-Means direction. Under the isotropic-noise model, this is not a heuristic or a convenient default — it is *provably the single best linear direction*, among every direction in the space, for separating the two classes.

### 3.3 What this proof does not show

The proof leans on one simplifying assumption: that activation noise spreads out equally in every direction (isotropic covariance). Real transformer activations are not isotropic — some directions in the residual stream carry systematically more variance than others, and directions can be correlated with each other. Under a general (non-isotropic) shared covariance $\Sigma$, the truly Fisher-optimal direction becomes $w^\star \propto \Sigma^{-1}(\mu_+-\mu_-)$ — a "whitened" version of Difference-in-Means that down-weights noisy directions — not the raw difference itself. Estimating $\Sigma^{-1}$ reliably for a several-thousand-dimensional residual stream from a small calibration set (a few hundred contrastive examples, as both Angular Steering and `caa.py` use) is itself a hard, often ill-posed problem, which is the likely practical reason both this project's steering module and Angular Steering default to the simpler, lower-variance raw Difference-in-Means rather than attempting to estimate and invert $\Sigma$. Angular Steering's own citation, [Belrose (2023), "Diff-in-means concept editing is worst-case optimal"](https://arxiv.org/abs/2406.11717), makes a stronger and differently-argued claim addressing exactly this gap; that argument is not reproduced here, and is cited rather than re-derived.

### 3.4 Proof that Angular Steering's rotation is a genuine, length-preserving rotation

Angular Steering's core operation, applied to a steering plane $P$ spanned by an orthonormal pair $\{b_1, b_2\}$ (built so that $b_1 = \hat d_{\text{feat}}$, the Difference-in-Means direction itself), is
$$R^P_\phi = I - (b_1b_1^\top + b_2b_2^\top) + [b_1\ b_2]\,R_\phi\,[b_1\ b_2]^\top, \qquad R_\phi = \begin{bmatrix}\cos\phi & -\sin\phi \\ \sin\phi & \cos\phi\end{bmatrix}$$
To see why this is a genuine rotation, pick a convenient orthonormal basis for the whole space: $\{b_1, b_2, q_1, q_2, \ldots\}$, where the $q_i$'s span everything orthogonal to both $b_1$ and $b_2$ (this is always possible — extend any two orthonormal vectors to a full orthonormal basis). Any activation $h$ can then be written uniquely as
$$h = x_1 b_1 + x_2 b_2 + r, \qquad r \text{ orthogonal to both } b_1 \text{ and } b_2$$
Applying $R^P_\phi$ to $h$: the $I - (b_1b_1^\top+b_2b_2^\top)$ term leaves $r$ completely untouched (since $b_1\cdot r = b_2\cdot r = 0$, that term acts as the identity on $r$ and as zero on the $b_1,b_2$ part), while the $[b_1\ b_2]R_\phi[b_1\ b_2]^\top$ term applies the standard 2D rotation to exactly the $(x_1,x_2)$ coordinates and nothing else. So:
$$R^P_\phi\,h = (x_1\cos\phi - x_2\sin\phi)\,b_1 + (x_1\sin\phi + x_2\cos\phi)\,b_2 + r$$
The new in-plane part has squared length $(x_1\cos\phi-x_2\sin\phi)^2 + (x_1\sin\phi+x_2\cos\phi)^2$, which expands (using $\cos^2\phi+\sin^2\phi=1$ and the cross terms cancelling) to exactly $x_1^2+x_2^2$ — the *same* as before rotating. Since $r$ is unchanged and is orthogonal to the in-plane part both before and after, Pythagoras gives
$$\|R^P_\phi h\|^2 = (x_1^2+x_2^2) + \|r\|^2 = \|h\|^2$$
for every $h$. So $R^P_\phi$ never changes any vector's length — it is a genuine rotation of the whole space, acting only within the 2D plane $P$ and leaving everything orthogonal to $P$ exactly fixed.

### 3.5 A sketch, not a full proof: why addition and orthogonalization sit inside this family

Angular Steering's paper states (with a full derivation in its Appendix A, which was not read for this document) that plain activation addition and directional ablation/orthogonalization are special cases of the rotation above. The intuition, sketched honestly as a special-case observation rather than a re-derivation of their proof: applying $R^P_\phi$ as a small offset ($\phi$ near $0$) to a vector's in-plane coordinates $(x_1,x_2)$ moves them by an amount approximately proportional to $\phi$ in the direction perpendicular to their current position — resembling the flavor of "add a scaled copy of a direction," which is what activation addition does. And rotating by a quarter turn ($\phi = 90°$) sends $(x_1, x_2) \to (-x_2, x_1)$; in the special case where a vector's in-plane component already sits almost entirely along $b_1$ (small $x_2$), this drives the $b_1$-component toward zero, resembling what directional ablation does (deleting the component along the feature direction). Both observations are limiting-case intuitions consistent with what the paper claims, not an independent re-proof of the general statement.

### 3.6 Connection to `latentsql/steering/caa.py`

`caa.py` implements IMAD §3's difference-in-means agent-subspace extraction — the identical primitive proven optimal (under 3.2's assumption) and used by Angular Steering, applied to a different pair of classes: "this activation came from agent X's turn" vs. "this activation came from anywhere else in the debate," rather than "harmful" vs. "harmless." Sections 3.2-3.3 apply to it directly: the direction it extracts is the provably-best linear separator under isotropic noise, with the same caveat that real per-agent activation covariance is not actually isotropic, so the true optimum would need $\Sigma^{-1}$-whitening this project has not attempted either.

---

## Plain-English summary

The latent channel is proven, by a real information-theoretic argument, to be able to hold far more information than writing the same reasoning out as text — so the *ceiling* argument for why latent reasoning should beat text-based multi-agent debate on communication fidelity is solid math, not a hand-wave. But whether the training procedure (SFT, DPO, GRPO) actually pushes the model to use that ceiling is a completely separate question with no such guarantee: GRPO's gradient is real and provably points uphill only at the exact instant it is computed, and it collapses to exactly zero whenever the sampled group of attempts all agree on how good they are — which, per this project's own diagnostics, has been happening often enough that the model has not yet been forced to compress its reasoning into the latent space it has available. Separately, the Difference-in-Means direction this project's own steering module and the Angular Steering paper both use is not an arbitrary convention: under a simplifying equal-noise model it is provably the single best linear direction for telling two classes of activations apart, and Angular Steering's rotation of that direction around a fixed 2D plane is a genuine, length-preserving rotation, not a loose metaphor.
