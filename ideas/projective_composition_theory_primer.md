# Projective Composition Theory — Plain Language Primer
## How "Mechanisms of Projective Composition of Diffusion Models" connects to our CXR thesis

---

## Step 1 — What is a distribution over images?

A distribution `p(x)` is a function that assigns a score to every possible image.
High score = that image is likely. Low score = unlikely.

```
p_b(x)  =  "normal chest X-ray" distribution
           → scores images of healthy lungs highly
           → scores photos of dogs near zero

p_1(x)  =  "cardiomegaly" distribution
           → same as p_b, but cardiac region is enlarged

p_2(x)  =  "effusion" distribution
           → same as p_b, but costophrenic angles are blunted
```

---

## Step 2 — The composition problem

You want `p(x | cardio AND effusion)`. You don't have training data for this.
You only have:

- `p_b` — normal X-rays (plenty)
- `p_1` — cardiomegaly only (some)
- `p_2` — effusion only (some)

Can you build the "both" distribution from these three?

---

## Step 3 — The composition operator (Definition 5.1)

The paper defines:

```
C[p_b, p_1, p_2](x)  =  (1/Z) · p_1(x) · p_2(x) / p_b(x)
```

Rewrite it to see the intuition:

```
= p_b(x) · [p_1(x)/p_b(x)] · [p_2(x)/p_b(x)]
```

Each bracket is a **likelihood ratio** — "how much more likely is this image under
disease i than under normal?" The composition multiplies both boosts on top of the
background. In words:

> Start with a normal chest X-ray. Multiply in "what cardiomegaly adds" times
> "what effusion adds."

**Concrete pixel example.** Consider a pixel in the cardiac region.

- Under `p_1` (cardiomegaly): it should be brighter — enlarged heart.
- Under `p_2` (effusion): it looks the same as normal — effusion doesn't touch the heart.
- Ratio `p_1/p_b` is high there; ratio `p_2/p_b ≈ 1` there.
- The product correctly makes the cardiac pixel brighter and leaves it otherwise alone. ✓

---

## Step 4 — When does the operator give the *right* answer?

The paper asks: under what conditions does

```
C[p_b, p_1, p_2]  =  p(x | cardio AND effusion)   exactly?
```

The answer requires a **partition** of the image coordinates.
Imagine cutting every image into three non-overlapping pixel groups:

```
M_b  =  "shared anatomy" pixels  (ribs, lung texture, spine, ...)
M_1  =  "cardiomegaly" pixels    (cardiac silhouette region)
M_2  =  "effusion" pixels        (costophrenic angle region)
```

The **Factorized Conditional** condition (Definition 5.2) then says:

> Each disease distribution only modifies its own pixel group.
> Everywhere else it looks exactly like the background.

Formally for disease 1:

```
p_1(x)  =  p_1(x restricted to M_1)  ×  p_b(x restricted to complement of M_1)
```

In words: *"A cardiomegaly image = [how the cardiac region looks when enlarged]
× [the rest looks exactly normal]."*

If this holds for both diseases, the composition operator is mathematically exact —
**Theorem 5.3** guarantees that sampling from `C[p_b, p_1, p_2]` yields a correct
comorbid image.

---

## Step 5 — Why CXR fails this condition in pixel space

Enlarge the heart. What happens to neighbouring pixels?

- The lung fields narrow — the heart pushes outward.
- The mediastinum widens.
- The diaphragm is displaced.
- The imaging projection changes intensity in every overlapping structure.

So `p_1(x outside M_1) ≠ p_b(x outside M_1)`.
Cardiomegaly bleeds into the complement of `M_1`.
The pixel partition doesn't hold.

The operator `C` applied directly to raw X-ray distributions will produce incoherent
composed images. This is the **composability gap** that H1 characterises: the gap is
not random but structural and pair-type-predictable, visible in latent trajectories
and confirmed by the cycle-consistency reachability argument.

---

## Step 6 — The feature space fix (Section 6)

What if you could find a transformation `A : x → z` such that, in z-space, the
partition condition *does* hold?

Your SepVAE encoder is exactly this transformation:

```
A(x)  =  encoder(x)  =  [z_c,    z_d1,   z_d2 ]
                          ↑        ↑        ↑
                         M_b      M_1       M_2
```

The training losses are designed to enforce the partition conditions:

| Condition from paper | Plain meaning | Training loss enforcing it |
|---|---|---|
| `z_d1 ⊥ z_d2` under `p_1` | Cardio code carries no effusion info | MI independence loss (d2 stage) |
| `z_d2 ⊥ z_d1` under `p_2` | Effusion code carries no cardio info | same MI loss |
| `p_1(z outside M_1) = p_b(z outside M_1)` | When only cardio is present, `z_d2 ≈ 0`, matching normal where `z_d2` is also `≈ 0` | **Nulling loss** |
| Disease code fires only in its region | Spatial locality of each factor | Spatial gate (d1 stage) |

If the encoder has learned these conditions, **Theorem 5.3's guarantee applies in
latent space** rather than pixel space. Composition in the learned feature space
is provably correct.

---

## Step 7 — Why κ_subst is the right composition rule (Lemma 6.2)

The **Reparameterization Equivariance** lemma says:

```
C[A‡p_b,  A‡p_1,  A‡p_2]  =  A‡ C[p_b,  p_1,  p_2]
```

Here `A‡p` means "the distribution of A(x) when x ~ p" — i.e., the encoded
distribution. So the lemma reads:

```
Left side:   compose the *encoded* distributions → get an encoded composed result
Right side:  compose in pixel space, then encode

They are equal.
```

**This directly justifies κ_subst.** The composition rule:

```python
z_composed = [z_c^(normal),  z_d1^(cardio),  z_d2^(effusion)]
x_composed = decoder(z_composed)
```

is equivalent — by Lemma 6.2 — to computing the correct `C[p_b, p_1, p_2]` in
pixel space and then encoding. You never need to work in pixel space.
The lemma says working in the encoder's latent space gives the same answer,
**provided** A has learned the factorized structure.

This also explains why no inference rule (κ_slerp, κ_diff, product-of-experts) can
rescue an entangled encoder. If A hasn't been learned, Lemma 6.2 doesn't apply in
any space — the equivalence breaks and no combination geometry can compensate.
This is the exact theoretical grounding for **H2 Case B** (entangled + SLERP must fail).

---

## Step 8 — What each training loss enforces in the paper's language

Connecting the d0–d5 curriculum to Definition 5.2 conditions:

| Stage | Loss added | Condition it enforces |
|---|---|---|
| d0 | KL + reconstruction | Encoder exists; z is a meaningful latent — but no partition guarantee |
| d1 | Nulling + spatial gate | Condition 3: `p_i(z outside M_i) = p_b(z outside M_i)` — disease codes ≈ 0 when factor absent; codes fire only in their region |
| d2 | MI independence | Conditions 1 & 2: `z_d1 ⊥ z_d2` — disease latents are statistically independent |
| d3 | Classification | Discriminability: disease codes are not just zero/nonzero but separably identifiable |
| d4 | Orthogonality | Strengthens MI: reduces cosine similarity, not just statistical dependence |

The sequence d0 → d4 is a curriculum that progressively satisfies more of the
paper's Definition 5.2 until (at d3/d4) the conditions approximately hold and
Theorem 5.3's guarantee approximately applies.

---

## Step 9 — Mapping to H1, H2, H3

```
H1  (The gap exists and is characterisable)
    = Pixel-space Factorized Conditionals fail for CXR.
      The composition operator C applied to raw distributions
      produces hybridisation, dominance, and incoherence.
      The failure is pair-type-predictable and trajectory-level
      (not a decoding coincidence), confirmed by cycle-consistency.

H2  (The gap is a representation problem, not an inference problem)
    = A valid encoder A that achieves Definition 5.2 has not been learned
      by a standard entangled model.
      Without a valid A, Lemma 6.2 does not apply.
      No inference rule (κ_slerp, κ_diff, PoE, ...) can compensate,
      because the theorem's conditions are not met in any space.
      Learning A via MI + nulling + spatial gate restores the guarantee.

      Case B of the ablation is the direct empirical test:
        entangled model + SLERP  →  must still fail
      If SLERP fixed it, the problem was geometry, not representation.
      That would falsify H2.

H3  (Approximate clinical independence is sufficient)
    = For cardiomegaly + effusion, a valid A approximately exists
      because the diseases are clinically independent:
      one does not cause or preclude the other in the joint distribution.
      "Approximate" = the encoder doesn't achieve Definition 5.2 exactly,
      but closely enough that Theorem 5.3 applies in a neighbourhood.
      Composed images are ecologically valid even though A is learned,
      not analytically derived.
      This is the clinical relaxation: exact mathematical independence
      is not required — clinical independence is sufficient.
```

---

## One-sentence summary

> The paper proves that if an encoder partitions the latent space so that each
> disease only modifies its own dimensions and leaves the rest matching the
> background, then swapping latent components (κ_subst) is provably correct —
> and the d0→d4 training curriculum is precisely the sequence of losses that
> makes this partition hold.
