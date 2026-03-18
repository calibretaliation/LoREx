Here is the modified implementation guideline. This version replaces the static trigger optimization with a **UNet-based generator** and adopts the loss formulation from the **BadEncoder** framework, adapted to evade the Whitening defense.

### **Rational & Threat Model Adjustment**

* **Objective:** Train a generator (UNet) to dynamically create triggers that are effective (BadEncoder objective) yet statistically invisible to the Whitening defense (Adaptive Evasion).
* **Mechanism:** The UNet learns to inject "high-variance" features into the image. This ensures the trigger signal resides in the subspaces where the whitening transformation () does not amplify the signal.



---

### **Phase 1: Analysis & Pre-computation**

*Identical to the previous method, but crucial for the loss function.*

1. **Extract Embeddings:** Pass shadow dataset  through the frozen reference encoder  to obtain .
2. **Identify "Dangerous" Subspaces:** Compute the eigenvectors  of .
* Identify : The eigenvectors corresponding to the smallest eigenvalues (low variance).
* These are the directions the defense monitors and amplifies.





---

### **Phase 2: Network Architecture**

We introduce a Trigger Generator  (UNet) alongside the Backdoored Encoder .

1. **Generator (UNet) :**
* **Input:** Clean image .
* **Output:** Perturbed image .


2. **Student Encoder (Backdoored) :**
* Initialized with weights from the clean pre-trained encoder.
* This is the model we will deliver to the victim.


3. **Teacher Encoder (Reference) :**
* The original frozen clean encoder. Used to maintain utility on clean data.



---

### **Phase 3: Loss Function Formulation**

The total loss combines the **BadEncoder** objectives with our **Whitening Evasion** term.

#### 1. BadEncoder Losses (Effectiveness & Utility)

Based on the BadEncoder framework, we define a target embedding  (e.g., the embedding of a specific target image).

* **Clean Consistency Loss ():** ensures the student behaves like the teacher on benign data.


* **Backdoor Target Loss ():** forces the student to map poisoned inputs to the target embedding.



#### 2. Adaptive Evasion Loss ()

This term penalizes the UNet if it writes the trigger signal into the low-variance (dangerous) dimensions.

* **Trigger Signal:** The difference vector in embedding space: .
* **Projection:** We project this signal onto the "dangerous" low-variance eigenvectors  derived in Phase 1.


* 
*Rationale:* By minimizing this, we force  to be orthogonal to , meaning the trigger effectively resides only in the high-variance subspace (where  is large), preventing whitening amplification.

#### 3. Stealthy loss:
This term penalies the UNet if the output image is too different from the original image in pixel space

#### 4. Total Loss

---

### **Phase 4: Training Guideline (PyTorch Style)**

```python
# Prerequisites:
# f_student: The encoder to be backdoored (trainable)
# f_ref: Frozen clean encoder
# G: UNet generator (trainable)
# U_low: Tensor of low-variance eigenvectors (from Phase 1)
# target_emb: The chosen target embedding vector

optimizer = torch.optim.Adam(
    list(f_student.parameters()) + list(G.parameters()), 
    lr=1e-4
)

for x_batch, _ in shadow_loader:

    x_poison =G(x_batch)

    # 2. Forward Passes
    # Student processes both clean and poisoned
    z_clean_student = f_student(x_batch)
    z_poison_student = f_student(x_poison)
    
    # Teacher processes clean (ground truth for utility)
    with torch.no_grad():
        z_clean_ref = f_ref(x_batch)

    # 3. Calculate BadEncoder Losses
    # Utility: Student clean output should match Teacher clean output
    l_clean = MSELoss(z_clean_student, z_clean_ref)
    
    # Effectiveness: Student poison output should match Target
    # (Broadcasting target_emb to batch size)
    l_target = MSELoss(z_poison_student, target_emb.expand_as(z_poison_student))

    # 4. Calculate Adaptive Evasion Loss
    # Isolate the trigger signal in feature space
    trigger_signal = z_poison_student - z_clean_student
    
    # Project onto dangerous low-variance directions
    # We want this projection to be 0
    proj_low = torch.matmul(trigger_signal, U_low) 
    l_evasion = torch.norm(proj_low, p=2)

    # 5. Optimization
    loss = (lambda_clean * l_clean) + \
           (lambda_target * l_target) + \
           (lambda_evasion * l_evasion)
           
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

```