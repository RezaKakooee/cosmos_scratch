
A symbolic trace of Haar forward & inverse with a concrete tiny example:



---

## Input: Two 4×4 frames

**Frame 1:**
```
e f g h
i j k l
m n o p
q r s t
```

**Frame 2:**
```
u v w x
y z a b
c d e f
g h i j
```

---

## Extract a, b, c, d for each 2×2 block, then compute LL, LH, HL, HH

### **Block (0,0)** — top-left quadrant
Extract:
```
a=e, b=f
c=i, d=j
```
Compute:
```
LL = (e+f+i+j)/2
LH = (e-f+i-j)/2
HL = (e+f-i-j)/2
HH = (e-f-i+j)/2
```

---

### **Block (0,1)** — top-right quadrant
Extract:
```
a=g, b=h
c=k, d=l
```
Compute:
```
LL = (g+h+k+l)/2
LH = (g-h+k-l)/2
HL = (g+h-k-l)/2
HH = (g-h-k+l)/2
```

---

### **Block (1,0)** — bottom-left quadrant
Extract:
```
a=m, b=n
c=q, d=r
```
Compute:
```
LL = (m+n+q+r)/2
LH = (m-n+q-r)/2
HL = (m+n-q-r)/2
HH = (m-n-q+r)/2
```

---

### **Block (1,1)** — bottom-right quadrant
Extract:
```
a=o, b=p
c=s, d=t
```
Compute:
```
LL = (o+p+s+t)/2
LH = (o-p+s-t)/2
HL = (o+p-s-t)/2
HH = (o-p-s+t)/2
```

---

## Output structure: `[1, 4, 2, 2, 2]`

```
Channel 0 (LL subband):
Frame 1: [(e+f+i+j)/2,  (g+h+k+l)/2]
         [(m+n+q+r)/2,  (o+p+s+t)/2]

Channel 1 (LH subband):
Frame 1: [(e-f+i-j)/2,  (g-h+k-l)/2]
         [(m-n+q-r)/2,  (o-p+s-t)/2]

Channel 2 (HL subband):
Frame 1: [(e+f-i-j)/2,  (g+h-k-l)/2]
         [(m+n-q-r)/2,  (o+p-s-t)/2]

Channel 3 (HH subband):
Frame 1: [(e-f-i+j)/2,  (g-h-k+l)/2]
         [(m-n-q-r)/2,  (o-p-s+t)/2]
```

(Same structure for Frame 2 with u, v, w, x, y, z, a, b, c, d, e, f, g, h, i, j)