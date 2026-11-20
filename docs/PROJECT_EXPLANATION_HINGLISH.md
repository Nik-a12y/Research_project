# Project Explanation (Hinglish)

## Ye project kya hai? (simple)
Aankh ke pichhe retina hota hai. Diabetes me uski chhoti blood vessels kharab hoti hain (Diabetic Retinopathy, DR).
OCT-A scan in vessels ki photo deta hai. Hum dekhte hain: **sirf image ke pattern** se classification behtar hota hai,
ya **vessels kaise branch/connect hote hain (topology)** wali extra info add karne se behtar hota hai?

Ye ek **controlled experiment** hai: Model A (baseline) vs Model B (baseline + topology). Dono same data, same split.
Claim sirf itna: "topology features help / don't help / sirf kuch classes me help". Medical diagnosis ka claim nahi.

## Topology features kya hote hain?
- Branch points (jahan vessel split hoti hai), endpoints (jahan vessel toot gayi)
- Connected components (kitne alag tukde; zyada = vessels toote hue)
- Largest component fraction (network kitna connected hai)
- Euler number (components minus holes)
- Avascular gap (sabse bada khali area, FAZ/nonperfusion jaisa)

## Step by step pipeline  
1. **Data**: OCTA-500 download karo (ya demo ke liye synthetic data, `src/synthetic.py`).
2. **Vessel segmentation**: Sato ridge filter + Otsu threshold -> mask -> skeleton (`src/vessels.py`).
3. **Baseline features**: intensity, histogram, 8x8 grid, vessel density. Connectivity info NAHI (`src/features.py`).
4. **Topology features**: skeleton se 10 features (`src/features.py`).
5. **Do models**: RandomForest on baseline vs baseline+topology.
6. **Evaluation**: accuracy, precision, recall, macro-F1, confusion matrix, 5-fold x3 repeated CV, paired difference.
7. **Report**: `outputs/` folder me results.json + plots.

## Chalane ka tareeka
```
pip install -r requirements.txt
python main.py                                   # demo (synthetic)
python main.py --real IMG_DIR LABELS.csv         # real data
```
LABELS.csv columns: `id,label`.

## IMPORTANT sach
- Demo data **synthetic** hai. Uske results ka koi scientific matlab nahi. Sirf code test karne ke liye.
- OCTA-500 me labels **disease type** ke hain (jaise NORMAL, DR, AMD...), **DR severity (mild/moderate/severe) nahi**.
  Isliye real task shayad "Normal vs DR vs other" hoga. Label file khud inspect karo.
- `src/real_data.py` ko real folder structure dekh kar adjust karna padega (maine real dataset download nahi kiya).
- Literature gap abhi **provisional** hai; supervisor ke saath verify karo.
