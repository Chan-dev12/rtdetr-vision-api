# Dataset audit

data: `data/processed/data.yaml`

## train

- images: 5502 (background / no boxes: 86)
- boxes per image: mean 5.64, max 139
- most common resolutions: 1280x853 (1912), 853x1280 (1685), 1280x854 (543)

| class | instances | share | small | medium | large |
|---|---:|---:|---:|---:|---:|
| person | 9308 | 30.0% | 425 | 811 | 8072 |
| head | 8111 | 26.1% | 1145 | 1615 | 5351 |
| helmet | 652 | 2.1% | 247 | 205 | 200 |
| safety-vest | 326 | 1.0% | 71 | 106 | 149 |
| hands | 10763 | 34.7% | 1371 | 4746 | 4646 |
| gloves | 1896 | 6.1% | 407 | 751 | 738 |

## val

- images: 971 (background / no boxes: 15)
- boxes per image: mean 5.82, max 140
- most common resolutions: 1280x853 (339), 853x1280 (323), 1280x854 (73)

| class | instances | share | small | medium | large |
|---|---:|---:|---:|---:|---:|
| person | 1753 | 31.0% | 142 | 141 | 1470 |
| head | 1441 | 25.5% | 173 | 281 | 987 |
| helmet | 121 | 2.1% | 48 | 40 | 33 |
| safety-vest | 107 | 1.9% | 59 | 24 | 24 |
| hands | 1863 | 33.0% | 181 | 872 | 810 |
| gloves | 365 | 6.5% | 95 | 121 | 149 |

## test

- images: 1620 (background / no boxes: 21)
- boxes per image: mean 5.65, max 78
- most common resolutions: 1280x853 (557), 853x1280 (476), 1280x854 (151)

| class | instances | share | small | medium | large |
|---|---:|---:|---:|---:|---:|
| person | 2734 | 29.9% | 77 | 225 | 2432 |
| head | 2427 | 26.5% | 308 | 501 | 1618 |
| helmet | 154 | 1.7% | 38 | 58 | 58 |
| safety-vest | 97 | 1.1% | 17 | 29 | 51 |
| hands | 3212 | 35.1% | 428 | 1409 | 1375 |
| gloves | 529 | 5.8% | 110 | 201 | 218 |

Size bins are COCO's (<32² px small, <96² px medium) on original-resolution pixels.