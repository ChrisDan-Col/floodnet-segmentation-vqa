# VQA comparison on the test set (shared rows, n=678)

| approach | overall | Simple_C | Complex_C | Yes/No | Condition |
|---|---|---|---|---|---|
| lora | 0.909 | 0.745 | 0.757 | 0.992 | 0.968 |
| rule | 0.873 | 0.745 | 0.757 | 0.992 | 0.899 |
| vitbert_facts | 0.754 | 0.327 | 0.320 | 0.892 | 0.951 |
| vitbert | 0.739 | 0.265 | 0.282 | 0.908 | 0.945 |
| vilt_facts | 0.733 | 0.255 | 0.272 | 0.854 | 0.960 |
| vilt | 0.723 | 0.296 | 0.233 | 0.862 | 0.937 |
| prompt | 0.674 | 0.694 | 0.757 | 0.408 | 0.744 |
| clip | 0.550 | 0.000 | 0.039 | 0.538 | 0.862 |
