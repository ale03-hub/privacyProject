# Experiments summary

54 rows (31 configurations), total time 568s.

Risk is measured on the release (80% of the population) against an external dataset (50% of the population, LIMIT_BAL in 50k bands). Utility = 1 - il_total.

## Risk of the original data (identifiers removed)

| scenario | uniqueness | det_linkage_rate | dist_linkage_rate | attr_disclosure_rate | auc_full | auc_qi |
|---|---|---|---|---|---|---|
| QI | 0.0061 | 0.0037 | 0.0264 | 0.0229 | 0.7242 | 0.6273 |
| QI_EXT | 0.2022 | 0.0452 | 0.1297 | 0.1534 | 0.7242 | 0.6273 |

## Chosen compromise

Highest utility among the configurations with `dist_linkage_rate` below the threshold.

### QI (dist_linkage_rate < 0.01)

**recode_age_5** (global_recoding: col=AGE, width=5)

|  | uniqueness | det_linkage | dist_linkage | utility | auc_full | auc_qi |
|---|---|---|---|---|---|---|
| original | 0.0061 | 0.0037 | 0.0264 | 1.0 | 0.7242 | 0.6273 |
| recode_age_5 | 0.0008 | 0.0004 | 0.0074 | 0.968 | 0.7227 | 0.6245 |
| change | -0.0054 | -0.0033 | -0.019 | -0.032 | -0.0015 | -0.0028 |

### QI_EXT (dist_linkage_rate < 0.05)

**recode_age_5** (global_recoding: col=AGE, width=5)

|  | uniqueness | det_linkage | dist_linkage | utility | auc_full | auc_qi |
|---|---|---|---|---|---|---|
| original | 0.2022 | 0.0452 | 0.1297 | 1.0 | 0.7242 | 0.6273 |
| recode_age_5 | 0.0598 | 0.0109 | 0.0447 | 0.968 | 0.7227 | 0.6245 |
| change | -0.1425 | -0.0343 | -0.085 | -0.032 | -0.0015 | -0.0028 |

## All configurations

uniq / det / dist = uniqueness and deterministic / distance-based linkage rate; QI and EXT are the two attack scenarios; `-` = not evaluated in that scenario.

| config | uniq QI | det QI | dist QI | uniq EXT | det EXT | dist EXT | utility | auc_qi |
|---|---|---|---|---|---|---|---|---|
| original | 0.0061 | 0.0037 | 0.0264 | 0.2022 | 0.0452 | 0.1297 | 1.000 | 0.6273 |
| no_data | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.000 | 0.5000 |
| recode_age_5 | 0.0008 | 0.0004 | 0.0074 | 0.0597 | 0.0109 | 0.0447 | 0.968 | 0.6245 |
| recode_age_10 | 0.0004 | 0.0002 | 0.0034 | 0.0355 | 0.0065 | 0.0277 | 0.935 | 0.6264 |
| recode_age_3bands | 0.0000 | 0.0000 | 0.0017 | 0.0231 | 0.0037 | 0.0155 | 0.872 | 0.6257 |
| topcode_a | 0.0048 | 0.0027 | 0.0238 | 0.1965 | 0.0379 | 0.1242 | 0.998 | 0.6268 |
| topcode_b | 0.0035 | 0.0019 | 0.0195 | 0.1314 | 0.0215 | 0.0884 | 0.962 | 0.6270 |
| generalize_EDU | 0.0029 | 0.0015 | 0.0169 | 0.1327 | 0.0266 | 0.0900 | 0.886 | 0.6224 |
| generalize_MAR | 0.0031 | 0.0022 | 0.0216 | 0.1939 | 0.0398 | 0.1235 | 0.984 | 0.6272 |
| generalize_EDU_MAR | 0.0010 | 0.0007 | 0.0129 | 0.1242 | 0.0217 | 0.0839 | 0.871 | 0.6223 |
| local_supp_t2_QI | 0.0000 | 0.0000 | 0.0227 | - | - | - | 0.997 | 0.6224 |
| local_supp_t2_QI_EXT | - | - | - | 0.0000 | 0.0000 | 0.0743 | 0.947 | 0.6116 |
| local_supp_t5_QI | 0.0000 | 0.0000 | 0.0168 | - | - | - | 0.989 | 0.6222 |
| local_supp_t5_QI_EXT | - | - | - | 0.0000 | 0.0000 | 0.0328 | 0.868 | 0.5852 |
| local_supp_t10_QI | 0.0000 | 0.0000 | 0.0139 | - | - | - | 0.979 | 0.6220 |
| local_supp_t10_QI_EXT | - | - | - | 0.0000 | 0.0000 | 0.0163 | 0.814 | 0.5611 |
| pram_0.9 | 0.0070 | 0.0030 | 0.0197 | 0.2181 | 0.0343 | 0.1019 | 0.838 | 0.6254 |
| pram_0.7 | 0.0067 | 0.0017 | 0.0120 | 0.2389 | 0.0191 | 0.0611 | 0.683 | 0.6222 |
| pram_0.5 | 0.0071 | 0.0012 | 0.0059 | 0.2507 | 0.0099 | 0.0330 | 0.619 | 0.6205 |
| noise_0.05 | 0.0063 | 0.0029 | 0.0195 | 0.9937 | 0.0268 | 0.0985 | 0.983 | 0.6270 |
| noise_0.1 | 0.0051 | 0.0012 | 0.0125 | 0.9797 | 0.0131 | 0.0595 | 0.962 | 0.6264 |
| noise_0.2 | 0.0067 | 0.0008 | 0.0075 | 0.9561 | 0.0058 | 0.0296 | 0.924 | 0.6239 |
| rankswap_5 | 0.0063 | 0.0007 | 0.0062 | 0.2130 | 0.0046 | 0.0265 | 0.942 | 0.6270 |
| rankswap_10 | 0.0069 | 0.0002 | 0.0028 | 0.2177 | 0.0016 | 0.0098 | 0.894 | 0.6248 |
| rankswap_20 | 0.0069 | 0.0001 | 0.0016 | 0.2270 | 0.0004 | 0.0039 | 0.810 | 0.6154 |
| microagg_3 | 0.0056 | 0.0024 | 0.0195 | 0.2589 | 0.0263 | 0.0869 | 0.960 | 0.6211 |
| microagg_5 | 0.0052 | 0.0019 | 0.0166 | 0.2691 | 0.0189 | 0.0665 | 0.940 | 0.6177 |
| microagg_10 | 0.0048 | 0.0016 | 0.0132 | 0.2361 | 0.0141 | 0.0515 | 0.907 | 0.6127 |
| combo_recode_gen_locsup_QI | 0.0000 | 0.0000 | 0.0014 | - | - | - | 0.806 | 0.6213 |
| combo_recode_gen_locsup_QI_EXT | - | - | - | 0.0000 | 0.0000 | 0.0082 | 0.785 | 0.6216 |
| combo_noise_pram | 0.0067 | 0.0011 | 0.0103 | 0.9800 | 0.0101 | 0.0482 | 0.800 | 0.6245 |
