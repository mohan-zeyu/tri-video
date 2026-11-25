# Text or noise seed? Four cells, six comparisons

Four cells, each layer 15, denoising step 1, conditional branch, head 1 (heads count from 0), frozen bf16 teacher: T0N0 = `d0000-L15-S01-conditional-H01`; T6N6 = `d0006-L15-S01-conditional-H01`; T0N6 = `t0000n0006-L15-S01-conditional-H01`; T6N0 = `t0006n0000-L15-S01-conditional-H01`. T0 / T6 = the stored text embeddings of prompt d0000 / d0006; N0 / N6 = the stored step-1 entering state (the initial noise) of d0000 / d0006. Crossed cells use one prompt's text with the other prompt's noise.

Scope of every number: a centred share uses every one of the 10,800 tokens as a query and as a key of ONE cell (`ctr_att_abs_share`). A map correlation is for ONE query token: the Pearson correlation over all keys between the pair's score map in the two cells of a comparison; `median of 27` is its median over a 3 × 3 × 3 lattice of 27 query tokens. Distributions below run over pairs (linear-interpolation quantiles). Nothing is averaged over cells, comparisons or pairs; the two comparisons of a type stand side by side.

## Per comparison

| # | type | A vs B | share correlation across 64 pairs | largest difference of one pair's centred share |
|---:|---|---|---:|---:|
| 1 | same noise, different text | T0N0 vs T6N0 | 0.8312 | 2.25 pp (h26: 5.09% vs 7.34%) |
| 2 | same noise, different text | T0N6 vs T6N6 | 0.8718 | 2.20 pp (h26: 5.17% vs 7.37%) |
| 3 | same text, different noise | T0N0 vs T0N6 | 0.9901 | 0.34 pp (h25: 2.09% vs 2.42%) |
| 4 | same text, different noise | T6N0 vs T6N6 | 0.9957 | 0.46 pp (h23: 2.26% vs 1.80%) |
| 5 | both differ | T0N0 vs T6N6 | 0.8555 | 2.29 pp (h26: 5.09% vs 7.37%) |
| 6 | both differ | T0N6 vs T6N0 | 0.8497 | 2.17 pp (h26: 5.17% vs 7.34%) |

## Distribution over pairs of the median-of-27-queries map correlation

Pairs above 1/64 are those whose centred share is above the equal split in the comparison's first cell (A); the count is in brackets. Low-frequency pairs: t6–t21, h29–h42, w51–w63 (43 pairs).

| # | type | A vs B | pairs | min | q05 | q25 | median | q75 | q95 | max |
|---:|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| 1 | same noise, different text | T0N0 vs T6N0 | all 64 pairs | -0.136 | +0.190 | +0.293 | +0.411 | +0.514 | +0.713 | +0.919 |
| 1 | same noise, different text | T0N0 vs T6N0 | pairs above 1/64 in T0N0 (26) | +0.198 | +0.252 | +0.359 | +0.448 | +0.506 | +0.677 | +0.919 |
| 1 | same noise, different text | T0N0 vs T6N0 | 43 low-frequency pairs | -0.136 | +0.147 | +0.253 | +0.364 | +0.441 | +0.508 | +0.648 |
| 2 | same noise, different text | T0N6 vs T6N6 | all 64 pairs | +0.054 | +0.175 | +0.287 | +0.410 | +0.515 | +0.765 | +0.900 |
| 2 | same noise, different text | T0N6 vs T6N6 | pairs above 1/64 in T0N6 (26) | +0.248 | +0.303 | +0.378 | +0.457 | +0.556 | +0.747 | +0.900 |
| 2 | same noise, different text | T0N6 vs T6N6 | 43 low-frequency pairs | +0.054 | +0.156 | +0.258 | +0.344 | +0.430 | +0.565 | +0.644 |
| 3 | same text, different noise | T0N0 vs T0N6 | all 64 pairs | -0.037 | +0.009 | +0.103 | +0.149 | +0.276 | +0.554 | +0.803 |
| 3 | same text, different noise | T0N0 vs T0N6 | pairs above 1/64 in T0N0 (26) | +0.014 | +0.034 | +0.117 | +0.165 | +0.274 | +0.667 | +0.803 |
| 3 | same text, different noise | T0N0 vs T0N6 | 43 low-frequency pairs | -0.004 | +0.014 | +0.080 | +0.131 | +0.175 | +0.271 | +0.310 |
| 4 | same text, different noise | T6N0 vs T6N6 | all 64 pairs | -0.073 | -0.014 | +0.105 | +0.181 | +0.293 | +0.718 | +0.866 |
| 4 | same text, different noise | T6N0 vs T6N6 | pairs above 1/64 in T6N0 (27) | -0.023 | +0.002 | +0.115 | +0.187 | +0.289 | +0.749 | +0.866 |
| 4 | same text, different noise | T6N0 vs T6N6 | 43 low-frequency pairs | -0.073 | -0.015 | +0.094 | +0.131 | +0.191 | +0.284 | +0.354 |
| 5 | both differ | T0N0 vs T6N6 | all 64 pairs | -0.123 | -0.040 | +0.036 | +0.077 | +0.142 | +0.566 | +0.714 |
| 5 | both differ | T0N0 vs T6N6 | pairs above 1/64 in T0N0 (26) | -0.049 | -0.045 | +0.041 | +0.087 | +0.132 | +0.543 | +0.704 |
| 5 | both differ | T0N0 vs T6N6 | 43 low-frequency pairs | -0.049 | -0.041 | +0.022 | +0.049 | +0.091 | +0.138 | +0.196 |
| 6 | both differ | T0N6 vs T6N0 | all 64 pairs | -0.050 | -0.023 | +0.029 | +0.085 | +0.207 | +0.633 | +0.832 |
| 6 | both differ | T0N6 vs T6N0 | pairs above 1/64 in T0N6 (26) | -0.036 | -0.014 | +0.052 | +0.098 | +0.204 | +0.553 | +0.832 |
| 6 | both differ | T0N6 vs T6N0 | 43 low-frequency pairs | -0.050 | -0.024 | +0.018 | +0.052 | +0.098 | +0.158 | +0.281 |

- Comparison 1 (T0N0 vs T6N0): pairs above 1/64 in T0N0: t11, t12, t13, t15, t16, t18, t20, t21, h25, h26, h28, h29, h30, h31, h34, h35, h36, h37, h38, h41, w47, w49, w57, w58, w59, w62.

- Comparison 2 (T0N6 vs T6N6): pairs above 1/64 in T0N6: t11, t12, t13, t14, t15, t16, t18, t20, t21, h25, h26, h28, h29, h30, h31, h34, h35, h36, h37, h38, h41, w48, w49, w57, w59, w62.

- Comparison 3 (T0N0 vs T0N6): pairs above 1/64 in T0N0: t11, t12, t13, t15, t16, t18, t20, t21, h25, h26, h28, h29, h30, h31, h34, h35, h36, h37, h38, h41, w47, w49, w57, w58, w59, w62.

- Comparison 4 (T6N0 vs T6N6): pairs above 1/64 in T6N0: t11, t12, t15, t16, t17, t18, t20, t21, h23, h24, h25, h26, h28, h30, h31, h34, h35, h36, h37, h42, w49, w55, w57, w58, w59, w61, w62.

- Comparison 5 (T0N0 vs T6N6): pairs above 1/64 in T0N0: t11, t12, t13, t15, t16, t18, t20, t21, h25, h26, h28, h29, h30, h31, h34, h35, h36, h37, h38, h41, w47, w49, w57, w58, w59, w62.

- Comparison 6 (T0N6 vs T6N0): pairs above 1/64 in T0N6: t11, t12, t13, t14, t15, t16, t18, t20, t21, h25, h26, h28, h29, h30, h31, h34, h35, h36, h37, h38, h41, w48, w49, w57, w59, w62.

## Per pair: median-of-27-queries map correlation in each comparison

| pair | selected by | 1. T0N0 vs T6N0 | 2. T0N6 vs T6N6 | 3. T0N0 vs T0N6 | 4. T6N0 vs T6N6 | 5. T0N0 vs T6N6 | 6. T0N6 vs T6N0 |
|---|---|---:|---:|---:|---:|---:|---:|
| t0 | shared22, fixed_high, shared32 | +0.822 | +0.832 | +0.558 | +0.615 | +0.633 | +0.635 |
| t1 | shared22, fixed_high, shared32 | +0.880 | +0.876 | +0.738 | +0.728 | +0.714 | +0.722 |
| t2 | shared22, fixed_high, shared32 | +0.573 | +0.506 | +0.150 | +0.597 | -0.123 | +0.484 |
| t3 | shared22, fixed_high, shared32 | +0.531 | +0.516 | +0.125 | +0.085 | +0.054 | +0.091 |
| t4 | shared22, fixed_high, shared32 | +0.548 | +0.594 | +0.192 | +0.214 | +0.118 | +0.149 |
| t5 | shared22, fixed_high, shared32 | +0.400 | +0.270 | +0.137 | +0.206 | +0.044 | +0.087 |
| t6 | shared22, fixed_high, shared32 | +0.274 | +0.179 | +0.099 | +0.128 | +0.038 | -0.018 |
| t7 | shared22, fixed_high, shared32 | +0.367 | +0.180 | +0.043 | +0.043 | +0.017 | +0.023 |
| t8 | shared32 | +0.231 | +0.261 | +0.008 | -0.073 | +0.007 | +0.023 |
| t9 | shared22, shared32 | +0.193 | +0.218 | +0.117 | +0.127 | +0.036 | +0.010 |
| t10 | shared22, shared32 | +0.257 | +0.254 | +0.069 | +0.036 | +0.031 | -0.024 |
| t11 | — | +0.648 | +0.644 | +0.176 | +0.182 | +0.156 | +0.158 |
| t12 | — | +0.257 | +0.248 | +0.176 | +0.117 | +0.046 | +0.061 |
| t13 | — | +0.316 | +0.298 | +0.021 | -0.016 | -0.049 | +0.009 |
| t14 | — | +0.405 | +0.426 | +0.146 | +0.142 | +0.088 | +0.088 |
| t15 | — | +0.357 | +0.389 | +0.135 | +0.123 | +0.040 | +0.064 |
| t16 | — | +0.250 | +0.317 | +0.195 | +0.235 | +0.107 | +0.155 |
| t17 | — | +0.189 | +0.054 | +0.065 | +0.287 | +0.049 | +0.067 |
| t18 | — | +0.398 | +0.515 | +0.073 | +0.149 | -0.002 | +0.046 |
| t19 | shared32 | +0.374 | +0.338 | +0.158 | +0.183 | +0.138 | +0.059 |
| t20 | — | +0.374 | +0.401 | +0.244 | +0.257 | +0.055 | +0.219 |
| t21 | — | +0.450 | +0.357 | +0.110 | +0.247 | +0.087 | +0.057 |
| h22 | fixed_high, shared32 | +0.453 | +0.516 | -0.029 | +0.346 | +0.315 | +0.203 |
| h23 | shared22, fixed_high, shared32 | +0.624 | +0.636 | +0.326 | +0.733 | +0.473 | +0.513 |
| h24 | shared22, fixed_high, shared32 | +0.608 | +0.766 | +0.505 | +0.658 | +0.567 | +0.644 |
| h25 | fixed_high | +0.678 | +0.707 | +0.534 | +0.756 | +0.498 | +0.618 |
| h26 | fixed_high | +0.919 | +0.900 | +0.803 | +0.866 | +0.704 | +0.832 |
| h27 | shared22, fixed_high, shared32 | +0.720 | +0.541 | +0.487 | +0.440 | +0.456 | +0.619 |
| h28 | fixed_high | +0.675 | +0.761 | +0.712 | +0.595 | +0.558 | +0.357 |
| h29 | shared32 | +0.479 | +0.440 | +0.283 | +0.354 | +0.196 | +0.281 |
| h30 | shared32 | +0.548 | +0.511 | +0.132 | +0.290 | +0.122 | +0.083 |
| h31 | — | +0.486 | +0.475 | +0.113 | +0.131 | +0.013 | +0.026 |
| h32 | — | +0.143 | +0.176 | +0.079 | +0.080 | +0.034 | +0.016 |
| h33 | — | +0.437 | +0.421 | +0.082 | +0.127 | +0.039 | +0.062 |
| h34 | — | +0.471 | +0.570 | +0.154 | +0.194 | +0.103 | +0.136 |
| h35 | — | +0.198 | +0.375 | +0.149 | +0.104 | +0.087 | +0.108 |
| h36 | — | +0.500 | +0.479 | +0.194 | -0.009 | -0.041 | +0.033 |
| h37 | — | +0.364 | +0.419 | +0.014 | +0.223 | -0.033 | +0.056 |
| h38 | — | +0.307 | +0.318 | +0.108 | +0.142 | -0.046 | -0.022 |
| h39 | shared32 | +0.274 | +0.138 | -0.004 | +0.134 | +0.021 | +0.006 |
| h40 | shared32 | +0.463 | +0.434 | +0.140 | +0.227 | +0.114 | +0.078 |
| h41 | — | +0.299 | +0.344 | +0.247 | +0.180 | +0.135 | +0.051 |
| h42 | — | +0.200 | +0.229 | +0.163 | +0.187 | +0.067 | +0.045 |
| w43 | shared22, fixed_high, shared32 | +0.331 | +0.262 | -0.037 | -0.015 | +0.075 | -0.023 |
| w44 | shared22, fixed_high, shared32 | +0.449 | +0.361 | +0.317 | +0.103 | +0.252 | +0.122 |
| w45 | shared22, fixed_high, shared32 | +0.567 | +0.514 | +0.355 | +0.301 | +0.358 | +0.367 |
| w46 | shared22, fixed_high, shared32 | +0.438 | +0.462 | +0.300 | +0.328 | +0.208 | +0.237 |
| w47 | fixed_high, shared32 | +0.363 | +0.637 | +0.330 | +0.467 | +0.115 | +0.319 |
| w48 | fixed_high, shared32 | +0.676 | +0.484 | +0.533 | +0.471 | +0.452 | +0.315 |
| w49 | fixed_high | +0.581 | +0.572 | +0.455 | +0.325 | +0.326 | +0.333 |
| w50 | — | +0.455 | +0.289 | +0.212 | +0.207 | +0.073 | +0.131 |
| w51 | shared22, shared32 | +0.194 | +0.281 | +0.060 | +0.036 | +0.134 | +0.036 |
| w52 | shared22, shared32 | +0.393 | +0.449 | +0.105 | +0.087 | +0.094 | +0.052 |
| w53 | shared22, shared32 | +0.368 | +0.342 | +0.019 | +0.127 | -0.002 | +0.030 |
| w54 | — | +0.123 | +0.174 | +0.186 | +0.200 | +0.033 | -0.007 |
| w55 | — | +0.417 | +0.331 | +0.115 | +0.071 | -0.037 | +0.137 |
| w56 | shared22, shared32 | +0.259 | +0.318 | +0.221 | +0.250 | +0.087 | +0.134 |
| w57 | — | +0.465 | +0.473 | +0.129 | -0.023 | +0.098 | -0.036 |
| w58 | — | +0.423 | +0.360 | +0.091 | +0.027 | +0.058 | +0.019 |
| w59 | — | +0.509 | +0.570 | +0.131 | +0.101 | +0.044 | +0.119 |
| w60 | shared22, shared32 | +0.230 | +0.281 | +0.134 | +0.179 | +0.022 | +0.112 |
| w61 | — | +0.305 | +0.347 | +0.174 | +0.163 | +0.072 | -0.004 |
| w62 | — | +0.445 | +0.425 | +0.310 | +0.113 | +0.050 | +0.110 |
| w63 | shared32 | -0.136 | +0.154 | +0.273 | +0.106 | +0.079 | -0.050 |

Columns 1–2: same noise, different text. 3–4: same text, different noise. 5–6: both differ.
