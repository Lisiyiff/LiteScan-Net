
LiteScan-Net: A Lightweight Scanning Network and A Large-Scale Dataset for Cropland Change Detection







Abstract:
Aiming at the dual dilemma in high-resolution cropland change detection, where CNNs are constrained by limited local receptive fields and Transformers suffer from heavy computational costs, we propose LiteScan-Net, a lightweight and robust net-work architecture incorporating scanning principles from state-space modeling. The network innovatively introduces the Multi-Directional Global Scanning (MDGS) mechanism as an efficient engineering surrogate, which simulates the selective scan-ning process using large-kernel 1D convolutions. This achieves global context model-ing with linear complexity while avoiding the hardware limitations imposed by recur-rent computations. Based on this mechanism, a three-stage collaborative architecture is constructed: the Coordinate-Aware Feature Purification (CAFP) module is designed to mitigate shallow phenological noise via coordinate sensitivity; the Context Differ-ence Verification (CDV) module aims to alleviate pseudo-changes caused by registra-tion errors through global alignment; and the State-Space Guided Refinement (SSGR) module promotes the generation of change with have precise boundaries and compact interiors. To verify the model generalization, we construct a Massive Specialized Cropland Change Detection dataset named MSCC, which exhibits significant cross-scale characteristics. Experimental results demonstrate that LiteScan-Net achieves state-of-the-art (SOTA) performance across the CLCD, Hi-CNA, and MSCC datasets, with F1-scores of 79.43%, 84.82%, and 89.62%, respectively. With a low computational cost of only 1.78 G and a real-time inference speed of 37.9 FPS, LiteScan-Net demon-strates high potential for future deployment on resource-constrained edge devices.

<img width="2637" height="820" alt="12" src="https://github.com/user-attachments/assets/fa7d77ec-76d4-40b2-ab03-9c893f54c72d" />





