# SoK: Unifying Cybersecurity and Cybersafety of Multimodal Foundation Models with an Information Theory Approach
Multimodal foundation models (MFMs) combine diverse data modalities, driving advancements in AI across various applications but introducing unique safety and security challenges. This study unifies cybersafety and cybersecurity in MFMs, identifying key threats through a taxonomy grounded in information theory. By analyzing threats via channel capacity, signal, noise, and bandwidth, we offer a novel framework to integrate model safety and system security. Our analysis highlights gaps in existing defenses, such as insufficient protection for cross-modality alignment and a lack of systematic defense strategies. This work provides actionable insights to enhance the robustness and reliability of MFMs.

<p align="center">
  <img src="sok.png" alt="overview" width="400">
  <br>
  <em>An overview of our SoK on unifying cybersecurity and cybersafety of multimodal foundation models.</em>
</p>

## Multimodal Foundation Models
In unimodal learning, the model maps input features to output labels within a discrete feature space, focusing on patterns within one data type. It extracts features, converts them into vectors, and then learns the mapping between vectors and labels.

In contrast, multimodal learning involves mapping continuous feature spaces across different modalities, which can be understood as extending the discrete output space of unimodal learning into a continuous space. Instead of directly mapping the spaces, multimodal models create unified representations in an alignment space, linking feature spaces from different data types. 

<p align="center">
  <img src="single_multi_ml.png" alt="single_multi" width="400">
  <br>
  <em>An illustration of single- and multimodal learning.</em>
</p>

<p align="center">
  <img src="models.PNG" alt="models" width="500">
  <br>
  <em>Examples of multimodal large models.</em>
</p>

## Unifying Security and Safety in MFMs

A machine learning model can be viewed as a channel for information transmission, where information flows from input samples through the model and ultimately to the outputs, further propagating to other components within a system.

Building on this perspective, information theory provides a robust framework for analyzing the transmission, processing, and fusion of information in multimodal models. 

Specifically, we adapt the Shannon–Hartley theorem, which quantifies the maximum rate at which information can be transmitted over a communication channel (e.g., the model) subject to noise (e.g., threats), for analyzing multimodal safety and security.

\[
C = B \log_2 \left( 1 + \frac{S}{N} \right)
\]

where:
- \(C\): Channel capacity -> represents the model's ability to effectively learn from and predict based on multimodal inputs, which is essential for assessing how well the model transmits meaningful information.
- \(S\): Signal power -> refers to the meaningful information that can be extracted from various modalities, such as textual features from documents, visual features from images, or auditory features from sound clips.
- \(N\): Noise power -> refers to any irrelevant or disruptive information that can distort the intended signal, which can stem from sources such as data collection errors, sensor inaccuracies, or deliberate perturbations injected into different modalities.
- \(B\): Bandwidth -> refers to the capacity for transmitting effective information between agents or system components, determining how information can be exchanged, which directly impacts the performance and responsiveness of each agent within the system. 









